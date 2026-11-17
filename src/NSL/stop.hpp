#ifndef NSL_STOP_HPP
#define NSL_STOP_HPP

/*! \file stop.hpp
 *  \brief Safe early stop on signals, e.g. SLURM's warning before the walltime.
 *
 *  NSL::init installs handlers for three signals:
 *   - SIGUSR1, the early warning (`#SBATCH --signal=USR1@600` sends it 600 s before the
 *     walltime): the current unit of work (trajectory, measured configuration) is finished
 *     and stored, then the run stops. NSL::Stop::requested() becomes true; the Markov chain
 *     generators and the propagator measurements check it at safe points.
 *   - SIGTERM (SLURM's kill at the walltime) and SIGINT (Ctrl-C): the run exits at once,
 *     unless an HDF5 write is in progress; then the write is finished and flushed first.
 *     Everything stored before is complete, because every write ends with a flush.
 *  A run that stopped early exits with NSL::Stop::exitCode (75, "temporary failure, try
 *  again" in sysexits.h) and continues where it stopped when the same command is run again.
 *
 *  Without a handler SIGUSR1 terminates a process, so `--signal=USR1@...` must only be used
 *  with executables that call NSL::init.
 */

#include <atomic>
#include <csignal>
#include <cstring>
#include <unistd.h>

namespace NSL::Stop {

//! Exit code of a run that stopped early on a signal and can be resumed (EX_TEMPFAIL).
inline constexpr int exitCode = 75;

static_assert(std::atomic<int>::is_always_lock_free, "NSL::Stop needs lock-free atomics in its signal handler");

namespace detail {
    // The handler only touches lock-free atomics and calls write/_exit (async-signal-safe).
    inline std::atomic<int> signal{0};     // signal that requested the stop, 0 if none
    inline std::atomic<int> terminate{0};  // SIGTERM/SIGINT arrived during an HDF5 write
    inline std::atomic<int> writing{0};    // nesting depth of HDF5 write sections

    template<std::size_t N>
    inline void say(const char (&msg)[N]) {
        const ssize_t r = ::write(STDERR_FILENO, msg, N - 1);
        (void) r;
    }

    inline void handler(int sig) {
        signal.store(sig);
        if (sig == SIGUSR1) {
            say("\nNSL: SIGUSR1 received; stopping after the current trajectory/configuration\n");
            return;
        }
        if (writing.load() == 0) {
            say("\nNSL: SIGTERM/SIGINT received; exiting now (no HDF5 write in progress)\n");
            ::_exit(exitCode);
        }
        terminate.store(1);
        say("\nNSL: SIGTERM/SIGINT received during an HDF5 write; exiting once it is flushed\n");
    }
} // namespace detail

//! True once a stop was requested by a signal.
inline bool requested() { return detail::signal.load() != 0; }

//! Name of the signal that requested the stop ("" if none).
inline const char * signalName() {
    switch (detail::signal.load()) {
        case SIGUSR1: return "SIGUSR1";
        case SIGTERM: return "SIGTERM";
        case SIGINT:  return "SIGINT";
        default:      return "";
    }
}

//! Install the handlers for SIGUSR1, SIGTERM and SIGINT (called by NSL::init).
inline void install() {
    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_handler = detail::handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESTART;
    sigaction(SIGUSR1, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGINT,  &sa, nullptr);
}

//! Marks an HDF5 write section. A SIGTERM/SIGINT arriving inside is deferred until the
//! outermost section ends; the code inside flushes the file before the section ends.
class WriteGuard {
    public:
    WriteGuard() { detail::writing.fetch_add(1); }
    ~WriteGuard() {
        if (detail::writing.fetch_sub(1) == 1 && detail::terminate.load()) {
            detail::say("NSL: HDF5 write flushed; exiting\n");
            ::_exit(exitCode);
        }
    }
    WriteGuard(const WriteGuard &) = delete;
    WriteGuard & operator=(const WriteGuard &) = delete;
};

} // namespace NSL::Stop

#endif // NSL_STOP_HPP
