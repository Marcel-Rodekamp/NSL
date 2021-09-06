#ifndef NANOSYSTEMLIBRARY_ASSERT_HPP
#define NANOSYSTEMLIBRARY_ASSERT_HPP

#include <typeinfo>
#include <cassert>
#define assertm(exp, msg) assert(((void)msg, exp))


namespace NSL{

// adapted from https://stackoverflow.com/a/29671981
template<bool...> struct bool_pack;

template<bool... bs>
using all_true = std::is_same<bool_pack<bs..., true>, bool_pack<true, bs...>>;

// Compile time conversion check from all parameter pack members to one
// can be used in combination with static_assert:
// static_assert(are_all_convertible<To, Froms...>::value, "Some msg");
template<class R, class... Ts>
using all_convertible = all_true<std::is_convertible<Ts, R>::value...>;

}

#endif //NANOSYSTEMLIBRARY_ASSERT_HPP