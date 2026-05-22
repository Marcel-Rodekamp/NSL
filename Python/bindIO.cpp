#include <pybind11/pybind11.h>
namespace py = pybind11;
using namespace pybind11::literals;

namespace pybind11 {
    namespace detail {
        template <>
        struct type_caster<YAML::Node> {
        public:
            PYBIND11_TYPE_CASTER(YAML::Node, _("YAML::Node"));

            // Conversion from Python to C++
            bool load(handle src, bool) {
                std::string yaml_path = src.cast<std::string>();
                value = YAML::LoadFile(yaml_path);
                return true;
            }

            // Conversion from C++ to Python
            static handle cast(const YAML::Node& src, return_value_policy /* policy */, handle /* parent */) {
                std::stringstream ss;
                ss << src;
                return py::str(ss.str()).release();
            }
        };

        template <>
        struct type_caster<NSL::H5IO> {
        public:
            PYBIND11_TYPE_CASTER(NSL::H5IO, _("NSL::H5IO"));

            // Conversion from Python to C++
            bool load(handle src, bool) {
                //TODO add option for overwrite
                std::string filename = src.cast<std::string>();
                value = NSL::H5IO(filename);
                return true;
            }

            // Conversion from C++ to Python
            static handle cast(const NSL::H5IO& src, return_value_policy /* policy */, handle /* parent */) {
                throw std::runtime_error("Not implemented");
                // return py::str(src.filename()).release();
            }
        };
    }  // namespace detail
}  // namespace pybind11