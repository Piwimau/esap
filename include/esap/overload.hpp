#ifndef ESAP_OVERLOAD_HPP
#define ESAP_OVERLOAD_HPP

namespace esap {

/** @brief A helper to allow overloading using lambdas. */
template<typename... Ts>
struct Overload : Ts... {
    using Ts::operator()...;
};

}

#endif