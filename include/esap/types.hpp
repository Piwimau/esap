#ifndef ESAP_TYPES_HPP
#define ESAP_TYPES_HPP

#include <cstddef>
#include <cstdint>

namespace esap {

/** @brief An unsigned integer for raw memory access. */
using byte = std::byte;

/** @brief An 8-bit signed integer. */
using i8 = std::int8_t;

/** @brief An 8-bit unsigned integer. */
using u8 = std::uint8_t;

/** @brief A 16-bit signed integer. */
using i16 = std::int16_t;

/** @brief A 16-bit unsigned integer. */
using u16 = std::uint16_t;

/** @brief A 32-bit signed integer. */
using i32 = std::int32_t;

/** @brief A 32-bit unsigned integer. */
using u32 = std::uint32_t;

/** @brief A 64-bit signed integer. */
using i64 = std::int64_t;

/** @brief A 64-bit unsigned integer. */
using u64 = std::uint64_t;

/** @brief A signed integer capable of holding a pointer. */
using iptr = std::intptr_t;

/** @brief An unsigned integer capable of holding a pointer. */
using uptr = std::uintptr_t;

/** @brief A signed integer for sizes, counts and indices. */
using isize = std::ptrdiff_t;

/** @brief An unsigned integer for sizes, counts and indices. */
using usize = std::size_t;

/** @brief A 32-bit floating-point number. */
using f32 = float;

/** @brief A 64-bit floating-point number. */
using f64 = double;

}

#endif