#ifndef ESAP_FILTER_HPP
#define ESAP_FILTER_HPP

#include <variant>
#include "esap/types.hpp"

namespace esap {

/** @brief Represents a lowpass filter. */
struct LowpassFilter {

    /** @brief The cutoff frequency (in Hz). */
    f32 cutoff;

};

/** @brief Represents a highpass filter. */
struct HighpassFilter {

    /** @brief The cutoff frequency (in Hz). */
    f32 cutoff;

};

/** @brief Represents a bandpass filter. */
struct BandpassFilter {

    /** @brief The lower cutoff frequency (in Hz). */
    f32 lowCutoff;

    /** @brief The upper cutoff frequency (in Hz). */
    f32 highCutoff;

};

/** @brief Represents a bandstop filter. */
struct BandstopFilter {

    /** @brief The lower cutoff frequency (in Hz). */
    f32 lowCutoff;

    /** @brief The upper cutoff frequency (in Hz). */
    f32 highCutoff;

};

/** @brief Represents a filter. */
using Filter = std::variant<
    LowpassFilter,
    HighpassFilter,
    BandpassFilter,
    BandstopFilter
>;

}

#endif