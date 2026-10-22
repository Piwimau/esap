#ifndef ESAP_FILTER_HPP
#define ESAP_FILTER_HPP

#include <variant>
#include "esap/types.hpp"

namespace esap {

/** @brief A lowpass filter. */
struct LowpassFilter {

    /** @brief The cutoff frequency (in Hz). */
    f32 cutoff;

};

/** @brief A highpass filter. */
struct HighpassFilter {

    /** @brief The cutoff frequency (in Hz). */
    f32 cutoff;

};

/** @brief A bandpass filter. */
struct BandpassFilter {

    /** @brief The lower cutoff frequency (in Hz). */
    f32 lowCutoff;

    /** @brief The upper cutoff frequency (in Hz). */
    f32 highCutoff;

};

/** @brief A bandstop filter. */
struct BandstopFilter {

    /** @brief The lower cutoff frequency (in Hz). */
    f32 lowCutoff;

    /** @brief The upper cutoff frequency (in Hz). */
    f32 highCutoff;

};

/** @brief A filter. */
using Filter = std::variant<
    LowpassFilter,
    HighpassFilter,
    BandpassFilter,
    BandstopFilter
>;

}

#endif