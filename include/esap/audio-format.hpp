#ifndef ESAP_AUDIO_FORMAT_HPP
#define ESAP_AUDIO_FORMAT_HPP

#include "esap/types.hpp"

namespace esap {

/** @brief Represents the format of an audio signal. */
struct AudioFormat {

    /** @brief The number of audio channels. */
    u16 numChannels;

    /** @brief The number of bits per sample. */
    u16 bitsPerSample;

    /** @brief The sample rate. */
    u32 sampleRate;

    /** @brief The number of samples per audio channel. */
    u32 numFrames;

};

}

#endif