#ifndef ESAP_WAV_HPP
#define ESAP_WAV_HPP

#include <memory>
#include <span>
#include <string>
#include "esap/audio-format.hpp"
#include "esap/types.hpp"

namespace esap {

/** @brief Represents a waveform audio file. */
class Wav final {
private:

    /** @brief The audio format of this waveform audio file. */
    AudioFormat _format;

    /**
     * @brief The audio samples, normalized to the range [`-1.0F`, `1.0F`] and
     * stored in a planar layout (i.e., all samples of an audio channel are
     * stored contiguously in memory).
     */
    std::unique_ptr<f32[]> _samples;

public:

    /**
     * @brief Initializes a new waveform audio file with the specified format
     * and samples.
     *
     * @param[in] format  The audio format of the waveform audio file.
     * @param[in] samples The audio samples, normalized to the range
     *                    [`-1.0F`, `1.0F`] and stored in a planar layout
     *                    (i.e., all samples of an audio channel are stored
     *                    contiguously in memory).
     */
    Wav(AudioFormat format, std::unique_ptr<f32[]> samples);

    /**
     * @brief Reads a waveform audio file from a specified path.
     *
     * @param[in] path The path to read the waveform audio file from.
     * @return The waveform audio file read from the specified path.
     * @throws `std::runtime_error`   Thrown when the file cannot be opened or
     *                                read from.
     * @throws `esap::InvalidWav`     Thrown when the file is not a valid
     *                                waveform audio file.
     * @throws `esap::UnsupportedWav` Thrown when the file has an unsupported
     *                                audio format (e.g., compressed audio).
     */
    static Wav read(const std::string& path);

    /**
     * @brief Returns the audio format.
     *
     * @return The audio format.
     */
    const AudioFormat& format() const noexcept;

    /**
     * @brief Returns a span of the audio samples.
     *
     * @return A span of the audio samples.
     */
    std::span<const f32> samples() const noexcept;

    /**
     * @brief Writes this waveform audio file to a specified path.
     *
     * @param[in] path The path to write this waveform audio file to.
     * @throws `std::runtime_error` Thrown when the file cannot be opened or
     *                              written to.
     */
    void write(const std::string& path) const;

};

}

#endif