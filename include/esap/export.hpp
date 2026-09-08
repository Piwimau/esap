#ifndef ESAP_EXPORT_HPP
#define ESAP_EXPORT_HPP

#include <complex>
#include <span>
#include <string>
#include "esap/types.hpp"

namespace esap {

/**
 * @brief Exports a magnitude spectrum to a CSV file with a specified path.
 *
 * The resulting CSV file will contain three columns for the channel index, the
 * frequency (in Hz), and the magnitude (in dB) of each frequency bin. The
 * magnitude is computed as the average magnitude of the corresponding frequency
 * bin across all STFT frames, and is normalized by the window size.
 *
 * @param[in] numChannels The number of audio channels.
 * @param[in] sampleRate  The sample rate of the audio signal (in Hz).
 * @param[in] numFrames   The number of STFT frames (i.e., the number of
 *                        windowed FFTs computed per audio channel).
 * @param[in] windowSize  The size of the window used for the STFT (i.e., the
 *                        number of audio samples in each window).
 * @param[in] bins        A read-only view of the bins (i.e., the complex
 *                        frequency-domain representation of the audio signal).
 * @param[in] path        The path to the CSV file to export the magnitude
 *                        spectrum to.
 * @throws `std::exception` Thrown when any error occurs while exporting the
 *                          magnitude spectrum.
 */
void export_spectrum(
    usize numChannels,
    usize sampleRate,
    usize numFrames,
    usize windowSize,
    std::span<const std::complex<f32>> bins,
    const std::string& path
);

/**
 * @brief Exports a magnitude spectrogram to a CSV file with a specified path.
 *
 * The resulting CSV file will contain a header row with the channel index and
 * time (in seconds) of each STFT frame, followed by the frequency (in Hz) of
 * each frequency bin. Each subsequent row will contain the channel index, the
 * time (in seconds) of the corresponding STFT frame, and the magnitude (in dB)
 * of each frequency bin. The magnitude is computed as the magnitude of the
 * corresponding frequency bin for the corresponding STFT frame, and is
 * normalized by the window size.
 *
 * @param[in] numChannels The number of audio channels.
 * @param[in] sampleRate  The sample rate of the audio signal (in Hz).
 * @param[in] numFrames   The number of STFT frames (i.e., the number of
 *                        windowed FFTs computed per audio channel).
 * @param[in] windowSize  The size of the window used for the STFT (i.e., the
 *                        number of audio samples in each window).
 * @param[in] hopSize     The hop size used for the STFT (i.e., the number of
 *                        audio samples to advance for each window).
 * @param[in] bins        A read-only view of the bins (i.e., the complex
 *                        frequency-domain representation of the audio signal).
 * @param[in] path        The path to the CSV file to export the magnitude
 *                        spectrogram to.
 * @throws `std::exception` Thrown when any error occurs while exporting the
 *                          magnitude spectrogram.
 */
void export_spectrogram(
    usize numChannels,
    usize sampleRate,
    usize numFrames,
    usize windowSize,
    usize hopSize,
    std::span<const std::complex<f32>> bins,
    const std::string& path
);

}

#endif