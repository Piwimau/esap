#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstdio>
#include <format>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include "esap/exceptions.hpp"
#include "esap/file-deleter.hpp"
#include "esap/wav.hpp"

namespace esap {

/** @brief The chunk ID for the `RIFF` chunk. */
static constexpr std::array<char, 4> RIFF_ID = { 'R', 'I', 'F', 'F' };

/** @brief The format ID for waveform audio files. */
static constexpr std::array<char, 4> WAVE_ID = { 'W', 'A', 'V', 'E' };

/** @brief The chunk ID for the `fmt ` chunk. */
static constexpr std::array<char, 4> FMT_ID = { 'f', 'm', 't', ' ' };

/** @brief The chunk ID for the `data` chunk. */
static constexpr std::array<char, 4> DATA_ID = { 'd', 'a', 't', 'a' };

/** @brief The size of a valid `fmt ` chunk for PCM data (in bytes). */
static constexpr u32 FMT_CHUNK_SIZE = 16;

/** @brief The audio format code for PCM data. */
static constexpr u16 AUDIO_FORMAT_PCM = 1;

/** @brief The scaling factor for 8-bit PCM samples. */
static constexpr f32 PCM_SCALE_8 = 128.0F;

/** @brief The scaling factor for 16-bit PCM samples. */
static constexpr f32 PCM_SCALE_16 = 32768.0F;

/** @brief The scaling factor for 24-bit PCM samples. */
static constexpr f32 PCM_SCALE_24 = 8388608.0F;

/** @brief The scaling factor for 32-bit PCM samples. */
static constexpr f32 PCM_SCALE_32 = 2147483648.0F;

/**
 * @brief The sign bit for 24-bit PCM samples, which is used to determine
 * whether a sample is negative or non-negative.
 */
static constexpr u32 I24_SIGN_BIT = 0x800000;

/**
 * @brief The sign extension mask for 24-bit PCM samples, which is used to
 * extend the sign bit to a 32-bit integer.
 */
static constexpr u32 I24_SIGN_EXT = 0xFF000000;

/** @brief The minimum value of a signed 24-bit integer. */
static constexpr i32 I24_MIN = -8388608;

/** @brief The maximum value of a signed 24-bit integer. */
static constexpr i32 I24_MAX = 8388607;

Wav::Wav(AudioFormat format, std::unique_ptr<f32[]> samples)
    : _format(format),
      _samples(std::move(samples)) {
    assert(_format.numChannels > 0);
    assert(_format.sampleRate > 0);
    assert(
        (_format.bitsPerSample == 8)
            || (_format.bitsPerSample == 16)
            || (_format.bitsPerSample == 24)
            || (_format.bitsPerSample == 32)
    );
}

/**
 * @brief Reads an unsigned integer from a specified buffer of bytes.
 *
 * @note This function assumes that the buffer contains the bytes of an unsigned
 * integer in little-endian byte order, which is the byte order used in waveform
 * audio files. If the host's native byte order is big-endian, then the function
 * will perform a byte swap to convert the value to the correct byte order.
 *
 * @tparam T The type of the unsigned integer to read.
 * @param[in] src The buffer to read the unsigned integer from.
 * @return The unsigned integer read from the buffer.
 */
template<std::unsigned_integral T>
requires (sizeof(T) == 2) || (sizeof(T) == 4)
static inline T read_le(std::span<const byte, sizeof(T)> src) noexcept {
    std::array<byte, sizeof(T)> buffer;
    std::ranges::copy(src, buffer.begin());
    T value = std::bit_cast<T>(buffer);
    if constexpr (std::endian::native == std::endian::big) {
        value = std::byteswap(value);
    }
    return value;
}

/**
 * @brief Reads an unsigned 24-bit integer from a specified buffer of bytes.
 *
 * @note This function assumes that the buffer contains the bytes of an unsigned
 * 24-bit integer in little-endian byte order, which is the byte order used in
 * waveform audio files. If the host's native byte order is big-endian, then the
 * function will perform a byte swap to convert the value to the correct byte
 * order.
 *
 * @tparam T The type of the unsigned integer to read.
 * @param[in] src The buffer to read the unsigned 24-bit integer from.
 * @return The unsigned 24-bit integer read from the buffer.
 */
template<std::unsigned_integral T>
requires (sizeof(T) == 4)
static inline T read_le(std::span<const byte, 3> src) noexcept {
    std::array<byte, 4> buffer;
    std::ranges::copy(src, buffer.begin());
    buffer[3] = static_cast<byte>(0);
    T value = std::bit_cast<T>(buffer);
    if constexpr (std::endian::native == std::endian::big) {
        value = std::byteswap(value);
    }
    return value;
}

/**
 * @brief Reads an unsigned integer from a specified file stream.
 *
 * @note This function assumes that the file stream contains the bytes of an
 * unsigned integer in little-endian byte order, which is the byte order used in
 * waveform audio files. If the host's native byte order is big-endian, then the
 * function will perform a byte swap to convert the value to the correct byte
 * order.
 *
 * @tparam T The type of the unsigned integer to read.
 * @param[in] file The file stream to read the unsigned integer from.
 * @return The unsigned integer read from the file stream.
 * @throws `std::runtime_error` Thrown when the file stream cannot be read from.
 */
template<std::unsigned_integral T>
requires (sizeof(T) == 2) || (sizeof(T) == 4)
static inline T read_le(std::FILE* file) {
    assert(file != nullptr);
    std::array<byte, sizeof(T)> buffer;
    if (std::fread(buffer.data(), buffer.size(), 1, file) != 1) {
        throw std::runtime_error("Failed to read from the file stream.");
    }
    T value = std::bit_cast<T>(buffer);
    if constexpr (std::endian::native == std::endian::big) {
        value = std::byteswap(value);
    }
    return value;
}

/**
 * @brief Writes an unsigned integer to a specified buffer of bytes.
 *
 * @note This function writes the bytes of the unsigned integer to the buffer in
 * little-endian byte order, which is the byte order used in waveform audio
 * files. If the host's native byte order is big-endian, then the function will
 * perform a byte swap to convert the value to the correct byte order.
 *
 * @tparam T The type of the unsigned integer to write.
 * @param[out] dst   The buffer to write the unsigned integer to.
 * @param[in]  value The unsigned integer to write.
 */
template<std::unsigned_integral T>
requires (sizeof(T) == 2) || (sizeof(T) == 4)
static inline void write_le(std::span<byte, sizeof(T)> dst, T value) noexcept {
    if constexpr (std::endian::native == std::endian::big) {
        value = std::byteswap(value);
    }
    std::ranges::copy(
        std::bit_cast<std::array<byte, sizeof(T)>>(value),
        dst.begin()
    );
}

/**
 * @brief Writes an unsigned 24-bit integer to a specified buffer of bytes.
 *
 * @note This function writes the bytes of the unsigned 24-bit integer to the
 * buffer in little-endian byte order, which is the byte order used in waveform
 * audio files. If the host's native byte order is big-endian, then the function
 * will perform a byte swap to convert the value to the correct byte order.
 *
 * @tparam T The type of the unsigned integer to write.
 * @param[out] dst   The buffer to write the unsigned 24-bit integer to.
 * @param[in]  value The unsigned 24-bit integer to write.
 */
template<std::unsigned_integral T>
requires (sizeof(T) == 4)
static inline void write_le(std::span<byte, 3> dst, T value) noexcept {
    if constexpr (std::endian::native == std::endian::big) {
        value = std::byteswap(value);
    }
    std::ranges::copy(
        std::bit_cast<std::array<byte, 4>>(value),
        dst.begin()
    );
}

/**
 * @brief Writes an unsigned integer to a specified file stream.
 *
 * @tparam T The type of the unsigned integer to write.
 * @param[in] file  The file stream to write the unsigned integer to.
 * @param[in] value The unsigned integer to write.
 * @throws `std::runtime_error` Thrown when the file stream cannot be written
 *                              to.
 */
template<std::unsigned_integral T>
requires (sizeof(T) == 2) || (sizeof(T) == 4)
static inline void write_le(std::FILE* file, T value) {
    assert(file != nullptr);
    if constexpr (std::endian::native == std::endian::big) {
        value = std::byteswap(value);
    }
    std::array<byte, sizeof(T)> buffer;
    std::ranges::copy(
        std::bit_cast<std::array<byte, sizeof(T)>>(value),
        buffer.begin()
    );
    if (std::fwrite(buffer.data(), buffer.size(), 1, file) != 1) {
        throw std::runtime_error("Failed to write to the file stream.");
    }
}

Wav Wav::read(const std::string& path) {
    std::unique_ptr<std::FILE, FileDeleter> file(
        std::fopen(path.c_str(), "rb")
    );
    if (file == nullptr) {
        throw std::runtime_error(std::format("Failed to open '{}'.", path));
    }
    // Valid waveform audio files must begin with a `RIFF` chunk containing the
    // `RIFF` identifier, followed by the file size (excluding eight bytes for
    // the `RIFF` identifier and file size itself), and the `WAVE` file format
    // identifier. Note that we ignore the file size specified in the `RIFF`
    // chunk, since it is redundant and may even be incorrect.
    std::array<char, 4> id;
    if (std::fread(id.data(), id.size(), 1, file.get()) != 1) {
        throw std::runtime_error(
            std::format("Failed to read from '{}'.", path)
        );
    }
    if (id != RIFF_ID) {
        throw InvalidWav(
            std::format(
                "Expected 'RIFF' chunk identifier, but found '{}'.",
                std::string(id.data(), id.size())
            )
        );
    }
    u32 size = read_le<u32>(file.get());
    if (std::fread(id.data(), id.size(), 1, file.get()) != 1) {
        throw std::runtime_error(
            std::format("Failed to read from '{}'.", path)
        );
    }
    if (id != WAVE_ID) {
        throw InvalidWav(
            std::format(
                "Expected 'WAVE' format identifier, but found '{}'.",
                std::string(id.data(), id.size())
            )
        );
    }
    // Following the `RIFF` chunk, a valid waveform audio file must contain two
    // mandatory chunks: A `fmt ` chunk (note the trailing space) containing a
    // description of the audio format, as well as a `data` chunk containing the
    // audio samples. These chunks must appear in that order, but they may be
    // interleaved with other optional chunks we are not interested in.
    bool foundFmt = false;
    bool foundData = false;
    u16 numChannels;
    u32 sampleRate;
    u16 blockAlign;
    u16 bitsPerSample;
    u32 numFrames;
    std::unique_ptr<f32[]> samples;
    while (
        !foundData && (std::fread(id.data(), id.size(), 1, file.get()) == 1)
    ) {
        size = read_le<u32>(file.get());
        if (id == FMT_ID) {
            u16 audioFormat = read_le<u16>(file.get());
            if (audioFormat != AUDIO_FORMAT_PCM) {
                throw UnsupportedWav(
                    std::format("Unsupported audio format '{}'.", audioFormat)
                );
            }
            if (size != FMT_CHUNK_SIZE) {
                throw InvalidWav(
                    std::format(
                        "Expected 'fmt ' chunk size to be '{}', but was '{}'.",
                        FMT_CHUNK_SIZE,
                        size
                    )
                );
            }
            numChannels = read_le<u16>(file.get());
            if (numChannels == 0) {
                throw InvalidWav(
                    "The number of audio channels must be greater than zero."
                );
            }
            sampleRate = read_le<u32>(file.get());
            if (sampleRate == 0) {
                throw InvalidWav("The sample rate must be greater than zero.");
            }
            u32 byteRate = read_le<u32>(file.get());
            if (byteRate == 0) {
                throw InvalidWav("The byte rate must be greater than zero.");
            }
            blockAlign = read_le<u16>(file.get());
            if (blockAlign == 0) {
                throw InvalidWav("The block align must be greater than zero.");
            }
            bitsPerSample = read_le<u16>(file.get());
            // We only support the common uncompressed PCM audio format with
            // either 8, 16, 24, or 32 bits per sample for simplicity.
            if (
                (bitsPerSample != 8) && (bitsPerSample != 16)
                    && (bitsPerSample != 24) && (bitsPerSample != 32)
            ) {
                throw UnsupportedWav(
                    std::format(
                        "Unsupported sample bit width '{}'.",
                        bitsPerSample
                    )
                );
            }
            if (blockAlign != (numChannels * bitsPerSample / 8)) {
                throw InvalidWav(
                    std::format("Inconsistent block align '{}'.", blockAlign)
                );
            }
            foundFmt = true;
        }
        else if (id == DATA_ID) {
            if (!foundFmt) {
                throw InvalidWav(
                    "Expected 'fmt ' chunk to appear before 'data' chunk."
                );
            }
            auto rawSamples = std::make_unique_for_overwrite<byte[]>(size);
            if (
                std::fread(
                    rawSamples.get(),
                    sizeof(byte),
                    size,
                    file.get()
                ) != size
            ) {
                throw std::runtime_error(
                    std::format("Failed to read from '{}'.", path)
                );
            }
            numFrames = size / blockAlign;
            samples = std::make_unique_for_overwrite<f32[]>(
                numChannels * numFrames
            );
            for (usize f = 0; f < numFrames; f++) {
                for (usize c = 0; c < numChannels; c++) {;
                    switch (bitsPerSample) {
                        case 8: {
                            u8 tmp = static_cast<u8>(
                                rawSamples[(f * numChannels + c) * sizeof(u8)]
                            );
                            samples[c * numFrames + f] =
                                (static_cast<f32>(tmp) - PCM_SCALE_8)
                                    / PCM_SCALE_8;
                            break;
                        }
                        case 16: {
                            u16 tmp = read_le<u16>(
                                std::span<const byte, sizeof(u16)>(
                                    rawSamples.get()
                                        + (f * numChannels + c) * sizeof(u16),
                                    sizeof(u16)
                                )
                            );
                            samples[c * numFrames + f] =
                                static_cast<f32>(static_cast<i16>(tmp))
                                    / PCM_SCALE_16;
                            break;
                        }
                        case 24: {
                            u32 tmp = read_le<u32>(
                                std::span<const byte, 3>(
                                    rawSamples.get()
                                        + (f * numChannels + c) * 3,
                                    3
                                )
                            );
                            if ((tmp & I24_SIGN_BIT) != 0) {
                                tmp |= I24_SIGN_EXT;
                            }
                            samples[c * numFrames + f] =
                                static_cast<f32>(static_cast<i32>(tmp))
                                    / PCM_SCALE_24;
                            break;
                        }
                        case 32: {
                            u32 tmp = read_le<u32>(
                                std::span<const byte, sizeof(u32)>(
                                    rawSamples.get()
                                        + (f * numChannels + c) * sizeof(u32),
                                    sizeof(u32)
                                )
                            );
                            samples[c * numFrames + f] =
                                static_cast<f32>(static_cast<i32>(tmp))
                                    / PCM_SCALE_32;
                            break;
                        }
                        default:
                            std::unreachable();
                    }
                }
            }
            foundData = true;
        }
        else {
            long skip = size + size % 2;
            if (std::fseek(file.get(), skip, SEEK_CUR) != 0) {
                throw std::runtime_error(
                    std::format("Failed to seek in '{}'.", path)
                );
            }
        }
    }
    if (!foundFmt) {
        throw InvalidWav("Missing 'fmt ' chunk.");
    }
    if (!foundData) {
        throw InvalidWav("Missing 'data' chunk.");
    }
    AudioFormat format = {
        .numChannels = numChannels,
        .bitsPerSample = bitsPerSample,
        .sampleRate = sampleRate,
        .numFrames = numFrames
    };
    return Wav(format, std::move(samples));
}

const AudioFormat& Wav::format() const noexcept {
    return _format;
}

std::span<const f32> Wav::samples() const noexcept {
    usize count = static_cast<usize>(_format.numChannels) * _format.numFrames;
    return std::span<const f32>(_samples.get(), count);
}

void Wav::write(const std::string& path) const {
    std::unique_ptr<std::FILE, FileDeleter> file(
        std::fopen(path.c_str(), "wb")
    );
    if (file == nullptr) {
        throw std::runtime_error(std::format("Failed to open '{}'.", path));
    }
    u16 bytesPerSample = _format.bitsPerSample / 8;
    u32 byteRate = _format.numChannels * _format.sampleRate * bytesPerSample;
    u16 blockAlign = _format.numChannels * bytesPerSample;
    u32 dataChunkSize = _format.numChannels * _format.numFrames
        * bytesPerSample;
    u32 riffChunkSize = 4 + 8 + FMT_CHUNK_SIZE + 8 + dataChunkSize
        + dataChunkSize % 2;
    if (std::fwrite(RIFF_ID.data(), RIFF_ID.size(), 1, file.get()) != 1) {
        throw std::runtime_error(std::format("Failed to write to '{}'.", path));
    }
    write_le<u32>(file.get(), riffChunkSize);
    if (
        (std::fwrite(WAVE_ID.data(), WAVE_ID.size(), 1, file.get()) != 1)
            || (std::fwrite(FMT_ID.data(), FMT_ID.size(), 1, file.get()) != 1)
    ) {
        throw std::runtime_error(std::format("Failed to write to '{}'.", path));
    }
    write_le<u32>(file.get(), FMT_CHUNK_SIZE);
    write_le<u16>(file.get(), AUDIO_FORMAT_PCM);
    write_le<u16>(file.get(), _format.numChannels);
    write_le<u32>(file.get(), _format.sampleRate);
    write_le<u32>(file.get(), byteRate);
    write_le<u16>(file.get(), blockAlign);
    write_le<u16>(file.get(), _format.bitsPerSample);
    if (std::fwrite(DATA_ID.data(), DATA_ID.size(), 1, file.get()) != 1) {
        throw std::runtime_error(std::format("Failed to write to '{}'.", path));
    }
    write_le<u32>(file.get(), dataChunkSize);
    // We need to convert the normalized floating-point audio samples back to
    // raw PCM samples. To do this, we first clamp each normalized sample to the
    // range [-1.0F, 1.0F], then scale it by the appropriate scaling factor for
    // the given bit width, and finally round it to the nearest integer.
    auto rawSamples = std::make_unique_for_overwrite<byte[]>(dataChunkSize);
    for (usize c = 0; c < _format.numChannels; c++) {
        for (usize f = 0; f < _format.numFrames; f++) {
            switch (_format.bitsPerSample) {
                case 8: {
                    f32 sample = std::clamp(
                        _samples[c * _format.numFrames + f],
                        -1.0F,
                        1.0F
                    );
                    i16 tmp = static_cast<i16>(std::lround(sample * PCM_SCALE_8))
                        + static_cast<i16>(PCM_SCALE_8);
                    tmp = std::clamp(
                        tmp,
                        static_cast<i16>(std::numeric_limits<u8>::min()),
                        static_cast<i16>(std::numeric_limits<u8>::max())
                    );
                    rawSamples[(f * _format.numChannels + c) * sizeof(u8)] =
                        static_cast<byte>(tmp);
                    break;
                }
                case 16: {
                    f32 sample = std::clamp(
                        _samples[c * _format.numFrames + f],
                        -1.0F,
                        1.0F
                    );
                    i32 tmp = static_cast<i32>(
                        std::lround(sample * PCM_SCALE_16)
                    );
                    tmp = std::clamp(
                        tmp,
                        static_cast<i32>(std::numeric_limits<i16>::min()),
                        static_cast<i32>(std::numeric_limits<i16>::max())
                    );
                    write_le<u16>(
                        std::span<byte, sizeof(u16)>(
                            rawSamples.get()
                                + (f * _format.numChannels + c) * sizeof(u16),
                            sizeof(u16)
                        ),
                        static_cast<u16>(tmp)
                    );
                    break;
                }
                case 24: {
                    f32 sample = std::clamp(
                        _samples[c * _format.numFrames + f],
                        -1.0F,
                        1.0F
                    );
                    i32 tmp = static_cast<i32>(
                        std::lround(sample * PCM_SCALE_24)
                    );
                    tmp = std::clamp(tmp, I24_MIN, I24_MAX);
                    write_le<u32>(
                        std::span<byte, 3>(
                            rawSamples.get()
                                + (f * _format.numChannels + c) * 3,
                            3
                        ),
                        static_cast<u32>(tmp)
                    );
                    break;
                }
                case 32: {
                    f32 sample = std::clamp(
                        _samples[c * _format.numFrames + f],
                        -1.0F,
                        1.0F
                    );
                    i64 tmp = std::llround(sample * PCM_SCALE_32);
                    tmp = std::clamp(
                        tmp,
                        static_cast<i64>(std::numeric_limits<i32>::min()),
                        static_cast<i64>(std::numeric_limits<i32>::max())
                    );
                    write_le<u32>(
                        std::span<byte, sizeof(u32)>(
                            rawSamples.get()
                                + (f * _format.numChannels + c) * sizeof(u32),
                            sizeof(u32)
                        ),
                        static_cast<u32>(tmp)
                    );
                    break;
                }
                default:
                    std::unreachable();
            }
        }
    }
    if (
        std::fwrite(
            rawSamples.get(),
            sizeof(byte),
            dataChunkSize,
            file.get()
        ) != dataChunkSize
    ) {
        throw std::runtime_error(std::format("Failed to write to '{}'.", path));
    }
    if ((dataChunkSize % 2) != 0) {
        byte pad = static_cast<byte>(0);
        if (std::fwrite(&pad, sizeof(byte), 1, file.get()) != 1) {
            throw std::runtime_error(
                std::format("Failed to write to '{}'.", path)
            );
        }
    }
}

}