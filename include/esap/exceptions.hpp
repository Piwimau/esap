#ifndef ESAP_EXCEPTIONS_HPP
#define ESAP_EXCEPTIONS_HPP

#include <stdexcept>
#include <string>

namespace esap {

/** @brief Represents an exception thrown by ESAP. */
class Exception : public std::runtime_error {
public:

    /**
     * @brief Initializes a new `Exception` with a specified error message.
     *
     * @param[in] msg The error message describing the exception.
     */
    explicit Exception(const std::string& msg) : std::runtime_error(msg) { }

    /**
     * @brief Initializes a new `Exception` with a specified error message.
     *
     * @param[in] msg The error message describing the exception.
     */
    explicit Exception(const char* msg) : std::runtime_error(msg) { }

};

/**
 * @brief Represents an exception thrown when a waveform audio file is invalid
 * for whatever reason.
 */
class InvalidWav : public Exception {
public:

    /**
     * @brief Initializes a new `InvalidWav` exception with a specified error
     * message.
     *
     * @param[in] msg The error message describing the exception.
     */
    explicit InvalidWav(const std::string& msg) : Exception(msg) { }

    /**
     * @brief Initializes a new `InvalidWav` exception with a specified error
     * message.
     *
     * @param[in] msg The error message describing the exception.
     */
    explicit InvalidWav(const char* msg) : Exception(msg) { }

};

/**
 * @brief Represents an exception thrown when a waveform audio file is valid but
 * uses an unsupported format (e.g., compressed audio).
 */
class UnsupportedWav : public Exception {
public:

    /**
     * @brief Initializes a new `UnsupportedWav` exception with a specified
     * error message.
     *
     * @param[in] msg The error message describing the exception.
     */
    explicit UnsupportedWav(const std::string& msg) : Exception(msg) { }

    /**
     * @brief Initializes a new `UnsupportedWav` exception with a specified
     * error message.
     *
     * @param[in] msg The error message describing the exception.
     */
    explicit UnsupportedWav(const char* msg) : Exception(msg) { }

};

}

#endif