#ifndef ESAP_EXCEPTIONS_HPP
#define ESAP_EXCEPTIONS_HPP

#include <stdexcept>
#include <string>

namespace esap {

/** @brief Represents an exception thrown by ESAP. */
class Exception : public std::runtime_error {
public:

    using runtime_error::runtime_error;

};

/**
 * @brief Represents an exception thrown when a waveform audio file is invalid
 * for whatever reason.
 */
class InvalidWav : public Exception {
public:

    using Exception::Exception;

};

/**
 * @brief Represents an exception thrown when a waveform audio file is valid but
 * uses an unsupported format (e.g., compressed audio).
 */
class UnsupportedWav : public Exception {
public:

    using Exception::Exception;

};

}

#endif