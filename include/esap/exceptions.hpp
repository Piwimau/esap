#ifndef ESAP_EXCEPTIONS_HPP
#define ESAP_EXCEPTIONS_HPP

#include <stdexcept>
#include <string>

namespace esap {

/** @brief An exception thrown by ESAP. */
class Exception : public std::runtime_error {
public:

    using runtime_error::runtime_error;

};

/** @brief An exception thrown when a waveform audio file is invalid. */
class InvalidWav : public Exception {
public:

    using Exception::Exception;

};

/**
 * @brief An exception thrown when a waveform audio file is valid but uses an
 * unsupported format (e.g., compressed audio).
 */
class UnsupportedWav : public Exception {
public:

    using Exception::Exception;

};

}

#endif