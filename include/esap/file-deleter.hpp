#ifndef ESAP_FILE_DELETER_HPP
#define ESAP_FILE_DELETER_HPP

#include <cstdio>

namespace esap {

/** @brief Represents a custom deleter for `std::FILE` handles. */
struct FileDeleter {

    /**
     * @brief Closes a specified file stream.
     *
     * @warning The behavior is undefined if `file` is a `nullptr` or if it has
     * already been closed. This function assumes that `file` was obtained from
     * a successful call to `std::fopen()` (or a similar function).
     *
     * @param[in, out] file The file stream to close.
     */
    void operator()(std::FILE* file) const noexcept {
        std::fclose(file);
    }

};

}

#endif