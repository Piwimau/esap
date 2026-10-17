#ifndef ESAP_FILE_DELETER_HPP
#define ESAP_FILE_DELETER_HPP

#include <cstdio>

/** @brief Represents a custom deleter for `std::FILE` handles. */
struct FileDeleter {

    /**
     * @brief Closes a `std::FILE` handle.
     *
     * @param[in, out] file The `std::FILE` handle to close.
     *
     * @warning The behavior is undefined if `file` is a `nullptr`, or if it has
     * already been closed. Note that this function assumes that `file` was
     * obtained from `std::fopen()` (or a similar function).
     */
    void operator()(std::FILE* file) const noexcept {
        std::fclose(file);
    }

};

#endif