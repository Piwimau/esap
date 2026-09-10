#ifndef ESAP_FUNC_DELETER_HPP
#define ESAP_FUNC_DELETER_HPP

namespace esap {

/**
 * @brief Represents a generic deleter that calls a specified function on an
 * object.
 *
 * @tparam Func The function to be called when deleting the object.
 */
template<auto Func>
struct FuncDeleter {

    /**
     * @brief Deletes a specified object by calling `Func` on it.
     *
     * @tparam T The type of the object to be deleted.
     * @param[in] ptr A pointer to the object to be deleted.
     */
    template<typename T>
    void operator()(T* ptr) const noexcept {
        Func(ptr);
    }

};

}

#endif