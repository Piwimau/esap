#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#else
    #include <time.h>
#endif
#include "esap/bench.hpp"

namespace esap {

std::optional<std::chrono::nanoseconds> cpu_time() {
#ifdef _WIN32
    FILETIME creation, exit, kernel, user;
    if (!GetThreadTimes(GetCurrentThread(), &creation, &exit, &kernel, &user)) {
        return std::nullopt;
    }
    ULARGE_INTEGER k, u;
    k.LowPart = kernel.dwLowDateTime;
    k.HighPart = kernel.dwHighDateTime;
    u.LowPart = user.dwLowDateTime;
    u.HighPart = user.dwHighDateTime;
    return std::chrono::nanoseconds(
        static_cast<i64>(k.QuadPart + u.QuadPart) * 100
    );
#else
    struct timespec ts;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0) {
        return std::nullopt;
    }
    return std::chrono::nanoseconds(
        static_cast<i64>(ts.tv_sec) * 1'000'000'000
            + static_cast<i64>(ts.tv_nsec)
    );
#endif
}

}