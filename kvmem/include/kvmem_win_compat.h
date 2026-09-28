#pragma once

// ===========================================================================
// kvmem_win_compat.h - minimal POSIX -> Win32 shim for the kvmem host library
// ===========================================================================
//
// WHY THIS EXISTS
//   kvmem is developed on Linux (Ubuntu/WSL2). Its host library uses POSIX
//   headers, positional file I/O and POSIX clock/file APIs that MSVC does not
//   provide. This header supplies just enough of that surface to compile the
//   host library on Windows with MSVC.
//
// LOCAL ADDITION - not part of upstream kvmem.
//   nvme_kv_tier.hpp includes this under #if defined(_WIN32); every other
//   kvmem translation unit reaches it transitively through that header.
//   Upstream README states the NVMe tier is NOT IMPLEMENTED, so these code
//   paths are not exercised at runtime; this shim exists to make the
//   translation units compile, and is written to be correct if they ever run.
//
// TO REVERT: delete this file and restore the #include block at the bottom of
//   nvme_kv_tier.hpp's include list. That is the only other change.
// ===========================================================================

#if !defined(_WIN32)
#error "kvmem_win_compat.h must only be included on Windows"
#endif

#if !defined(_MSC_VER)
#error "kvmem_win_compat.h currently supports MSVC only"
#endif

// ---------------------------------------------------------------------------
// 1. windows.h, and the min/max macro problem
// ---------------------------------------------------------------------------
// windows.h defines min() and max() as macros. Every std::numeric_limits<T>::max()
// in the kvmem sources then fails to parse ("illegal token on right side of ::").
// NOMINMAX suppresses them; the #undef pair below also covers the case where
// windows.h was already pulled in by another header before us.

#if !defined(NOMINMAX)
#define NOMINMAX
#endif

#include <windows.h>

#if defined(min)
#undef min
#endif
#if defined(max)
#undef max
#endif

// ---------------------------------------------------------------------------
// 2. Standard headers MSVC needs
// ---------------------------------------------------------------------------

#include <io.h>        // _open/_close/_commit/_get_osfhandle
#include <fcntl.h>     // _O_* flags
#include <direct.h>    // _mkdir
#include <sys/stat.h>
#include <sys/types.h>
#include <BaseTsd.h>   // SSIZE_T
#include <time.h>      // timespec

#include <cstdint>
#include <cstddef>
#include <cerrno>

// ---------------------------------------------------------------------------
// 3. Missing constants
// ---------------------------------------------------------------------------

// Linux-only. Upstream guards its use with #if defined(__linux__); this only
// needs to exist so the guarded branches are skipped cleanly.
#if !defined(O_CLOEXEC)
#define O_CLOEXEC _O_NOINHERIT
#endif

#if !defined(F_OK)
#define F_OK 0
#endif

#if !defined(CLOCK_MONOTONIC)
#define CLOCK_MONOTONIC 1
#endif

// ---------------------------------------------------------------------------
// 4. Force BINARY mode on every open
// ---------------------------------------------------------------------------
// MSVC defaults to TEXT mode, in which the CRT rewrites CRLF on read and write.
// For a KV store that would silently corrupt every payload. Every ::open call
// in nvme_kv_tier.hpp passes at least one of O_CREAT / O_RDWR / O_TRUNC, so
// folding _O_BINARY into those flags guarantees binary mode without touching
// any call site.

#undef O_RDWR
#define O_RDWR (_O_RDWR | _O_BINARY)
#undef O_WRONLY
#define O_WRONLY (_O_WRONLY | _O_BINARY)
#undef O_CREAT
#define O_CREAT (_O_CREAT | _O_BINARY)
#undef O_TRUNC
#define O_TRUNC (_O_TRUNC | _O_BINARY)

// ---------------------------------------------------------------------------
// 5. Missing types
// ---------------------------------------------------------------------------

// MSVC defines SSIZE_T (BaseTsd.h) but not the POSIX spelling.
typedef SSIZE_T ssize_t;

// MSVC typedefs off_t as 32-bit `long`, which would truncate files over 2 GiB.
// A macro rather than a typedef, because <sys/types.h> already declared off_t.
// The extra guard pair keeps this idempotent across repeated includes.
#if !defined(KVMEM_OFF_T_IS_64BIT)
#define KVMEM_OFF_T_IS_64BIT 1
#define off_t long long
#endif

// ---------------------------------------------------------------------------
// 6. Missing functions
// ---------------------------------------------------------------------------

// POSIX mkdir(path, mode) -> Win32 _mkdir(path). The mode argument is ignored.
#if !defined(mkdir)
#define mkdir(path, mode) _mkdir(path)
#endif

// POSIX fdatasync(fd) -> MSVC _commit(fd), which flushes to disk.
#if !defined(fdatasync)
#define fdatasync(fd) _commit(fd)
#endif

// ---------------------------------------------------------------------------
// 7. Positional I/O: pread / pwrite
// ---------------------------------------------------------------------------
// MSVC has no pread/pwrite. The upstream comment explains why they are needed:
//   "Positional pread/pwrite removes the shared FILE* cursor so stage-out
//    writes and stage-in reads may safely run on different host workers."
// An offset-seek emulation under a mutex would defeat that purpose, so we use
// ReadFile/WriteFile with an explicit OVERLAPPED offset: on Windows that is
// true positional I/O and leaves the shared file pointer untouched.

namespace kvmem_win_compat {

inline void set_overlapped_offset(OVERLAPPED & ov, long long offset) {
    ZeroMemory(&ov, sizeof(ov));
    const unsigned long long u = static_cast<unsigned long long>(offset);
    ov.Offset     = static_cast<DWORD>(u & 0xFFFFFFFFull);
    ov.OffsetHigh = static_cast<DWORD>(u >> 32);
}

inline ssize_t pread_impl(int fd, void * buf, size_t count, long long offset) {
    HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }

    OVERLAPPED ov;
    set_overlapped_offset(ov, offset);

    DWORD got = 0;
    if (!ReadFile(h, buf, static_cast<DWORD>(count), &got, &ov)) {
        const DWORD err = GetLastError();
        if (err == ERROR_HANDLE_EOF) return 0;
        errno = EIO;
        return -1;
    }
    return static_cast<ssize_t>(got);
}

inline ssize_t pwrite_impl(int fd, const void * buf, size_t count, long long offset) {
    HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    if (h == INVALID_HANDLE_VALUE) { errno = EBADF; return -1; }

    OVERLAPPED ov;
    set_overlapped_offset(ov, offset);

    DWORD put = 0;
    if (!WriteFile(h, buf, static_cast<DWORD>(count), &put, &ov)) {
        errno = EIO;
        return -1;
    }
    return static_cast<ssize_t>(put);
}

// Monotonic clock backed by the performance counter.
inline int clock_gettime_impl(struct timespec * tp) {
    static LARGE_INTEGER freq = [] {
        LARGE_INTEGER f{};
        QueryPerformanceFrequency(&f);
        return f;
    }();

    LARGE_INTEGER now{};
    QueryPerformanceCounter(&now);

    if (freq.QuadPart <= 0) { tp->tv_sec = 0; tp->tv_nsec = 0; return 0; }

    tp->tv_sec  = static_cast<time_t>(now.QuadPart / freq.QuadPart);
    tp->tv_nsec = static_cast<long>(
        (now.QuadPart % freq.QuadPart) * 1000000000LL / freq.QuadPart);
    return 0;
}

}  // namespace kvmem_win_compat

// The upstream call sites are written as ::pread(...) / ::pwrite(...) /
// clock_gettime(...), so the replacements must be visible globally.

inline ssize_t pread(int fd, void * buf, size_t count, off_t offset) {
    return kvmem_win_compat::pread_impl(fd, buf, count, offset);
}

inline ssize_t pwrite(int fd, const void * buf, size_t count, off_t offset) {
    return kvmem_win_compat::pwrite_impl(fd, buf, count, offset);
}

inline int clock_gettime(int clk_id, struct timespec * tp) {
    (void)clk_id;  // only CLOCK_MONOTONIC is used by kvmem
    return kvmem_win_compat::clock_gettime_impl(tp);
}
