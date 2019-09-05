#pragma once

#include <clcore.hpp>

enum clMemcpyKind {
    clMemcpyDeviceToHost=111,
    clMemcpyHostToDevice=222,
    clMemcpyDeviceToDevice=333,
    clMemcpyDefault=444
};

void clMalloc( void ** ptr, size_t size);
void clFree(void * ptr);
void clMemcpy(void * dst, const void * src, size_t count, enum clMemcpyKind kind);
void clMemset(void * ptr, unsigned int value, size_t count);
