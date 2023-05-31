#pragma once
#include <stdio.h>

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const hipError_t error_code = call;              \
    if (error_code != hipSuccess)                    \
    {                                                 \
        printf("Hip Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            hipGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

