﻿// -----------------------------------------------------------------------------------------
// AviutlColor by rigaya
// -----------------------------------------------------------------------------------------
// The MIT License
//
// Copyright (c) 2017 rigaya
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//
// --------------------------------------------------------------------------------------------

#define NOMINMAX
#include <Windows.h>
#include <cstdint>
#include <cstdlib>
#include "color.h"
#include "convert_csp.h"
#include "color_simd.h"
#include <algorithm>

void convert_yuy2_yc48_c(int thread_id, int thread_num, void *param1, void *param2) {
    COLOR_PROC_INFO *cpip = (COLOR_PROC_INFO *)param1;
    const int max_threads = std::min(thread_num, *(int *)param2);
    if (thread_id >= max_threads) return;

    convert_yuy2_yc48(cpip, thread_id, max_threads, btxxx_to_bt601);
}
void convert_yc48_yuy2_c(int thread_id, int thread_num, void *param1, void *param2) {
    COLOR_PROC_INFO *cpip = (COLOR_PROC_INFO *)param1;
    const int max_threads = std::min(thread_num, *(int *)param2);
    if (thread_id >= max_threads) return;

    convert_yc48_yuy2(cpip, thread_id, max_threads, bt601_to_btxxx);
}
void convert_yc48_btxxx_bt601_c(int thread_id, int thread_num, void *param1, void *param2) {
    COLOR_PROC_INFO *cpip = (COLOR_PROC_INFO *)param1;
    const int max_threads = std::min(thread_num, *(int *)param2);
    if (thread_id >= max_threads) return;

    convert_matrix_yc48<true>(cpip, thread_id, max_threads, btxxx_to_bt601);
}
void convert_yc48_bt601_btxxx_c(int thread_id, int thread_num, void *param1, void *param2) {
    COLOR_PROC_INFO *cpip = (COLOR_PROC_INFO *)param1;
    const int max_threads = std::min(thread_num, *(int *)param2);
    if (thread_id >= max_threads) return;

    convert_matrix_yc48<false>(cpip, thread_id, max_threads, bt601_to_btxxx);
}

void convert_yuy2_yc48_sse2(int thread_id, int thread_num, void *param1, void *param2);
void convert_yuy2_yc48_ssse3(int thread_id, int thread_num, void *param1, void *param2);
void convert_yuy2_yc48_sse41(int thread_id, int thread_num, void *param1, void *param2);
void convert_yuy2_yc48_avx(int thread_id, int thread_num, void *param1, void *param2);
void convert_yuy2_yc48_avx2(int thread_id, int thread_num, void *param1, void *param2);
void convert_yc48_yuy2_sse2(int thread_id, int thread_num, void *param1, void *param2);
void convert_yc48_yuy2_ssse3(int thread_id, int thread_num, void *param1, void *param2);
void convert_yc48_yuy2_sse41(int thread_id, int thread_num, void *param1, void *param2);
void convert_yc48_yuy2_avx(int thread_id, int thread_num, void *param1, void *param2);
void convert_yc48_yuy2_avx2(int thread_id, int thread_num, void *param1, void *param2);
void convert_yc48_btxxx_bt601_sse2(int thread_id, int thread_num, void *param1, void *param2);
void convert_yc48_btxxx_bt601_ssse3(int thread_id, int thread_num, void *param1, void *param2);
void convert_yc48_btxxx_bt601_sse41(int thread_id, int thread_num, void *param1, void *param2);
void convert_yc48_btxxx_bt601_avx(int thread_id, int thread_num, void *param1, void *param2);
void convert_yc48_btxxx_bt601_avx2(int thread_id, int thread_num, void *param1, void *param2);
void convert_yc48_bt601_btxxx_sse2(int thread_id, int thread_num, void *param1, void *param2);
void convert_yc48_bt601_btxxx_ssse3(int thread_id, int thread_num, void *param1, void *param2);
void convert_yc48_bt601_btxxx_sse41(int thread_id, int thread_num, void *param1, void *param2);
void convert_yc48_bt601_btxxx_avx(int thread_id, int thread_num, void *param1, void *param2);
void convert_yc48_bt601_btxxx_avx2(int thread_id, int thread_num, void *param1, void *param2);

void get_func(convert_color_func *func_list, uint32_t simd_avail) {
    struct func_data {
        uint32_t simd;
        MULTI_THREAD_FUNC func;
    };
    static const func_data FUNC_YUY2_YC48[] = {
        { AVX2,  convert_yuy2_yc48_avx2 },
        { AVX,   convert_yuy2_yc48_avx },
        { SSE41, convert_yuy2_yc48_sse41 },
        { SSSE3, convert_yuy2_yc48_ssse3 },
        { SSE2,  convert_yuy2_yc48_sse2 },
        { NONE,  convert_yuy2_yc48_c },
    };
    static const func_data FUNC_YC48_YUY2[] = {
        { AVX2,  convert_yc48_yuy2_avx2 },
        { AVX,   convert_yc48_yuy2_avx },
        { SSE41, convert_yc48_yuy2_sse41 },
        { SSSE3, convert_yc48_yuy2_ssse3 },
        { SSE2,  convert_yc48_yuy2_sse2 },
        { NONE,  convert_yc48_yuy2_c },
    };
    static const func_data FUNC_YC48_BTXXX_BT601[] = {
        { AVX2,  convert_yc48_btxxx_bt601_avx2 },
        { AVX,   convert_yc48_btxxx_bt601_avx },
        { SSE41, convert_yc48_btxxx_bt601_sse41 },
        { SSSE3, convert_yc48_btxxx_bt601_ssse3 },
        { SSE2,  convert_yc48_btxxx_bt601_sse2 },
        { NONE,  convert_yc48_btxxx_bt601_c },
    };
    static const func_data FUNC_YC48_BT601_BTXXX[] = {
        { AVX2,  convert_yc48_bt601_btxxx_avx2 },
        { AVX,   convert_yc48_bt601_btxxx_avx },
        { SSE41, convert_yc48_bt601_btxxx_sse41 },
        { SSSE3, convert_yc48_bt601_btxxx_ssse3 },
        { SSE2,  convert_yc48_bt601_btxxx_sse2 },
        { NONE,  convert_yc48_bt601_btxxx_c },
    };

    for (int i = 0; i < _countof(FUNC_YUY2_YC48); i++) {
        if ((FUNC_YUY2_YC48[i].simd & simd_avail) == FUNC_YUY2_YC48[i].simd) {
            func_list->yuy2_yc48 = FUNC_YUY2_YC48[i].func;
            break;
        }
    }
    for (int i = 0; i < _countof(FUNC_YC48_YUY2); i++) {
        if ((FUNC_YC48_YUY2[i].simd & simd_avail) == FUNC_YC48_YUY2[i].simd) {
            func_list->yc48_yuy2 = FUNC_YC48_YUY2[i].func;
            break;
        }
    }
    for (int i = 0; i < _countof(FUNC_YC48_BTXXX_BT601); i++) {
        if ((FUNC_YC48_BTXXX_BT601[i].simd & simd_avail) == FUNC_YC48_BTXXX_BT601[i].simd) {
            func_list->yc48_btxxx_bt601 = FUNC_YC48_BTXXX_BT601[i].func;
            break;
        }
    }
    for (int i = 0; i < _countof(FUNC_YC48_BT601_BTXXX); i++) {
        if ((FUNC_YC48_BT601_BTXXX[i].simd & simd_avail) == FUNC_YC48_BT601_BTXXX[i].simd) {
            func_list->yc48_bt601_btxxx = FUNC_YC48_BT601_BTXXX[i].func;
            break;
        }
    }
}

void get_func(convert_color_func *func_list) {
    get_func(func_list, get_availableSIMD());
}

#include <intrin.h>

uint32_t get_availableSIMD() {
    int CPUInfo[4];
    __cpuid(CPUInfo, 1);
    uint32_t simd = NONE;
    if (CPUInfo[3] & 0x04000000) simd |= SSE2;
    if (CPUInfo[2] & 0x00000001) simd |= SSE3;
    if (CPUInfo[2] & 0x00000200) simd |= SSSE3;
    if (CPUInfo[2] & 0x00080000) simd |= SSE41;
    if (CPUInfo[2] & 0x00100000) simd |= SSE42;
    if (CPUInfo[2] & 0x00800000) simd |= POPCNT;
#if (_MSC_VER >= 1600)
    uint64_t xgetbv = 0;
    if ((CPUInfo[2] & 0x18000000) == 0x18000000) {
        xgetbv = _xgetbv(0);
        if ((xgetbv & 0x06) == 0x06)
            simd |= AVX;
    }
#endif
#if (_MSC_VER >= 1700)
    __cpuid(CPUInfo, 7);
    if ((simd & AVX) && (CPUInfo[1] & 0x00000020))
        simd |= AVX2;
#endif
    return simd;
}

