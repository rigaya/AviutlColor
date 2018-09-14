// -----------------------------------------------------------------------------------------
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

#define USE_SSE2  1
#define USE_SSSE3 1
#define USE_SSE41 1
#define USE_AVX   1
#define USE_AVX2  0

#define NOMINMAX
#include <windows.h>
#include "color.h"
#include "convert_csp_simd.h"
#include <algorithm>

void convert_yuy2_yc48_avx(int thread_id, int thread_num, void *param1, void *param2) {
    COLOR_PROC_INFO *cpip = (COLOR_PROC_INFO *)param1;
    const int max_threads = std::min(thread_num, *(int *)param2);
    if (thread_id >= max_threads) return;

    convert_yuy2_yc48_simd(cpip, thread_id, max_threads, btxxx_to_bt601);
}
void convert_yc48_yuy2_avx(int thread_id, int thread_num, void *param1, void *param2) {
    COLOR_PROC_INFO *cpip = (COLOR_PROC_INFO *)param1;
    const int max_threads = std::min(thread_num, *(int *)param2);
    if (thread_id >= max_threads) return;

    convert_yc48_yuy2_simd(cpip, thread_id, max_threads, bt601_to_btxxx);
}
void convert_yc48_btxxx_bt601_avx(int thread_id, int thread_num, void *param1, void *param2) {
    COLOR_PROC_INFO *cpip = (COLOR_PROC_INFO *)param1;
    const int max_threads = std::min(thread_num, *(int *)param2);
    if (thread_id >= max_threads) return;

    convert_matrix_yc48_simd<true>(cpip, thread_id, max_threads, btxxx_to_bt601);
}
void convert_yc48_bt601_btxxx_avx(int thread_id, int thread_num, void *param1, void *param2) {
    COLOR_PROC_INFO *cpip = (COLOR_PROC_INFO *)param1;
    const int max_threads = std::min(thread_num, *(int *)param2);
    if (thread_id >= max_threads) return;

    convert_matrix_yc48_simd<false>(cpip, thread_id, max_threads, bt601_to_btxxx);
}
