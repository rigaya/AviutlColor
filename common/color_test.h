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

#ifndef __COLOR_TEST_H__
#define __COLOR_TEST_H__

#include <cstdint>

void get_func(convert_color_func *func_list, uint32_t simd_avail);

static int compare_yc48(const PIXEL_YC *ycp0, const PIXEL_YC *ycp1, int w, int pitch_pixels, int h) {
    int ret = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = y * pitch_pixels + x;
            if (0 != memcmp(&ycp0[idx], &ycp1[idx], sizeof(PIXEL_YC))) {
#if PRINT_DETAIL
                fprintf(stderr, "Error at (%4d,%4d): None(%5d,%5d,%5d)-SIMD(%5d,%5d,%5d)\n",
                    x, y,
                    ycp0[idx].y, ycp0[idx].cb, ycp0[idx].cr,
                    ycp1[idx].y, ycp1[idx].cb, ycp1[idx].cr);
#endif
                ret = 1;
            }
        }
    }
    return ret;
}

static int compare_yuy2(const unsigned char *ptr0, const unsigned char *ptr1, int w, int h) {
    int ret = 0;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x+=2) {
            int idx_byte = (y * w + x) * 2;
            if (0 != memcmp(&ptr0[idx_byte], &ptr1[idx_byte], 4)) {
                fprintf(stderr, "Error at (%4d,%4d): None(%3d,%3d,%3d,%3d)-SIMD(%3d,%3d,3d,%3d)\n",
                    x, y,
                    ptr0[idx_byte+0], ptr0[idx_byte+1], ptr0[idx_byte+2], ptr0[idx_byte+3],
                    ptr1[idx_byte+0], ptr1[idx_byte+1], ptr1[idx_byte+2], ptr1[idx_byte+3]);
                ret = 1;
            }
        }
    }
    return ret;
}

static void set_random_yc48(PIXEL_YC *ycp, int pixels, std::mt19937& mt) {
    std::uniform_int_distribution<int> random_yc48(0, 4096);
    for (int i = 0; i < pixels; i++) {
        ycp[i].y  = random_yc48(mt);
        ycp[i].cb = random_yc48(mt) - 2048;
        ycp[i].cr = random_yc48(mt) - 2048;
    }
}

static void set_random_yuy2(unsigned char *ptr, int pixels, std::mt19937& mt) {
    std::uniform_int_distribution<int> random_yuy2(16, 235);
    for (int i = 0; i < pixels; i++) {
        ptr[2*i + 0] = (unsigned char)random_yuy2(mt);
        ptr[2*i + 1] = (unsigned char)random_yuy2(mt);
    }
}

static const char *simd_str(uint32_t simd) {
    if (simd & AVX2)  return "AVX2 ";
    if (simd & AVX)   return "AVX  ";
    if (simd & SSE41) return "SSE41";
    if (simd & SSSE3) return "SSSE3";
    if (simd & SSE2)  return "SSE2 ";
    return "NONE ";
}

int run_test(int w, int yc48_pitch, int h) {
    int ret = 0;
    COLOR_PROC_INFO cpinfo;
    cpinfo.w = w;
    cpinfo.h = h;
    cpinfo.line_size = yc48_pitch * sizeof(PIXEL_YC);
    cpinfo.yc_size = 6;

    PIXEL_YC *ycp0 = (PIXEL_YC *)_aligned_malloc(sizeof(PIXEL_YC) * yc48_pitch * h, 16);
    PIXEL_YC *ycp1 = (PIXEL_YC *)_aligned_malloc(sizeof(PIXEL_YC) * yc48_pitch * h, 16);

    PIXEL_YC *dib_ycp0 = (PIXEL_YC *)_aligned_malloc(sizeof(PIXEL_YC) * w * h, 16);
    PIXEL_YC *dib_ycp1 = (PIXEL_YC *)_aligned_malloc(sizeof(PIXEL_YC) * w * h, 16);

    unsigned char *dib_yuy20 = (unsigned char *)_aligned_malloc(2 * w * h, 16);
    unsigned char *dib_yuy21 = (unsigned char *)_aligned_malloc(2 * w * h, 16);

    convert_color_func func_none;
    get_func(&func_none, NONE);

    std::mt19937 mt(1023);
    set_random_yc48(ycp0, yc48_pitch * h, mt);
    set_random_yc48(ycp1, yc48_pitch * h, mt);
    set_random_yc48(dib_ycp0, w * h, mt);
    set_random_yc48(dib_ycp1, w * h, mt);
    set_random_yuy2(dib_yuy20, w * h, mt);
    set_random_yuy2(dib_yuy21, w * h, mt);

    uint32_t SIMD_LIST[] = { SSE2, SSSE3, SSE41, AVX, AVX2 };
    uint32_t check_simd = 0x00;
    for (int isimd = 0; isimd < _countof(SIMD_LIST); isimd++) {
        auto check = [&](int ret, const char *check_type) {
            fprintf(stderr, "%4dx%4d, SIMD %s, %s: %s\n", w, h, simd_str(check_simd), check_type, ret ? "Error" : "OK");
            return ret;
        };
        check_simd |= SIMD_LIST[isimd];
        convert_color_func func_simd;
        get_func(&func_simd, check_simd);

        //YUY2 -> YC48
        cpinfo.pixelp = dib_yuy20;
        cpinfo.ycp = ycp0;
        func_none.yuy2_yc48(&cpinfo);
        cpinfo.ycp = ycp1;
        func_simd.yuy2_yc48(&cpinfo);
        ret |= check(compare_yc48(ycp0, ycp1, w, yc48_pitch, h), "YUY2      -> YC48     ");

        //YC48 -> YUY2
        cpinfo.ycp = ycp0;
        cpinfo.pixelp = dib_yuy20;
        func_none.yc48_yuy2(&cpinfo);
        cpinfo.pixelp = dib_yuy21;
        func_simd.yc48_yuy2(&cpinfo);
        ret |= check(compare_yuy2(dib_yuy20, dib_yuy21, w, h), "YC48      -> YUY2     ");

        //YC48(DIB) -> YC48
        cpinfo.pixelp = dib_ycp0;
        cpinfo.ycp = ycp0;
        func_none.yc48_btxxx_bt601(&cpinfo);
        cpinfo.ycp = ycp1;
        func_simd.yc48_btxxx_bt601(&cpinfo);
        ret |= check(compare_yc48(ycp0, ycp1, w, yc48_pitch, h), "YC48(DIB) -> YC48     ");

        //YC48 -> YC48(DIB)
        cpinfo.ycp = ycp0;
        cpinfo.pixelp = dib_ycp0;
        func_none.yc48_bt601_btxxx(&cpinfo);
        cpinfo.pixelp = dib_ycp1;
        func_simd.yc48_bt601_btxxx(&cpinfo);
        ret |= check(compare_yc48(dib_ycp0, dib_ycp1, w, w, h), "YC48      -> YC48(DIB)");
    }

    _aligned_free(ycp0);
    _aligned_free(ycp1);
    _aligned_free(dib_ycp0);
    _aligned_free(dib_ycp1);
    _aligned_free(dib_yuy20);
    _aligned_free(dib_yuy21);
    return ret;
}

#endif //__COLOR_TEST_H__
