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

#ifndef __CONVERT_CSP_H__
#define __CONVERT_CSP_H__

#include "color.h"

#ifndef CLAMP
#define CLAMP(x, low, high) (((x) > (high)) ? (high) : ((x) < (low))? (low) : (x))
#endif

struct PIXEL_YUV {
    unsigned char y, u, v;
};

struct CSP_CONVERT_MATRIX {
    int y1, y2;   //÷16384
    int cb1, cb2; //÷16384
    int cr1, cr2; //÷16384
};

static const CSP_CONVERT_MATRIX bt709_bt601 = {
     6657, 12953, //÷65536
    16218, -1813, //÷16384
    -1187, 16112  //÷16384
};

static const CSP_CONVERT_MATRIX bt2020nc_bt601 = {
     7726,  6915, //÷65536
    16306,  -976, //÷16384
    -1378, 15999  //÷16384
};

static const CSP_CONVERT_MATRIX bt601_bt709 = {
    -7746, -14030, //÷65536
    16689,   1878, //÷16384
     1230,  16799  //÷16384
};

static const CSP_CONVERT_MATRIX bt601_bt2020nc = {
    -8405, -7594, //÷65536
    16548,  1009, //÷16384
     1425, 16865  //÷16384
};

static PIXEL_YC inline convert_csp(PIXEL_YC src, CSP_CONVERT_MATRIX matrix) {
    PIXEL_YC dst;
    dst.y  = (short)CLAMP(src.y + ((src.cb * matrix.y1 + src.cr * matrix.y2) >> 16), SHRT_MIN, SHRT_MAX);
    dst.cb = (short)CLAMP((src.cb * matrix.cb1 + src.cr * matrix.cb2) >> 14, SHRT_MIN, SHRT_MAX);
    dst.cr = (short)CLAMP((src.cb * matrix.cr1 + src.cr * matrix.cr2) >> 14, SHRT_MIN, SHRT_MAX);
    return dst;
}

static PIXEL_YUV inline convert_yc48(PIXEL_YC src) {
    PIXEL_YUV dst;
    dst.y = (char)CLAMP(((src.y           * 219 + 383)>>12) + 16, 0, 255);
    dst.u = (char)CLAMP((((src.cb + 2048) *   7 +  66)>> 7) + 16, 0, 255);
    dst.v = (char)CLAMP((((src.cr + 2048) *   7 +  66)>> 7) + 16, 0, 255);
    return dst;
}

static PIXEL_YC inline convert_yc48(PIXEL_YUV src) {
    PIXEL_YC dst;
    dst.y  = (short)((src.y * 1197)>>6) - 299;
    dst.cb = (short)(((src.u - 128)*4681 + 164) >> 8);
    dst.cr = (short)(((src.v - 128)*4681 + 164) >> 8);
    return dst;
}

static __forceinline void convert_matrix_yc48(COLOR_PROC_INFO *cpip, const CSP_CONVERT_MATRIX matrix) {
    const int height = cpip->h;
    const int width = cpip->w;
    const int pitch = cpip->line_size;
    for (int y = 0; y < height; y++) {
        PIXEL_YC *ycp_dst = (PIXEL_YC *)((BYTE *)cpip->ycp + y * pitch);
        PIXEL_YC *ycp_src = (PIXEL_YC *)cpip->pixelp + y * width;
        for (int x = 0; x < width; x++, ycp_dst++, ycp_src++) {
            *ycp_dst = convert_csp(*ycp_src, matrix);
        }
    }
}

static __forceinline void convert_yuy2_yc48(COLOR_PROC_INFO *cpip, const CSP_CONVERT_MATRIX matrix) {
    const int height = cpip->h;
    const int width = cpip->w;
    const int pitch = cpip->line_size;
    for (int y = 0; y < height; y++) {
        PIXEL_YC *ycp = (PIXEL_YC *)((BYTE *)cpip->ycp + y * pitch);
        BYTE *pixelp = (BYTE *)cpip->pixelp + y * (((width + 1) / 2) * 4);
        for (int x = 0; x < width; x += 2) {
            PIXEL_YUV yuv ={ pixelp[0], pixelp[1], pixelp[3] };
            *ycp = convert_csp(convert_yc48(yuv), matrix);
            ycp++;
            const int idx = (x + 2 < width) ? 4 : 0;
            yuv.y = pixelp[2];
            yuv.u = (BYTE)(((int)pixelp[1] + pixelp[1+idx] + 1) >> 1);
            yuv.v = (BYTE)(((int)pixelp[3] + pixelp[3+idx] + 1) >> 1);
            *ycp = convert_csp(convert_yc48(yuv), matrix);
            ycp++;
            pixelp += 4;
        }
    }
}

static __forceinline void convert_yc48_yuy2(COLOR_PROC_INFO *cpip, const CSP_CONVERT_MATRIX matrix) {
    const int height = cpip->h;
    const int width = cpip->w;
    const int pitch = cpip->line_size;
    for (int y = 0; y < height; y++) {
        PIXEL_YC *ycp = (PIXEL_YC *)((BYTE *)cpip->ycp + y * pitch);
        BYTE *pixelp = (BYTE *)cpip->pixelp + y * (((width + 1) / 2) * 4);
        for (int x = 0; x < width; x += 2) {
            PIXEL_YUV yuv = convert_yc48(convert_csp(*ycp, matrix));
            pixelp[0] = yuv.y;
            pixelp[1] = yuv.u;
            pixelp[3] = yuv.v;
            ycp++;
            yuv = convert_yc48(convert_csp(*ycp, matrix));
            pixelp[2] = yuv.y;
            ycp++;
            pixelp += 4;
        }
    }
}

#endif //__CONVERT_CSP_H__
