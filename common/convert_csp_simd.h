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

#ifndef __CONVERT_CSP_SIMD_H__
#define __CONVERT_CSP_SIMD_H__

#include <cstdint>
#include <emmintrin.h> //イントリンシック命令 SSE2
#if USE_SSSE3
#include <tmmintrin.h> //イントリンシック命令 SSSE3
#endif
#if USE_SSE41
#include <smmintrin.h> //イントリンシック命令 SSE4.1
#endif
#include "color.h"
#include "convert_const.h"
#include "convert_csp.h"

#define _mm_store_switch_si128(ptr, xmm) ((aligned_store) ? _mm_store_si128(ptr, xmm) : _mm_storeu_si128(ptr, xmm))

#if USE_SSSE3
#define _mm_alignr_epi8_simd(a,b,i) _mm_alignr_epi8(a,b,i)
#else
#define _mm_alignr_epi8_simd(a,b,i) _mm_or_si128( _mm_slli_si128(a, 16-i), _mm_srli_si128(b, i) )
#endif

static __forceinline __m128i select_by_mask(__m128i a, __m128i b, __m128i mask) {
#if USE_SSE41
    return _mm_blendv_epi8(a, b, mask);
#else
    return _mm_or_si128( _mm_andnot_si128(mask,a), _mm_and_si128(b,mask) );
#endif
}

static __forceinline __m128i _mm_packus_epi32_simd(__m128i a, __m128i b) {
#if USE_SSE41
    return _mm_packus_epi32(a, b);
#else
    static const _declspec(align(64)) uint32_t VAL[2][4] = {
        { 0x00008000, 0x00008000, 0x00008000, 0x00008000 },
        { 0x80008000, 0x80008000, 0x80008000, 0x80008000 }
    };
#define LOAD_32BIT_0x8000 _mm_load_si128((__m128i *)VAL[0])
#define LOAD_16BIT_0x8000 _mm_load_si128((__m128i *)VAL[1])
    a = _mm_sub_epi32(a, LOAD_32BIT_0x8000);
    b = _mm_sub_epi32(b, LOAD_32BIT_0x8000);
    a = _mm_packs_epi32(a, b);
    return _mm_add_epi16(a, LOAD_16BIT_0x8000);
#undef LOAD_32BIT_0x8000
#undef LOAD_16BIT_0x8000
#endif
}

static void __forceinline memcpy_sse(char *dst, const char *src, int size) {
    if (size < 64) {
        for (int i = 0; i < size; i++)
            dst[i] = src[i];
        return;
    }
    char *dst_fin = (char *)dst + size;
    char *dst_aligned_fin = (char *)(((size_t)(dst_fin + 15) & ~15) - 64);
    __m128 x0, x1, x2, x3;
    const int start_align_diff = (int)((size_t)dst & 15);
    if (start_align_diff) {
        x0 = _mm_loadu_ps((float*)src);
        _mm_storeu_ps((float*)dst, x0);
        dst += 16 - start_align_diff;
        src += 16 - start_align_diff;
    }
    for (; dst < dst_aligned_fin; dst += 64, src += 64) {
        x0 = _mm_loadu_ps((float*)(src +  0));
        x1 = _mm_loadu_ps((float*)(src + 16));
        x2 = _mm_loadu_ps((float*)(src + 32));
        x3 = _mm_loadu_ps((float*)(src + 48));
        _mm_store_ps((float*)(dst +  0), x0);
        _mm_store_ps((float*)(dst + 16), x1);
        _mm_store_ps((float*)(dst + 32), x2);
        _mm_store_ps((float*)(dst + 48), x3);
    }
    char *dst_tmp = dst_fin - 64;
    src -= (dst - dst_tmp);
    x0 = _mm_loadu_ps((float*)(src +  0));
    x1 = _mm_loadu_ps((float*)(src + 16));
    x2 = _mm_loadu_ps((float*)(src + 32));
    x3 = _mm_loadu_ps((float*)(src + 48));
    _mm_storeu_ps((float*)(dst_tmp +  0), x0);
    _mm_storeu_ps((float*)(dst_tmp + 16), x1);
    _mm_storeu_ps((float*)(dst_tmp + 32), x2);
    _mm_storeu_ps((float*)(dst_tmp + 48), x3);
}

static __forceinline void convert_csp_y_cbcr(__m128i& xY, __m128i& xCbCrEven, __m128i& xCbCrOdd, const CSP_CONVERT_MATRIX matrix) {
#if MATRIX_CONVERSION
    __m128i xCb = _mm_or_si128(_mm_and_si128(xCbCrEven, _mm_set1_epi32(0xffff)), _mm_slli_epi32(xCbCrOdd, 16));
    __m128i xCr = _mm_or_si128(_mm_srli_epi32(xCbCrEven, 16), _mm_andnot_si128(_mm_set1_epi32(0xffff), xCbCrOdd));
    xY = _mm_add_epi16(_mm_add_epi16(xY, _mm_mulhi_epi16(xCb, _mm_set1_epi16(matrix.y1))), _mm_mulhi_epi16(xCr, _mm_set1_epi16(matrix.y2)));

    __m128i xCbCrCbCrEvenLo = _mm_unpacklo_epi32(xCbCrEven, xCbCrEven);
    __m128i xCbCrCbCrEvenHi = _mm_unpackhi_epi32(xCbCrEven, xCbCrEven);
    __m128i xCbCrCbCrOddLo = _mm_unpacklo_epi32(xCbCrOdd, xCbCrOdd);
    __m128i xCbCrCbCrOddHi = _mm_unpackhi_epi32(xCbCrOdd, xCbCrOdd);

    const __m128i xMul = _mm_set1_epi64x(
          (int64_t)matrix.cb1 |
        (((int64_t)matrix.cb2 << 16) & (int64_t)0x00000000ffff0000) |
        (((int64_t)matrix.cr1 << 32) & (int64_t)0x0000ffff00000000) |
         ((int64_t)matrix.cr2 << 48));
    xCbCrEven = _mm_packs_epi32(_mm_srai_epi32(_mm_madd_epi16(xCbCrCbCrEvenLo, xMul), 14),
                                _mm_srai_epi32(_mm_madd_epi16(xCbCrCbCrEvenHi, xMul), 14));
    xCbCrOdd = _mm_packs_epi32(_mm_srai_epi32(_mm_madd_epi16(xCbCrCbCrOddLo, xMul), 14),
                               _mm_srai_epi32(_mm_madd_epi16(xCbCrCbCrOddHi, xMul), 14));
#endif //#if MATRIX_CONVERSION
}

static __forceinline void gather_y_uv_from_yc48(__m128i& x0, __m128i& x1, __m128i& x2) {
    __m128i x3, x4, x5;
#if USE_SSE41
    const int MASK_INT_Y  = 0x80 + 0x10 + 0x02;
    const int MASK_INT_UV = 0x40 + 0x20 + 0x01;
    x3 = _mm_blend_epi16(x0, x1, MASK_INT_Y);
    x3 = _mm_blend_epi16(x3, x2, MASK_INT_Y>>2);

    x4 = _mm_blend_epi16(x0, x1, MASK_INT_UV);
    x4 = _mm_blend_epi16(x4, x2, MASK_INT_UV>>2);

    x5 = _mm_blend_epi16(x0, x1, 0x08 + 0x04);
    x5 = _mm_blend_epi16(x5, x2, 0x80 + 0x40 + 0x02 + 0x01);

    x0 = _mm_shuffle_epi8(x3, xC_SUFFLE_YCP_Y);

    x4 = _mm_alignr_epi8_simd(x4, x4, 2);
    x1 = _mm_shuffle_epi32(x4, _MM_SHUFFLE(1, 2, 3, 0));//UV偶数
    x2 = _mm_shuffle_epi32(x5, _MM_SHUFFLE(3, 0, 1, 2));//UV奇数
#else
    x3 = select_by_mask(x0, x1, xC_MASK_YCP2Y(0));
    x3 = select_by_mask(x3, x2, xC_MASK_YCP2Y(1));

    x4 = select_by_mask(x0, x1, xC_MASK_YCP2UV(0));
    x4 = select_by_mask(x4, x2, xC_MASK_YCP2UV(1));

    x5 = select_by_mask(x0, x1, xC_MASK_YCP2UV(2));
    x5 = select_by_mask(x5, x2, xC_MASK_YCP2UV(3));

    x4 = _mm_alignr_epi8_simd(x4, x4, 2);
    x1 = _mm_shuffle_epi32(x4, _MM_SHUFFLE(1, 2, 3, 0));//UV偶数
    x2 = _mm_shuffle_epi32(x5, _MM_SHUFFLE(3, 0, 1, 2));//UV奇数
#if USE_SSSE3
    x0 = _mm_shuffle_epi8(x3, xC_SUFFLE_YCP_Y);
#else
    x0 = _mm_shuffle_epi32(x3, _MM_SHUFFLE(3, 1, 2, 0));
    x0 = _mm_shufflehi_epi16(x0, _MM_SHUFFLE(1, 2, 3, 0));
    x0 = _mm_shuffle_epi32(x0, _MM_SHUFFLE(1, 2, 3, 0));
    x0 = _mm_shufflelo_epi16(x0, _MM_SHUFFLE(1, 2, 3, 0));
    x0 = _mm_shufflehi_epi16(x0, _MM_SHUFFLE(3, 0, 1, 2));
#endif //USE_SSSE3
#endif //USE_SSE41
}

static __forceinline void gather_y_u_v_from_yc48(__m128i& x0, __m128i& x1, __m128i& x2) {
#if USE_SSE41
    __m128i x3, x4, x5;
    const int MASK_INT = 0x40 + 0x08 + 0x01;
    x3 = _mm_blend_epi16(x2, x0, MASK_INT);
    x4 = _mm_blend_epi16(x1, x2, MASK_INT);
    x5 = _mm_blend_epi16(x0, x1, MASK_INT);

    x3 = _mm_blend_epi16(x3, x1, MASK_INT<<1);
    x4 = _mm_blend_epi16(x4, x0, MASK_INT<<1);
    x5 = _mm_blend_epi16(x5, x2, MASK_INT<<1);

    x0 = _mm_shuffle_epi8(x3, xC_SUFFLE_YCP_Y);
    x1 = _mm_shuffle_epi8(x4, _mm_alignr_epi8_simd(xC_SUFFLE_YCP_Y, xC_SUFFLE_YCP_Y, 6));
    x2 = _mm_shuffle_epi8(x5, _mm_alignr_epi8_simd(xC_SUFFLE_YCP_Y, xC_SUFFLE_YCP_Y, 12));
#else
    //code from afs v7.5a+10
    __m128i x5, x6, x7, xMask;
    //select y
    static const _declspec(align(16)) uint16_t maskY_select[8] = { 0xffff, 0x0000, 0x0000, 0xffff, 0x0000, 0x0000, 0xffff, 0x0000 };
    xMask = _mm_load_si128((__m128i*)maskY_select);

    x5 = select_by_mask(x2, x0, xMask);
    xMask = _mm_slli_si128(xMask, 2);
    x5 = select_by_mask(x5, x1, xMask); //52741630

    x6 = _mm_unpacklo_epi16(x5, x5);    //11663300
    x7 = _mm_unpackhi_epi16(x5, x5);    //55227744

    static const _declspec(align(16)) uint16_t maskY_shuffle[8] = { 0xffff, 0x0000, 0xffff, 0x0000, 0x0000, 0xffff, 0xffff, 0x0000 };
    xMask = _mm_load_si128((__m128i*)maskY_shuffle);
    x5 = select_by_mask(x7, x6, xMask);                 //51627340
    x5 = _mm_shuffle_epi32(x5, _MM_SHUFFLE(1, 2, 3, 0));   //73625140

    x5 = _mm_unpacklo_epi16(x5, _mm_srli_si128(x5, 8)); //75316420
    x5 = _mm_unpacklo_epi16(x5, _mm_srli_si128(x5, 8)); //76543210

                                                        //select uv
    xMask = _mm_srli_si128(_mm_cmpeq_epi8(xMask, xMask), 8); //0x00000000, 0x00000000, 0xffffffff, 0xffffffff
    x6 = select_by_mask(_mm_srli_si128(x1, 2), _mm_srli_si128(x2, 2), xMask); //x  x v4 u4 v6 u6 x  x 
    x7 = select_by_mask(x0, x1, xMask);               //x  x  v1 u1 v3 u3 x  x
    xMask = _mm_slli_si128(xMask, 4);                 //0x00000000, 0xffffffff, 0xffffffff, 0x00000000
    x0 = _mm_alignr_epi8_simd(x1, x0, 2);             //v2 u2  x  x  x  x v0 u0
    x6 = select_by_mask(x0, x6, xMask);               //v2 u2 v4 u4 v6 u6 v0 u0
    x7 = select_by_mask(x2, x7, xMask);               //v7 u7 v1 u1 v3 u3 v5 u5
    x0 = _mm_shuffle_epi32(x6, _MM_SHUFFLE(1, 2, 3, 0)); //v6 u6 v4 u4 v2 u2 v0 u0
    x1 = _mm_shuffle_epi32(x7, _MM_SHUFFLE(3, 0, 1, 2)); //v7 u7 v5 u5 v3 u3 v1 u1

    x6 = _mm_unpacklo_epi16(x0, x1); //v3 v2 u3 u2 v1 v0 u1 u0
    x7 = _mm_unpackhi_epi16(x0, x1); //v7 v6 u7 u6 v5 v4 u5 u4

    x0 = _mm_unpacklo_epi32(x6, x7); //v5 v4 v1 v0 u5 u4 u1 u0
    x1 = _mm_unpackhi_epi32(x6, x7); //v7 v6 v3 v2 u7 u6 u3 u2

    x6 = _mm_unpacklo_epi32(x0, x1); //u7 u6 u5 u4 u3 u2 u1 u0
    x7 = _mm_unpackhi_epi32(x0, x1); //v7 v6 v5 v4 v3 v2 v1 v0

    x0 = x5;
    x1 = x6;
    x2 = x7;
#endif //USE_SSE41
}

static __forceinline void gather_y_uv_from_yc48(const void *ptr_src, __m128i& x0, __m128i& x1, __m128i& x2) {
    x0 = _mm_loadu_si128((const __m128i *)((const char *)ptr_src +  0));
    x1 = _mm_loadu_si128((const __m128i *)((const char *)ptr_src + 16));
    x2 = _mm_loadu_si128((const __m128i *)((const char *)ptr_src + 32));
    gather_y_uv_from_yc48(x0, x1, x2);
}

static __forceinline void gather_y_u_v_from_yc48(const void *ptr_src, __m128i& x0, __m128i& x1, __m128i& x2) {
    x0 = _mm_loadu_si128((const __m128i *)((const char *)ptr_src +  0));
    x1 = _mm_loadu_si128((const __m128i *)((const char *)ptr_src + 16));
    x2 = _mm_loadu_si128((const __m128i *)((const char *)ptr_src + 32));
    gather_y_u_v_from_yc48(x0, x1, x2);
}

static __forceinline void afs_pack_yc48(__m128i& x0, __m128i& x1, __m128i& x2, const __m128i& xY, const __m128i& xC0, __m128i& xC1) {
    __m128i xYtemp, xCtemp0, xCtemp1;
#if USE_SSSE3
    xYtemp = _mm_shuffle_epi8(xY, xC_SUFFLE_YCP_Y); //52741630
#else
    //select y
    __m128i x6, x7;
    x6 = _mm_unpacklo_epi16(xY, xY);    //33221100
    x7 = _mm_unpackhi_epi16(xY, xY);    //77665544

    x6 = _mm_shuffle_epi32(x6, _MM_SHUFFLE(0, 2, 1, 3)); //00221133
    x6 = _mm_alignr_epi8_simd(x6, x6, 2);                //30022113
    x6 = _mm_shuffle_epi32(x6, _MM_SHUFFLE(2, 1, 0, 3)); //02211330

    x7 = _mm_shuffle_epi32(x7, _MM_SHUFFLE(0, 2, 1, 3)); //44665577
    x7 = _mm_alignr_epi8_simd(x7, x7, 2);                //74466557
    x7 = _mm_shuffle_epi32(x7, _MM_SHUFFLE(0, 3, 2, 1)); //57744665

    static const __declspec(align(16)) uint16_t MASK_Y[] ={
        0x0000, 0x0000, 0xffff, 0x0000, 0xffff, 0xffff, 0x0000, 0xffff,
    };
    xYtemp = select_by_mask(x6, x7, _mm_load_si128((const __m128i *)MASK_Y)); //52741630
#endif
    xCtemp1 = _mm_shuffle_epi32(xC1, _MM_SHUFFLE(3, 0, 1, 2));
    xCtemp0 = _mm_shuffle_epi32(xC0, _MM_SHUFFLE(1, 2, 3, 0));
    xCtemp0 = _mm_alignr_epi8_simd(xCtemp0, xCtemp0, 14);
#if USE_SSE41
    x0 = _mm_blend_epi16(xYtemp, xCtemp0, 0x80+0x04+0x02);
    x1 = _mm_blend_epi16(xYtemp, xCtemp1, 0x08+0x04);
    x2 = _mm_blend_epi16(xYtemp, xCtemp0, 0x10+0x08);
    x2 = _mm_blend_epi16(x2, xCtemp1, 0x80+0x40+0x02+0x01);
    x1 = _mm_blend_epi16(x1, xCtemp0, 0x40+0x20+0x01);
    x0 = _mm_blend_epi16(x0, xCtemp1, 0x20+0x10);
#else
    static const __declspec(align(16)) uint16_t MASK[] ={
        0x0000, 0xffff, 0xffff, 0x0000, 0x0000, 0x0000, 0x0000, 0xffff,
        0xffff, 0xffff, 0x0000, 0x0000, 0x0000, 0x0000, 0xffff, 0xffff
    };
    __m128i xMask = _mm_load_si128((const __m128i *)MASK);
    x0 = select_by_mask(xYtemp, xCtemp0, xMask);
    x1 = select_by_mask(xYtemp, xCtemp1, _mm_slli_si128(xMask, 2));
    x2 = select_by_mask(xYtemp, xCtemp0, _mm_slli_si128(xMask, 4));
    xMask = _mm_load_si128((const __m128i *)(MASK + 8));
    x2 = select_by_mask(x2, xCtemp1, xMask);
    x1 = select_by_mask(x1, xCtemp0, _mm_srli_si128(xMask, 2));
    x0 = select_by_mask(x0, xCtemp1, _mm_srli_si128(xMask, 4));
#endif
}

static __forceinline void store_yc48(void *ptr_dst, __m128i& xY, __m128i& xCbCrEven, __m128i& xCbCrOdd) {
    __m128i x0, x1, x2;
    afs_pack_yc48(x0, x1, x2, xY, xCbCrEven, xCbCrOdd);
    _mm_storeu_si128((__m128i *)((char *)ptr_dst +  0), x0);
    _mm_storeu_si128((__m128i *)((char *)ptr_dst + 16), x1);
    _mm_storeu_si128((__m128i *)((char *)ptr_dst + 32), x2);
}

template<bool SRC_DIB>
static __forceinline void convert_matrix_yc48_simd(COLOR_PROC_INFO *cpip, const CSP_CONVERT_MATRIX matrix) {
    const int height = cpip->h;
    const int width = cpip->w;
    const int x_fin = width - 8;
    int src_pitch = (SRC_DIB) ? sizeof(PIXEL_YC) * cpip->w : cpip->line_size;
    int dst_pitch = (SRC_DIB) ? cpip->line_size : sizeof(PIXEL_YC) * cpip->w;
    const char *ycp_src_line = (const char *)((SRC_DIB) ? cpip->pixelp : cpip->ycp);
    char *ycp_dst_line = (char *)((SRC_DIB) ? cpip->ycp : cpip->pixelp);
    __m128i xY, xCbCrEven, xCbCrOdd;
    for (int y = 0; y < height; y++, ycp_dst_line += dst_pitch, ycp_src_line += src_pitch) {
#if MATRIX_CONVERSION
        const char *ptr_src = ycp_src_line;
        char *ptr_dst = ycp_dst_line;
        int x = 0;
        for (; x < x_fin; x += 8, ptr_dst += 48, ptr_src += 48) {
            gather_y_uv_from_yc48(ptr_src, xY, xCbCrEven, xCbCrOdd);
            convert_csp_y_cbcr(xY, xCbCrEven, xCbCrOdd, matrix);
            store_yc48(ptr_dst, xY, xCbCrEven, xCbCrOdd);
        }
        int offset = x - (width - 8);
        if (offset > 0) {
            ptr_src -= offset * sizeof(PIXEL_YC);
            ptr_dst -= offset * sizeof(PIXEL_YC);
        }
        gather_y_uv_from_yc48(ptr_src, xY, xCbCrEven, xCbCrOdd);
        convert_csp_y_cbcr(xY, xCbCrEven, xCbCrOdd, matrix);
        store_yc48(ptr_dst, xY, xCbCrEven, xCbCrOdd);
#else
        memcpy_sse(ycp_dst_line, ycp_src_line, sizeof(PIXEL_YC) * cpip->w);
#endif
    }
}

static __forceinline void convert_range_y_yuy2_to_yc48(__m128i& x0, __m128i& x1) {
    x1 = _mm_unpackhi_epi8(x0, _mm_setzero_si128());
    x0 = _mm_unpacklo_epi8(x0, _mm_setzero_si128());
    x0 = _mm_slli_epi16(x0, 6);
    x1 = _mm_slli_epi16(x1, 6);
    x0 = _mm_mulhi_epi16(x0, _mm_set1_epi16(19152));
    x1 = _mm_mulhi_epi16(x1, _mm_set1_epi16(19152));
    x0 = _mm_sub_epi16(x0, _mm_set1_epi16(299));
    x1 = _mm_sub_epi16(x1, _mm_set1_epi16(299));
}

static __forceinline void convert_range_c_yuy2_to_yc48(__m128i& x0, __m128i& x1) {
    x1 = _mm_unpackhi_epi8(x0, _mm_setzero_si128());
    x0 = _mm_unpacklo_epi8(x0, _mm_setzero_si128());
    x0 = _mm_slli_epi16(x0, 6);
    x1 = _mm_slli_epi16(x1, 6);
    x0 = _mm_mulhi_epi16(x0, _mm_set1_epi16(18752));
    x1 = _mm_mulhi_epi16(x1, _mm_set1_epi16(18752));
    x0 = _mm_sub_epi16(x0, _mm_set1_epi16(2340));
    x1 = _mm_sub_epi16(x1, _mm_set1_epi16(2340));
}

static __forceinline __m128i afs_uv_interp_linear(const __m128i& x1, const __m128i& x2) {
    __m128i x3, x4;
    x3 = _mm_alignr_epi8_simd(x2, x1, 4);
    x4 = _mm_add_epi16(x1, _mm_srli_epi16(_mm_cmpeq_epi16(x1, x1), 15));
    x3 = _mm_add_epi16(x3, x4);
    return _mm_srai_epi16(x3, 1);
}

static __forceinline void convert_yuy2_yc48_simd(COLOR_PROC_INFO *cpip, const CSP_CONVERT_MATRIX matrix) {
    const int height = cpip->h;
    const int width = cpip->w;
    const int x_fin = width - 16;
    const int dst_pitch = cpip->line_size;
    const int src_pitch = ((width + 1) / 2) * 4;
    const char *ycp_src_line = (const char *)cpip->pixelp;
    char *ycp_dst_line = (char *)cpip->ycp;
    for (int y = 0; y < height; y++, ycp_dst_line += dst_pitch, ycp_src_line += src_pitch) {
        const char *ptr_src = ycp_src_line;
        char *ptr_dst = ycp_dst_line;
        int x = 0;
        __m128i m0, m1;
        __m128i my0a, my0b, my4a, mc0a, mc0b, mc4a, mc4b;
        m0 = _mm_loadu_si128((const __m128i *)ptr_src);
        m1 = _mm_loadu_si128((const __m128i *)(ptr_src + 16));

        my0a = _mm_packus_epi16(_mm_and_si128(m0, _mm_set1_epi16(0xff)), _mm_and_si128(m1, _mm_set1_epi16(0xff)));
        mc0a = _mm_packus_epi16(_mm_srli_epi16(m0, 8), _mm_srli_epi16(m1, 8));
        convert_range_c_yuy2_to_yc48(mc0a, mc0b);
        //mc00 = _mm_shuffle_epi32(mc0a, _MM_SHUFFLE(0, 0, 0, 0));

        for (; x < x_fin; x += 16, ptr_dst += 96, ptr_src += 32) {
            _mm_prefetch(ptr_src + 64, _MM_HINT_NTA);
            m0 = _mm_loadu_si128((const __m128i *)(ptr_src + 32));
            m1 = _mm_loadu_si128((const __m128i *)(ptr_src + 48));
            my4a = _mm_packus_epi16(_mm_and_si128(m0, _mm_set1_epi16(0xff)), _mm_and_si128(m1, _mm_set1_epi16(0xff)));
            mc4a = _mm_packus_epi16(_mm_srli_epi16(m0, 8), _mm_srli_epi16(m1, 8));

            convert_range_y_yuy2_to_yc48(my0a, my0b);
            convert_range_c_yuy2_to_yc48(mc4a, mc4b);

            __m128i mc1a = afs_uv_interp_linear(mc0a, mc0b);
            __m128i mc1b = afs_uv_interp_linear(mc0b, mc4a);

            convert_csp_y_cbcr(my0a, mc0a, mc1a, matrix);
            convert_csp_y_cbcr(my0b, mc0b, mc1b, matrix);

            __m128i x0, x1, x2;
            afs_pack_yc48(x0, x1, x2, my0a, mc0a, mc1a);

            _mm_stream_si128((__m128i*)((uint8_t *)ptr_dst +  0), x0);
            _mm_stream_si128((__m128i*)((uint8_t *)ptr_dst + 16), x1);
            _mm_stream_si128((__m128i*)((uint8_t *)ptr_dst + 32), x2);

            afs_pack_yc48(x0, x1, x2, my0b, mc0b, mc1b);

            _mm_stream_si128((__m128i*)((uint8_t *)ptr_dst + 48), x0);
            _mm_stream_si128((__m128i*)((uint8_t *)ptr_dst + 64), x1);
            _mm_stream_si128((__m128i*)((uint8_t *)ptr_dst + 80), x2);

            my0a = my4a;
            mc0a = mc4a;
            mc0b = mc4b;
        }
        int offset = x - (width - 16);
        if (offset > 0) {
            ptr_dst -= offset * sizeof(PIXEL_YC);
            ptr_src -= offset * 2;
            m0 = _mm_loadu_si128((const __m128i *)ptr_src);
            m1 = _mm_loadu_si128((const __m128i *)(ptr_src + 16));

            my0a = _mm_packus_epi16(_mm_and_si128(m0, _mm_set1_epi16(0xff)), _mm_and_si128(m1, _mm_set1_epi16(0xff)));
            mc0a = _mm_packus_epi16(_mm_srli_epi16(m0, 8), _mm_srli_epi16(m1, 8));
            convert_range_c_yuy2_to_yc48(mc0a, mc0b);
            //mc00 = _mm_shuffle_epi32(mc0a, _MM_SHUFFLE(0, 0, 0, 0));
        }

        convert_range_y_yuy2_to_yc48(my0a, my0b);
        mc4a = _mm_shuffle_epi32(mc0b, _MM_SHUFFLE(3, 3, 3, 3));

        __m128i mc1a = afs_uv_interp_linear(mc0a, mc0b);
        __m128i mc1b = afs_uv_interp_linear(mc0b, mc4a);

        convert_csp_y_cbcr(my0a, mc0a, mc1a, matrix);
        convert_csp_y_cbcr(my0b, mc0b, mc1b, matrix);

        __m128i x0, x1, x2;
        afs_pack_yc48(x0, x1, x2, my0a, mc0a, mc1a);

        _mm_storeu_si128((__m128i*)((uint8_t *)ptr_dst +  0), x0);
        _mm_storeu_si128((__m128i*)((uint8_t *)ptr_dst + 16), x1);
        _mm_storeu_si128((__m128i*)((uint8_t *)ptr_dst + 32), x2);

        afs_pack_yc48(x0, x1, x2, my0b, mc0b, mc1b);

        _mm_storeu_si128((__m128i*)((uint8_t *)ptr_dst + 48), x0);
        _mm_storeu_si128((__m128i*)((uint8_t *)ptr_dst + 64), x1);
        _mm_storeu_si128((__m128i*)((uint8_t *)ptr_dst + 80), x2);
    }
}

static __forceinline __m128i convert_y_range_from_yc48(__m128i x0, const __m128i& xC_Y_MA_16, int Y_RSH_16, const __m128i& xC_YCC, const __m128i& xC_pw_one) {
    __m128i x7;
    x7 = _mm_unpackhi_epi16(x0, xC_pw_one);
    x0 = _mm_unpacklo_epi16(x0, xC_pw_one);

    x0 = _mm_madd_epi16(x0, xC_Y_MA_16);
    x7 = _mm_madd_epi16(x7, xC_Y_MA_16);
    x0 = _mm_srai_epi32(x0, Y_RSH_16);
    x7 = _mm_srai_epi32(x7, Y_RSH_16);
    x0 = _mm_add_epi32(x0, xC_YCC);
    x7 = _mm_add_epi32(x7, xC_YCC);

    x0 = _mm_packus_epi32_simd(x0, x7);

    return x0;
}

static __forceinline __m128i convert_uv_range_after_adding_offset(__m128i x0, const __m128i& xC_UV_MA_16, int UV_RSH_16, const __m128i& xC_YCC, const __m128i& xC_pw_one) {
    __m128i x1;
    x1 = _mm_unpackhi_epi16(x0, xC_pw_one);
    x0 = _mm_unpacklo_epi16(x0, xC_pw_one);

    x0 = _mm_madd_epi16(x0, xC_UV_MA_16);
    x1 = _mm_madd_epi16(x1, xC_UV_MA_16);
    x0 = _mm_srai_epi32(x0, UV_RSH_16);
    x1 = _mm_srai_epi32(x1, UV_RSH_16);
    x0 = _mm_add_epi32(x0, xC_YCC);
    x1 = _mm_add_epi32(x1, xC_YCC);

    x0 = _mm_packus_epi32_simd(x0, x1);

    return x0;
}

static __forceinline __m128i convert_uv_range_from_yc48(__m128i x0, const __m128i& xC_UV_OFFSET_x1, const __m128i& xC_UV_MA_16, int UV_RSH_16, __m128i xC_YCC, const __m128i& xC_pw_one) {
    x0 = _mm_add_epi16(x0, xC_UV_OFFSET_x1);

    return convert_uv_range_after_adding_offset(x0, xC_UV_MA_16, UV_RSH_16, xC_YCC, xC_pw_one);
}

static __forceinline void convert_yc48_yuy2_simd(void *ptr_dst, const void *ptr_src, const CSP_CONVERT_MATRIX matrix) {
    const __m128i xC_pw_one = _mm_set1_epi16(1);
    __m128i xY, xCbCrEven, xCbCrOdd;
    gather_y_uv_from_yc48(ptr_src, xY, xCbCrEven, xCbCrOdd);
    convert_csp_y_cbcr(xY, xCbCrEven, xCbCrOdd, matrix);
    xY = convert_y_range_from_yc48(xY, xC_Y_L_MA_8, Y_L_RSH_8, _mm_set1_epi32(1<<LSFT_YCC_8), xC_pw_one);
    xCbCrEven = convert_uv_range_from_yc48(xCbCrEven, _mm_set1_epi16(UV_OFFSET_x1), xC_UV_L_MA_8_444, UV_L_RSH_8_444, _mm_set1_epi32(1<<LSFT_YCC_8), xC_pw_one);

    _mm_storeu_si128((__m128i *)ptr_dst,
        _mm_or_si128(_mm_and_si128(xY, _mm_set1_epi16(0xff)), _mm_slli_epi16(xCbCrEven, 8)));
}

static __forceinline void convert_yc48_yuy2_simd(COLOR_PROC_INFO *cpip, const CSP_CONVERT_MATRIX matrix) {
    const int height = cpip->h;
    const int width = cpip->w;
    const int x_fin = width - 8;
    const int pitch = cpip->line_size;
    const int src_pitch = cpip->line_size;
    const int dst_pitch = ((width + 1) / 2) * 4;
    const char *ycp_src_line = (const char *)cpip->ycp;
    char *ycp_dst_line = (char *)cpip->pixelp;
    for (int y = 0; y < height; y++, ycp_dst_line += dst_pitch, ycp_src_line += src_pitch) {
        const char *ptr_src = ycp_src_line;
        char *ptr_dst = ycp_dst_line;
        int x = 0;
        for (; x < x_fin; x += 8, ptr_dst += 16, ptr_src += 48) {
            convert_yc48_yuy2_simd(ptr_dst, ptr_src, matrix);
        }
        int offset = x - (width - 8);
        if (offset > 0) {
            ptr_src -= offset * sizeof(PIXEL_YC);
            ptr_dst -= offset * 2;
        }
        convert_yc48_yuy2_simd(ptr_dst, ptr_src, matrix);
    }
}


#endif //__CONVERT_CSP_H__
