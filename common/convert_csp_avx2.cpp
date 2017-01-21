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
#define USE_AVX2  1

#define NOMINMAX
#include <windows.h>
#include <cstdint>
#include <immintrin.h> //イントリンシック命令 AVX/AVX2
#include "color.h"
#include "convert_const.h"
#include "convert_csp.h"

#define _mm256_store_switch_si256(ptr, ymm)  ((aligned_store) ? _mm256_store_si256(ptr, ymm)  : _mm256_storeu_si256(ptr, ymm))
#define _mm256_stream_switch_si256(ptr, ymm) ((aligned_store) ? _mm256_stream_si256(ptr, ymm) : _mm256_storeu_si256(ptr, ymm))
//本来の256bit alignr
#define MM_ABS(x) (((x) < 0) ? -(x) : (x))
#define _mm256_alignr256_epi8(a, b, i) ((i<=16) ? _mm256_alignr_epi8(_mm256_permute2x128_si256(a, b, (0x00<<4) + 0x03), b, i) : _mm256_alignr_epi8(a, _mm256_permute2x128_si256(a, b, (0x00<<4) + 0x03), MM_ABS(i-16)))

//_mm256_srli_si256, _mm256_slli_si256は
//単に128bitシフト×2をするだけの命令である
#define _mm256_bsrli_epi128 _mm256_srli_si256
#define _mm256_bslli_epi128 _mm256_slli_si256
//本当の256bitシフト
#define _mm256_srli256_si256(a, i) ((i<=16) ? _mm256_alignr_epi8(_mm256_permute2x128_si256(a, a, (0x08<<4) + 0x03), a, i) : _mm256_bsrli_epi128(_mm256_permute2x128_si256(a, a, (0x08<<4) + 0x03), MM_ABS(i-16)))
#define _mm256_slli256_si256(a, i) ((i<=16) ? _mm256_alignr_epi8(a, _mm256_permute2x128_si256(a, a, (0x00<<4) + 0x08), MM_ABS(16-i)) : _mm256_bslli_epi128(_mm256_permute2x128_si256(a, a, (0x00<<4) + 0x08), MM_ABS(i-16)))


static __forceinline void convert_csp_y_cbcr(__m256i& yY, __m256i& yCbCrEven, __m256i& yCbCrOdd, const CSP_CONVERT_MATRIX matrix) {
    __m256i yCb = _mm256_or_si256(_mm256_and_si256(yCbCrEven, _mm256_set1_epi32(0xffff)), _mm256_srli_epi32(yCbCrOdd, 16));
    __m256i yCr = _mm256_or_si256(_mm256_slli_epi32(yCbCrEven, 16), _mm256_andnot_si256(_mm256_set1_epi32(0xffff), yCbCrOdd));
    yY = _mm256_add_epi16(_mm256_add_epi16(yY, _mm256_mulhi_epi16(yCb, _mm256_set1_epi16(matrix.y1))), _mm256_mulhi_epi16(yCr, _mm256_set1_epi16(matrix.y2)));

    __m256i yCbCrCbCrEvenLo = _mm256_unpacklo_epi32(yCbCrEven, yCbCrEven);
    __m256i yCbCrCbCrEvenHi = _mm256_unpackhi_epi32(yCbCrEven, yCbCrEven);
    __m256i yCbCrCbCrOddLo = _mm256_unpacklo_epi32(yCbCrOdd, yCbCrOdd);
    __m256i yCbCrCbCrOddHi = _mm256_unpackhi_epi32(yCbCrOdd, yCbCrOdd);

    const __m256i yMul = _mm256_set1_epi64x(
          (int64_t)matrix.cb1 |
        (((int64_t)matrix.cb2 << 16) & (int64_t)0x00000000ffff0000) |
        (((int64_t)matrix.cr1 << 32) & (int64_t)0x0000ffff00000000) |
         ((int64_t)matrix.cr2 << 48));
    yCbCrEven = _mm256_packs_epi32(_mm256_srai_epi32(_mm256_madd_epi16(yCbCrCbCrEvenLo, yMul), 14),
                                   _mm256_srai_epi32(_mm256_madd_epi16(yCbCrCbCrEvenHi, yMul), 14));
    yCbCrOdd = _mm256_packs_epi32(_mm256_srai_epi32(_mm256_madd_epi16(yCbCrCbCrOddLo, yMul), 14),
                                  _mm256_srai_epi32(_mm256_madd_epi16(yCbCrCbCrOddHi, yMul), 14));
}

static __forceinline void gather_y_uv_from_yc48(__m256i& y0, __m256i& y1, __m256i& y2) {
    __m256i y3 = y0;
    __m256i y4 = y1;
    __m256i y5 = y2;
    const int MASK_INT_Y  = 0x80 + 0x10 + 0x02;
    const int MASK_INT_UV = 0x40 + 0x20 + 0x01;

    y0 = _mm256_blend_epi32(y3, y4, 0xf0);                    // 384, 0
    y1 = _mm256_permute2x128_si256(y3, y5, (0x02<<4) + 0x01); // 512, 128
    y2 = _mm256_blend_epi32(y4, y5, 0xf0);                    // 640, 256

    y3 = _mm256_blend_epi16(y0, y1, MASK_INT_Y);
    y3 = _mm256_blend_epi16(y3, y2, MASK_INT_Y>>2);

    y4 = _mm256_blend_epi16(y0, y1, MASK_INT_UV);
    y4 = _mm256_blend_epi16(y4, y2, MASK_INT_UV>>2);

    y5 = _mm256_blend_epi16(y0, y1, 0x08 + 0x04);
    y5 = _mm256_blend_epi16(y5, y2, 0x80 + 0x40 + 0x02 + 0x01);

    y0 = _mm256_shuffle_epi8(y3, yC_SUFFLE_YCP_Y);

    y4 = _mm256_alignr_epi8(y4, y4, 2);
    y1 = _mm256_shuffle_epi32(y4, _MM_SHUFFLE(1, 2, 3, 0));//UV偶数
    y2 = _mm256_shuffle_epi32(y5, _MM_SHUFFLE(3, 0, 1, 2));//UV奇数
}

static __forceinline void gather_y_u_v_from_yc48(__m256i& y0, __m256i& y1, __m256i& y2) {
    __m256i y3, y4, y5;
    const int MASK_INT = 0x40 + 0x08 + 0x01;
    y3 = _mm256_blend_epi32(y0, y1, 0xf0);                    // 384, 0
    y4 = _mm256_permute2x128_si256(y0, y2, (0x02<<4) + 0x01); // 512, 128
    y5 = _mm256_blend_epi32(y1, y2, 0xf0);                    // 640, 256

    y0 = _mm256_blend_epi16(y5, y3, MASK_INT);
    y1 = _mm256_blend_epi16(y4, y5, MASK_INT);
    y2 = _mm256_blend_epi16(y3, y4, MASK_INT);

    y0 = _mm256_blend_epi16(y0, y4, MASK_INT<<1);
    y1 = _mm256_blend_epi16(y1, y3, MASK_INT<<1);
    y2 = _mm256_blend_epi16(y2, y5, MASK_INT<<1);

    y0 = _mm256_shuffle_epi8(y0, yC_SUFFLE_YCP_Y);
    y1 = _mm256_shuffle_epi8(y1, _mm256_alignr_epi8(yC_SUFFLE_YCP_Y, yC_SUFFLE_YCP_Y, 6));
    y2 = _mm256_shuffle_epi8(y2, _mm256_alignr_epi8(yC_SUFFLE_YCP_Y, yC_SUFFLE_YCP_Y, 12));
}

static __forceinline void gather_y_uv_from_yc48(const void *ptr_src, __m256i& y0, __m256i& y1, __m256i& y2) {
    y0 = _mm256_loadu_si256((const __m256i *)((const char *)ptr_src +  0));
    y1 = _mm256_loadu_si256((const __m256i *)((const char *)ptr_src + 32));
    y2 = _mm256_loadu_si256((const __m256i *)((const char *)ptr_src + 64));
    gather_y_uv_from_yc48(y0, y1, y2);
}

static __forceinline void gather_y_u_v_from_yc48(const void *ptr_src, __m256i& y0, __m256i& y1, __m256i& y2) {
    y0 = _mm256_loadu_si256((const __m256i *)((const char *)ptr_src +  0));
    y1 = _mm256_loadu_si256((const __m256i *)((const char *)ptr_src + 32));
    y2 = _mm256_loadu_si256((const __m256i *)((const char *)ptr_src + 64));
    gather_y_u_v_from_yc48(y0, y1, y2);
}

static __forceinline void afs_pack_yc48(__m256i& y0, __m256i& y1, __m256i& y2, const __m256i& yY, const __m256i& yC0, const __m256i& yC1) {
    __m256i yYtemp, yCtemp0, yCtemp1;

    yCtemp0 = yC0; // _mm256_permute2x128_si256(yC0, yC1, (2<<4) | 0);
    yCtemp1 = yC1; // _mm256_permute2x128_si256(yC0, yC1, (3<<4) | 1);

    yYtemp = _mm256_shuffle_epi8(yY, yC_SUFFLE_YCP_Y); //52741630

    yCtemp1 = _mm256_shuffle_epi32(yCtemp1, _MM_SHUFFLE(3, 0, 1, 2));
    yCtemp0 = _mm256_shuffle_epi32(yCtemp0, _MM_SHUFFLE(1, 2, 3, 0));
    yCtemp0 = _mm256_alignr_epi8(yCtemp0, yCtemp0, 14);

    __m256i yA0, yA1, yA2;
    yA0 = _mm256_blend_epi16(yYtemp, yCtemp0, 0x80+0x04+0x02);
    yA1 = _mm256_blend_epi16(yYtemp, yCtemp1, 0x08+0x04);
    yA2 = _mm256_blend_epi16(yYtemp, yCtemp0, 0x10+0x08);
    yA2 = _mm256_blend_epi16(yA2, yCtemp1, 0x80+0x40+0x02+0x01);
    yA1 = _mm256_blend_epi16(yA1, yCtemp0, 0x40+0x20+0x01);
    yA0 = _mm256_blend_epi16(yA0, yCtemp1, 0x20+0x10);

    y0 = _mm256_permute2x128_si256(yA0, yA1, (2<<4)|0);
    y1 = _mm256_blend_epi32(yA0, yA2, 0x0f);
    y2 = _mm256_permute2x128_si256(yA1, yA2, (3<<4)|1);
}

static __forceinline void storeu_yc48(void *ptr_dst, __m256i& yY, __m256i& yCbCrEven, __m256i& yCbCrOdd) {
    __m256i y0, y1, y2;
    afs_pack_yc48(y0, y1, y2, yY, yCbCrEven, yCbCrOdd);
    _mm256_storeu_si256((__m256i *)((char *)ptr_dst +  0), y0);
    _mm256_storeu_si256((__m256i *)((char *)ptr_dst + 32), y1);
    _mm256_storeu_si256((__m256i *)((char *)ptr_dst + 64), y2);
}

static __forceinline void store_yc48(void *ptr_dst, __m256i& yY, __m256i& yCbCrEven, __m256i& yCbCrOdd) {
    __m256i y0, y1, y2;
    afs_pack_yc48(y0, y1, y2, yY, yCbCrEven, yCbCrOdd);
    _mm256_store_si256((__m256i *)((char *)ptr_dst +  0), y0);
    _mm256_store_si256((__m256i *)((char *)ptr_dst + 32), y1);
    _mm256_store_si256((__m256i *)((char *)ptr_dst + 64), y2);
}

template<bool SRC_DIB>
static __forceinline void convert_matrix_yc48_avx2_base(COLOR_PROC_INFO *cpip, const CSP_CONVERT_MATRIX matrix) {
    const int height = cpip->h;
    const int width = cpip->w;
    const int x_fin = width - 16;
    int src_pitch = (SRC_DIB) ? sizeof(PIXEL_YC) * cpip->w : cpip->line_size;
    int dst_pitch = (SRC_DIB) ? cpip->line_size : sizeof(PIXEL_YC) * cpip->w;
    const char *ycp_src_line = (const char *)((SRC_DIB) ? cpip->pixelp : cpip->ycp);
    char *ycp_dst_line = (char *)((SRC_DIB) ? cpip->ycp : cpip->pixelp);
    __m256i xY, xCbCrEven, xCbCrOdd;
    for (int y = 0; y < height; y++, ycp_dst_line += dst_pitch, ycp_src_line += src_pitch) {
        const char *ptr_src = ycp_src_line;
        const char *ptr_src_fin = ptr_src + x_fin * sizeof(PIXEL_YC);
        char *ptr_dst = ycp_dst_line;
        const int dst_mod32 = (size_t)ptr_dst & 0x1f;
        if (dst_mod32) {
            gather_y_uv_from_yc48(ptr_src, xY, xCbCrEven, xCbCrOdd);
            convert_csp_y_cbcr(xY, xCbCrEven, xCbCrOdd, matrix);
            storeu_yc48(ptr_dst, xY, xCbCrEven, xCbCrOdd);
            int mod6 = dst_mod32 % 6;
            int dw = (32 * (((mod6) ? mod6 : 6)>>1)-dst_mod32);
            ptr_dst += dw;
            ptr_src += dw;
        }
        for (; ptr_src < ptr_src_fin; ptr_dst += 96, ptr_src += 96) {
            gather_y_uv_from_yc48(ptr_src, xY, xCbCrEven, xCbCrOdd);
            convert_csp_y_cbcr(xY, xCbCrEven, xCbCrOdd, matrix);
            store_yc48(ptr_dst, xY, xCbCrEven, xCbCrOdd);
        }
        if (ptr_src_fin < ptr_src) {
            int offset = ptr_src - ptr_src_fin;
            ptr_src -= offset;
            ptr_dst -= offset;
        }
        gather_y_uv_from_yc48(ptr_src, xY, xCbCrEven, xCbCrOdd);
        convert_csp_y_cbcr(xY, xCbCrEven, xCbCrOdd, matrix);
        storeu_yc48(ptr_dst, xY, xCbCrEven, xCbCrOdd);
    }
}
void convert_yc48_btxxx_bt601_avx2(COLOR_PROC_INFO *cpip) {
    convert_matrix_yc48_avx2_base<true>(cpip, btxxx_to_bt601);
}
void convert_yc48_bt601_btxxx_avx2(COLOR_PROC_INFO *cpip) {
    convert_matrix_yc48_avx2_base<false>(cpip, bt601_to_btxxx);
}

static __forceinline void convert_range_y_yuy2_to_yc48(__m256i& y0, __m256i& y1) {
    y0 = _mm256_permute4x64_epi64(y0, _MM_SHUFFLE(3, 1, 2, 0));
    y1 = _mm256_unpackhi_epi8(y0, _mm256_setzero_si256());
    y0 = _mm256_unpacklo_epi8(y0, _mm256_setzero_si256());
    y0 = _mm256_slli_epi16(y0, 6);
    y1 = _mm256_slli_epi16(y1, 6);
    y0 = _mm256_mulhi_epi16(y0, _mm256_set1_epi16(19152));
    y1 = _mm256_mulhi_epi16(y1, _mm256_set1_epi16(19152));
    y0 = _mm256_sub_epi16(y0, _mm256_set1_epi16(299));
    y1 = _mm256_sub_epi16(y1, _mm256_set1_epi16(299));
}

static __forceinline void convert_range_c_yuy2_to_yc48(__m256i& y0, __m256i& y1) {
    y0 = _mm256_permute4x64_epi64(y0, _MM_SHUFFLE(3, 1, 2, 0));
    y1 = _mm256_unpackhi_epi8(y0, _mm256_setzero_si256());
    y0 = _mm256_unpacklo_epi8(y0, _mm256_setzero_si256());
    y0 = _mm256_slli_epi16(y0, 6);
    y1 = _mm256_slli_epi16(y1, 6);
    y0 = _mm256_mulhi_epi16(y0, _mm256_set1_epi16(18752));
    y1 = _mm256_mulhi_epi16(y1, _mm256_set1_epi16(18752));
    y0 = _mm256_sub_epi16(y0, _mm256_set1_epi16(2340));
    y1 = _mm256_sub_epi16(y1, _mm256_set1_epi16(2340));
}

static __forceinline __m256i afs_uv_interp_linear(const __m256i& y1, const __m256i& y2) {
    //y1 ... | 1 | 0 |
    //y2 ... | 3 | 2 |
    __m256i y3, y4;
    y3 = _mm256_alignr256_epi8(y2, y1, 4); // | 2 | 1 |
    y4 = _mm256_add_epi16(y1, _mm256_srli_epi16(_mm256_cmpeq_epi16(y1, y1), 15));
    y3 = _mm256_add_epi16(y3, y4);
    return _mm256_srai_epi16(y3, 1);
}

void convert_yuy2_yc48_avx2(COLOR_PROC_INFO *cpip, const CSP_CONVERT_MATRIX matrix) {
    const int height = cpip->h;
    const int width = cpip->w;
    const int x_fin = width - 32;
    const int dst_pitch = cpip->line_size;
    const int src_pitch = ((width + 1) / 2) * 4;
    const char *ycp_src_line = (const char *)cpip->pixelp;
    char *ycp_dst_line = (char *)cpip->ycp;
    for (int y = 0; y < height; y++, ycp_dst_line += dst_pitch, ycp_src_line += src_pitch) {
        const char *ptr_src = ycp_src_line;
        char *ptr_dst = ycp_dst_line;
        const char *ptr_src_fin = ptr_src + x_fin * 2;
        __m256i m0, m1;
        __m256i my0a, my0b, my4a, mc00, mc0a, mc0b, mc4a, mc4b;
        m0 = _mm256_loadu2_m128i((const __m128i *)(ptr_src + 32), (const __m128i *)(ptr_src +  0));
        m1 = _mm256_loadu2_m128i((const __m128i *)(ptr_src + 48), (const __m128i *)(ptr_src + 16));

        my0a = _mm256_packus_epi16(_mm256_and_si256(m0, _mm256_set1_epi16(0xff)), _mm256_and_si256(m1, _mm256_set1_epi16(0xff)));
        mc0a = _mm256_packus_epi16(_mm256_srli_epi16(m0, 8), _mm256_srli_epi16(m1, 8));
        convert_range_c_yuy2_to_yc48(mc0a, mc0b);
        //mc00 = _mm256_broadcastd_epi32(_mm256_castsi256_si128(mc0a));

        const int dst_mod32 = (size_t)ptr_dst & 0x1f;
        if (dst_mod32) {
            m0 = _mm256_loadu2_m128i((const __m128i *)(ptr_src +  96), (const __m128i *)(ptr_src + 64));
            m1 = _mm256_loadu2_m128i((const __m128i *)(ptr_src + 112), (const __m128i *)(ptr_src + 80));
            my4a = _mm256_packus_epi16(_mm256_and_si256(m0, _mm256_set1_epi16(0xff)), _mm256_and_si256(m1, _mm256_set1_epi16(0xff)));
            mc4a = _mm256_packus_epi16(_mm256_srli_epi16(m0, 8), _mm256_srli_epi16(m1, 8));

            convert_range_y_yuy2_to_yc48(my0a, my0b);
            convert_range_c_yuy2_to_yc48(mc4a, mc4b);

            __m256i mc1a = afs_uv_interp_linear(mc0a, mc0b);
            __m256i mc1b = afs_uv_interp_linear(mc0b, mc4a);

            convert_csp_y_cbcr(my0a, mc0a, mc1a, matrix);
            convert_csp_y_cbcr(my0b, mc0b, mc1b, matrix);

            __m256i x0, x1, x2;
            afs_pack_yc48(x0, x1, x2, my0a, mc0a, mc1a);

            _mm256_storeu_si256((__m256i*)((uint8_t *)ptr_dst +   0), x0);
            _mm256_storeu_si256((__m256i*)((uint8_t *)ptr_dst +  32), x1);
            _mm256_storeu_si256((__m256i*)((uint8_t *)ptr_dst +  64), x2);

            afs_pack_yc48(x0, x1, x2, my0b, mc0b, mc1b);

            _mm256_storeu_si256((__m256i*)((uint8_t *)ptr_dst +  96), x0);
            _mm256_storeu_si256((__m256i*)((uint8_t *)ptr_dst + 128), x1);
            _mm256_storeu_si256((__m256i*)((uint8_t *)ptr_dst + 160), x2);

            //ずれ修正
            int mod6 = dst_mod32 % 6;
            int dw = (32 * (((mod6) ? mod6 : 6)>>1)-dst_mod32);
            ptr_dst += dw;
            ptr_src += dw / sizeof(PIXEL_YC) * 2;

            m0 = _mm256_loadu_si256((const __m256i *)ptr_src);
            m1 = _mm256_loadu_si256((const __m256i *)(ptr_src + 32));

            my0a = _mm256_packus_epi16(_mm256_and_si256(m0, _mm256_set1_epi16(0xff)), _mm256_and_si256(m1, _mm256_set1_epi16(0xff)));
            mc0a = _mm256_packus_epi16(_mm256_srli_epi16(m0, 8), _mm256_srli_epi16(m1, 8));
            convert_range_c_yuy2_to_yc48(mc0a, mc0b);
            //mc00 = _mm256_broadcastd_epi32(_mm256_castsi256_si128(mc0a));
        }

        for (; ptr_src < ptr_src_fin; ptr_dst += 192, ptr_src += 64) {
            m0 = _mm256_loadu2_m128i((const __m128i *)(ptr_src +  96), (const __m128i *)(ptr_src + 64));
            m1 = _mm256_loadu2_m128i((const __m128i *)(ptr_src + 112), (const __m128i *)(ptr_src + 80));
            my4a = _mm256_packus_epi16(_mm256_and_si256(m0, _mm256_set1_epi16(0xff)), _mm256_and_si256(m1, _mm256_set1_epi16(0xff)));
            mc4a = _mm256_packus_epi16(_mm256_srli_epi16(m0, 8), _mm256_srli_epi16(m1, 8));

            convert_range_y_yuy2_to_yc48(my0a, my0b);
            convert_range_c_yuy2_to_yc48(mc4a, mc4b);

            __m256i mc1a = afs_uv_interp_linear(mc0a, mc0b);
            __m256i mc1b = afs_uv_interp_linear(mc0b, mc4a);

            convert_csp_y_cbcr(my0a, mc0a, mc1a, matrix);
            convert_csp_y_cbcr(my0b, mc0b, mc1b, matrix);

            __m256i x0, x1, x2;
            afs_pack_yc48(x0, x1, x2, my0a, mc0a, mc1a);

            _mm256_store_si256((__m256i*)((uint8_t *)ptr_dst +   0), x0);
            _mm256_store_si256((__m256i*)((uint8_t *)ptr_dst +  32), x1);
            _mm256_store_si256((__m256i*)((uint8_t *)ptr_dst +  64), x2);

            afs_pack_yc48(x0, x1, x2, my0b, mc0b, mc1b);

            _mm256_store_si256((__m256i*)((uint8_t *)ptr_dst +  96), x0);
            _mm256_store_si256((__m256i*)((uint8_t *)ptr_dst + 128), x1);
            _mm256_store_si256((__m256i*)((uint8_t *)ptr_dst + 160), x2);

            my0a = my4a;
            mc0a = mc4a;
            mc0b = mc4b;
        }
        if (ptr_src_fin < ptr_src) {
            int offset = (ptr_src - ptr_src_fin) >> 1;
            ptr_dst -= offset * sizeof(PIXEL_YC);
            ptr_src -= offset * 2;
            m0 = _mm256_loadu2_m128i((const __m128i *)(ptr_src + 32), (const __m128i *)(ptr_src +  0));
            m1 = _mm256_loadu2_m128i((const __m128i *)(ptr_src + 48), (const __m128i *)(ptr_src + 16));

            my0a = _mm256_packus_epi16(_mm256_and_si256(m0, _mm256_set1_epi16(0xff)), _mm256_and_si256(m1, _mm256_set1_epi16(0xff)));
            mc0a = _mm256_packus_epi16(_mm256_srli_epi16(m0, 8), _mm256_srli_epi16(m1, 8));
            convert_range_c_yuy2_to_yc48(mc0a, mc0b);
            //mc00 = _mm256_broadcastd_epi32(_mm256_castsi256_si128(mc0a));
        }

        convert_range_y_yuy2_to_yc48(my0a, my0b);
        mc4a = _mm256_permute4x64_epi64(_mm256_shuffle_epi32(mc0b, _MM_SHUFFLE(3, 3, 3, 3)), _MM_SHUFFLE(3, 3, 3, 3));

        __m256i mc1a = afs_uv_interp_linear(mc0a, mc0b);
        __m256i mc1b = afs_uv_interp_linear(mc0b, mc4a);

        convert_csp_y_cbcr(my0a, mc0a, mc1a, matrix);
        convert_csp_y_cbcr(my0b, mc0b, mc1b, matrix);

        __m256i x0, x1, x2;
        afs_pack_yc48(x0, x1, x2, my0a, mc0a, mc1a);

        _mm256_storeu_si256((__m256i*)((uint8_t *)ptr_dst +   0), x0);
        _mm256_storeu_si256((__m256i*)((uint8_t *)ptr_dst +  32), x1);
        _mm256_storeu_si256((__m256i*)((uint8_t *)ptr_dst +  64), x2);

        afs_pack_yc48(x0, x1, x2, my0b, mc0b, mc1b);

        _mm256_storeu_si256((__m256i*)((uint8_t *)ptr_dst +  96), x0);
        _mm256_storeu_si256((__m256i*)((uint8_t *)ptr_dst + 128), x1);
        _mm256_storeu_si256((__m256i*)((uint8_t *)ptr_dst + 160), x2);
    }
}

void convert_yuy2_yc48_avx2(COLOR_PROC_INFO *cpip) {
    convert_yuy2_yc48_avx2(cpip, btxxx_to_bt601);
}

static __forceinline __m256i convert_y_range_from_yc48(__m256i y0, __m256i yC_Y_MA_16, int Y_RSH_16, const __m256i& yC_YCC, const __m256i& yC_pw_one) {
    __m256i y7;

    y7 = _mm256_unpackhi_epi16(y0, yC_pw_one);
    y0 = _mm256_unpacklo_epi16(y0, yC_pw_one);

    y0 = _mm256_madd_epi16(y0, yC_Y_MA_16);
    y7 = _mm256_madd_epi16(y7, yC_Y_MA_16);
    y0 = _mm256_srai_epi32(y0, Y_RSH_16);
    y7 = _mm256_srai_epi32(y7, Y_RSH_16);
    y0 = _mm256_add_epi32(y0, yC_YCC);
    y7 = _mm256_add_epi32(y7, yC_YCC);

    y0 = _mm256_packus_epi32(y0, y7);

    return y0;
}
static __forceinline __m256i convert_uv_range_after_adding_offset(__m256i y0, const __m256i& yC_UV_MA_16, int UV_RSH_16, const __m256i& yC_YCC, const __m256i& yC_pw_one) {
    __m256i y7;
    y7 = _mm256_unpackhi_epi16(y0, yC_pw_one);
    y0 = _mm256_unpacklo_epi16(y0, yC_pw_one);

    y0 = _mm256_madd_epi16(y0, yC_UV_MA_16);
    y7 = _mm256_madd_epi16(y7, yC_UV_MA_16);
    y0 = _mm256_srai_epi32(y0, UV_RSH_16);
    y7 = _mm256_srai_epi32(y7, UV_RSH_16);
    y0 = _mm256_add_epi32(y0, yC_YCC);
    y7 = _mm256_add_epi32(y7, yC_YCC);

    y0 = _mm256_packus_epi32(y0, y7);

    return y0;
}
static __forceinline __m256i convert_uv_range_from_yc48(__m256i y0, const __m256i& yC_UV_OFFSET_x1, const __m256i& yC_UV_MA_16, int UV_RSH_16, const __m256i& yC_YCC, const __m256i& yC_pw_one) {
    y0 = _mm256_add_epi16(y0, yC_UV_OFFSET_x1);

    return convert_uv_range_after_adding_offset(y0, yC_UV_MA_16, UV_RSH_16, yC_YCC, yC_pw_one);
}

template<bool aligned_store>
static __forceinline void convert_yc48_yuy2_simd(void *ptr_dst, const void *ptr_src, const CSP_CONVERT_MATRIX matrix) {
    const __m256i yC_pw_one = _mm256_set1_epi16(1);
    __m256i yY, yCbCrEven, yCbCrOdd;
    gather_y_uv_from_yc48(ptr_src, yY, yCbCrEven, yCbCrOdd);
    convert_csp_y_cbcr(yY, yCbCrEven, yCbCrOdd, matrix);
    yY = convert_y_range_from_yc48(yY, yC_Y_L_MA_8, Y_L_RSH_8, _mm256_set1_epi32(1<<LSFT_YCC_8), yC_pw_one);
    yCbCrEven = convert_uv_range_from_yc48(yCbCrEven, _mm256_set1_epi16(UV_OFFSET_x1), yC_UV_L_MA_8_444, UV_L_RSH_8_444, _mm256_set1_epi32(1<<LSFT_YCC_8), yC_pw_one);

    __m256i yYUY2 = _mm256_or_si256(_mm256_and_si256(yY, _mm256_set1_epi16(0xff)), _mm256_slli_epi16(yCbCrEven, 8));
    (aligned_store) ? _mm256_store_si256((__m256i *)ptr_dst, yYUY2) : _mm256_storeu_si256((__m256i *)ptr_dst, yYUY2);
}

static __forceinline void convert_yc48_yuy2_avx2(COLOR_PROC_INFO *cpip, const CSP_CONVERT_MATRIX matrix) {
    const int height = cpip->h;
    const int width = cpip->w;
    const int x_fin = width - 16;
    const int pitch = cpip->line_size;
    const int src_pitch = cpip->line_size;
    const int dst_pitch = ((width + 1) / 2) * 4;
    const char *ycp_src_line = (const char *)cpip->ycp;
    char *ycp_dst_line = (char *)cpip->pixelp;
    for (int y = 0; y < height; y++, ycp_dst_line += dst_pitch, ycp_src_line += src_pitch) {
        const char *ptr_src = ycp_src_line;
        const char *ptr_src_fin = ptr_src + x_fin * sizeof(PIXEL_YC);
        char *ptr_dst = ycp_dst_line;
        const int dst_mod32 = (size_t)ptr_dst & 0x1f;
        if (dst_mod32) {
            convert_yc48_yuy2_simd<false>(ptr_dst, ptr_src, matrix);

            //ずれ修正
            int dw = (32 - dst_mod32) / 2;
            ptr_src += dw * sizeof(PIXEL_YC);
            ptr_dst += dw * 2;
        }
        for (; ptr_src < ptr_src_fin; ptr_dst += 32, ptr_src += 96) {
            convert_yc48_yuy2_simd<true>(ptr_dst, ptr_src, matrix);
        }
        if (ptr_src_fin < ptr_src) {
            int offset = (ptr_src - ptr_src_fin) / sizeof(PIXEL_YC);
            ptr_src -= offset * sizeof(PIXEL_YC);
            ptr_dst -= offset * 2;
        }
        convert_yc48_yuy2_simd<false>(ptr_dst, ptr_src, matrix);
    }
}

void convert_yc48_yuy2_avx2(COLOR_PROC_INFO *cpip) {
    convert_yc48_yuy2_avx2(cpip, bt601_to_btxxx);
}
