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

#define NOMINMAX
#include <windows.h>
#include "color.h"
#include "convert_csp.h"
#include "color_simd.h"

BOOL func_init(void);

convert_color_func g_func_list;

//---------------------------------------------------------------------
//		色変換プラグイン構造体定義
//---------------------------------------------------------------------
COLOR_PLUGIN_TABLE color_plugin_table = {
    NULL,												//	フラグ
    "BT.709 (1T)",									//	プラグインの名前
    "BT.709 (1T) version 0.00 by rigaya",		//	プラグインの情報
    func_init,				        					//	DLL開始時に呼ばれる関数へのポインタ (NULLなら呼ばれません)
    NULL,												//	DLL終了時に呼ばれる関数へのポインタ (NULLなら呼ばれません)
    func_pixel2yc,										//	DIB形式の画像からからPIXEL_YC形式の画像に変換します (NULLなら呼ばれません)
    func_yc2pixel,										//	PIXEL_YC形式の画像からからDIB形式の画像に変換します (NULLなら呼ばれません)
};


//---------------------------------------------------------------------
//		色変換プラグイン構造体のポインタを渡す関数
//---------------------------------------------------------------------
EXTERN_C COLOR_PLUGIN_TABLE __declspec(dllexport) * __stdcall GetColorPluginTable( void )
{
    return &color_plugin_table;
}
//	※GetColorPluginTableYUY2でYUY2フィルタモードにも対応できます


//---------------------------------------------------------------------
//      DLL開始時に呼ばれる関数
//---------------------------------------------------------------------
BOOL func_init(void) {
    get_func(&g_func_list);
    return TRUE;
}

//---------------------------------------------------------------------
//		入力変換
//---------------------------------------------------------------------
BOOL func_pixel2yc( COLOR_PROC_INFO *cpip ) {
    if (cpip->format == MAKEFOURCC('Y', 'U', 'Y', '2')) {
        g_func_list.yuy2_yc48(cpip);
        return TRUE;
    } else if (cpip->format == MAKEFOURCC('Y', 'C', '4', '8')) {
        g_func_list.yc48_btxxx_bt601(cpip);
        return TRUE;
    }
    //	RGBはAviUtl本体の変換に任せる
    return FALSE;
}


//---------------------------------------------------------------------
//		出力変換
//---------------------------------------------------------------------
BOOL func_yc2pixel( COLOR_PROC_INFO *cpip ) {
    if (cpip->format == MAKEFOURCC('Y', 'U', 'Y', '2')) {
        g_func_list.yc48_yuy2(cpip);
        return TRUE;
    } else if (cpip->format == MAKEFOURCC('Y', 'C', '4', '8')) {
        g_func_list.yc48_bt601_btxxx(cpip);
        return TRUE;
    }
    //	RGBはAviUtl本体の変換に任せる
    return FALSE;
}


