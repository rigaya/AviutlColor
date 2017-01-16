//----------------------------------------------------------------------------------
//	色変換プラグイン ヘッダーファイル for AviUtl version 0.99h 以降
//	By ＫＥＮくん
//----------------------------------------------------------------------------------

#ifndef __COLOR_H__
#define __COLOR_H__

//	YC構造体
#ifndef PIXEL_YC
typedef	struct {
	short	y;					//	画素(輝度    )データ (     0 ～ 4096 )
	short	cb;					//	画素(色差(青))データ ( -2048 ～ 2048 )
	short	cr;					//	画素(色差(赤))データ ( -2048 ～ 2048 )
								//	画素データは範囲外に出ていることがあります
								//	また範囲内に収めなくてもかまいません
} PIXEL_YC;
#endif

//	マルチスレッド関数用の定義
typedef void (*MULTI_THREAD_FUNC)( int thread_id,int thread_num,void *param1,void *param2 );
								//	thread_id	: スレッド番号 ( 0 ～ thread_num-1 )
								//	thread_num	: スレッド数 ( 1 ～ )
								//	param1		: 汎用パラメータ
								//	param2		: 汎用パラメータ

//	色変換の処理情報構造体
typedef struct {
	int			flag;			//	フラグ
								//	COLOR_PROC_INFO_FLAG_INVERT_HEIGHT	: pixelpの縦方向を上下逆に処理する
								//	COLOR_PROC_INFO_FLAG_USE_SSE		: SSE使用
								//	COLOR_PROC_INFO_FLAG_USE_SSE2		: SSE2使用
	PIXEL_YC	*ycp;			//	PIXEL_YC構造体へのポインタ
	void 		*pixelp;		//	DIB形式データへのポインタ
	DWORD 		format; 		//	DIB形式データのフォーマット( NULL = RGB24bit / 'Y''U''Y''2' = YUY2 / 'Y''C''4''8' = PIXEL_YC )
	int			w,h;			//	画像データの縦横サイズ
	int			line_size;		//	PIXEL_YC構造体の横幅のバイトサイズ
	int			yc_size;		//	PIXEL_YC構造体の画素のバイトサイズ
	BOOL		(*exec_multi_thread_func)( MULTI_THREAD_FUNC func,void *param1,void *param2 );
								//	指定した関数をシステムの設定値に応じたスレッド数で呼び出します
								//	呼び出された関数内からWin32APIや外部関数を使用しないでください
								//	func	: マルチスレッドで呼び出す関数
								//	param1 	: 呼び出す関数に渡す汎用パラメータ
								//	param2 	: 呼び出す関数に渡す汎用パラメータ
								//  戻り値	: TRUEなら成功
	int			reserve[16];
} COLOR_PROC_INFO;

#define COLOR_PROC_INFO_FLAG_INVERT_HEIGHT	1
#define COLOR_PROC_INFO_FLAG_USE_SSE		256
#define COLOR_PROC_INFO_FLAG_USE_SSE2		512

//	色変換プラグイン構造体
typedef struct {
	int		flag;				//	フラグ
	LPSTR	name;				//	プラグインの名前
	LPSTR	information;		//	プラグインの情報
	BOOL 	(*func_init)( void );
								//	DLL開始時に呼ばれる関数へのポインタ (NULLなら呼ばれません)
	BOOL 	(*func_exit)( void );
								//	DLL終了時に呼ばれる関数へのポインタ (NULLなら呼ばれません)
	BOOL 	(*func_pixel2yc)( COLOR_PROC_INFO *cpip );
								//	DIB形式の画像からからPIXEL_YC形式の画像に変換します (NULLなら呼ばれません)
								//  戻り値		: TRUEなら成功
								//				: FALSEならAviUtl側でデフォルト変換されます
	BOOL 	(*func_yc2pixel)( COLOR_PROC_INFO *cpip );
								//	PIXEL_YC形式の画像からからDIB形式の画像に変換します (NULLなら呼ばれません)
								//  戻り値		: TRUEなら成功
								//				: FALSEならAviUtl側でデフォルト変換されます
	int		reserve[16];
} COLOR_PLUGIN_TABLE;

BOOL func_init( void );
BOOL func_exit( void );
BOOL func_pixel2yc( COLOR_PROC_INFO *cpip );
BOOL func_yc2pixel( COLOR_PROC_INFO *cpip );

#endif //__COLOR_H__
