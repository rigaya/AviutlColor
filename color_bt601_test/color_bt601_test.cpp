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

#define PRINT_DETAIL 0
#define NOMINMAX
#include <windows.h>
#include <vector>
#include <random>
#include <cstdio>
#include "color.h"
#include "convert_csp.h"
#include "color_simd.h"
#include "color_test.h"

int main(int argc, char **argv) {
    std::vector<std::pair<int, int>> resolution_list = {
        { 1280, 720 },
        {  704, 480 },
        {  708, 480 },
        {  716, 480 },
        {  720, 480 },
        {  724, 480 },
        {  728, 480 },
        {  732, 480 },
        {  736, 480 },
        {  740, 480 },
        {  744, 480 },
        {  748, 480 },
        {  752, 480 },
        {  756, 480 },
        {  760, 480 },
        {  764, 480 },
        {  768, 480 },
    };
    int ret = 0;
    for (auto resolution : resolution_list) {
        ret |= run_test(resolution.first, 1920, resolution.second);
    }
    fprintf(stderr, (ret) ? "Error\n!" : "Pass!\n");
    return ret;
}
