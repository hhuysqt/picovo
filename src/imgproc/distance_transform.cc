/**
 * This file is part of PicoVO.
 *
 * Copyright (C) 2020-2021 Yuquan He <heyuquan20b at ict dot ac dot cn> 
 * (Institute of Computing Technology, Chinese Academy of Sciences)
 *
 * PicoVO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PicoVO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PicoVO. If not, see <http://www.gnu.org/licenses/>.
 */
/**
 * @file distance_transform.cc
 * @brief Re-implement of cv::distanceTransform(), single-threadded, with a
 *        threshold of 15. The output is the square of distance of uint8_t.
 */

#include <stdint.h>
#include <math.h>
#include <stdio.h>

// LUT for bitmap to CV_8U
static uint32_t lut_bitmap[16];

// LUT for dt stage 1
static uint32_t lut_dt_s1[256];

// LUT for fixed-point divide left shifted by DIV_SHIFT
#define DIV_SHIFT 12
static int32_t lut_div[1024];

void init_dt(void)
{
  for (int i = 0; i < 16; i++) {
    uint32_t pix8b = 0;
    int pix1b = i;
    for (int j = 0; j < 4; j++) {
      if (0 == (pix1b & (1<<j))) {
        pix8b |= (0xff << (j*8));
      }
    }
    lut_bitmap[i] = pix8b;
  }

  for (int i = 0; i < 256; i++) {
    // { in[7:4], cur[3:0] }
    uint cur = i & 0xf, in = i >> 4;
    uint next = cur+1;
    uint choose = next < in ? next : in;
    // { 8'h00, choose_sqr[23:16], 8'h00, choose[7:0] }
    lut_dt_s1[i] = ((choose*choose) << 16) | choose;
  }

  lut_div[0] = 0;
  for (int i = 1; i < 1024; i++) {
    lut_div[i] = (1l << DIV_SHIFT) / i;
  }
}

void distance_transform(uint8_t *input, uint8_t *output, int width, int height)
{
  // The input is aligned to 32 pixels, while the processing width is 
  // truncated to 8 pixel boundary.
  int w_truncate = width & ~7, w_align = ((width + 31) & (~31))/8;

  // stage 1: iterate by column to find delta y
  // Use a batch size of 8 to cope with binary map input
  for (int x = 0; x < w_truncate; x += 8) {
    uint8_t *pin = input + (x>>3);
    uint8_t *pout = output + x;
    uint64_t cur_dy = 0x0f0f0f0f0f0f0f0f;
    for (int y = 0; y < height; y++, pin += w_align, pout += width) {
      // for each batch in cur_dy: batch++
      cur_dy += 0x0101010101010101;
      // for each batch in cur_dy: if batch >= 16 then batch--
      cur_dy -= ((cur_dy & 0x1010101010101010) >> 4);
      // 0x00 if has keypoint, 0xff if not
      uint8_t pix1b = *pin;
      uint64_t pix8b = lut_bitmap[pix1b & 0xf] | ((uint64_t)lut_bitmap[pix1b >> 4] << 32);
      cur_dy &= pix8b;
      *(uint64_t*)pout = cur_dy;
    }
    pout -= width;
    for (int y = 0; y < height; y++, pout -= width) {
      uint64_t indexes = cur_dy | (*(uint64_t*)pout << 4);
      int i1 = (indexes >> 0 ) & 0xff, i2 = (indexes >> 8 ) & 0xff,
          i3 = (indexes >> 16) & 0xff, i4 = (indexes >> 24) & 0xff,
          i5 = (indexes >> 32) & 0xff, i6 = (indexes >> 40) & 0xff,
          i7 = (indexes >> 48) & 0xff, i8 = (indexes >> 56) & 0xff;
      uint32_t v12 = lut_dt_s1[i1] | (lut_dt_s1[i2] << 8);
      uint32_t v34 = lut_dt_s1[i3] | (lut_dt_s1[i4] << 8);
      uint32_t v56 = lut_dt_s1[i5] | (lut_dt_s1[i6] << 8);
      uint32_t v78 = lut_dt_s1[i7] | (lut_dt_s1[i8] << 8);
      cur_dy = (v12 & 0xffff) | ((v34 & 0xffff) << 16) | 
        ((uint64_t)((v56 & 0xffff) | ((v78 & 0xffff) << 16)) << 32);
      *(uint64_t*)pout = (v12 >> 16) | (v34 & 0xffff0000) |
        ((uint64_t)((v56 >> 16) | (v78 & 0xffff0000)) << 32);
    }
  }

  // stage 2: iterate by row and compute distance transform
  uint8_t *pout = output;
  int32_t istack[width];
  uint16_t vstack[width];
  uint8_t tmp_w[width];
  for (int y = 0; y < height; y++, pout += width) {
    int sp = 0;
    vstack[0] = 0;
    istack[0] = 0x80000000;

    // stage 2.1: stack the intersects
    for (int16_t x = 0; x < w_truncate; x++) {
      int32_t i_s;
      uint8_t cur_y = pout[x];
      if (cur_y == 0xe1) {
        // 15*15 = 225
        continue;
      } else if (sp != 0) {
        int16_t last_x = vstack[sp];
        uint8_t last_y = tmp_w[sp];
        i_s = (cur_y - last_y) * lut_div[x-last_x] + ((x+last_x) << DIV_SHIFT);
        while (i_s <= istack[sp]) {
          // pop
          sp--;
          last_x = vstack[sp];
          last_y = tmp_w[sp];
          i_s = (cur_y - last_y) * lut_div[x-last_x] + ((x+last_x) << DIV_SHIFT);
        }
      } else {
        i_s = 0x80000000;
      }
      // push
      sp++;
      vstack[sp] = x;
      istack[sp] = i_s;
      tmp_w[sp] = cur_y;
    }

    // stage 2.2: fill in buffer according to the intersects
    if (sp == 0) {
      for (int x = 0; x < w_truncate; x++) {
        pout[x] = 0xe1;
      }
    } else {
      int i_s_x = vstack[sp];
      int cur_dy = tmp_w[sp];
      for (int x = w_truncate-1; x >= 0; x--) {
        int x2 = x << (DIV_SHIFT + 1);
        while (x2 < istack[sp]) {
          sp--;
          i_s_x = vstack[sp];
          cur_dy = tmp_w[sp];
        }
        int cur_dx = i_s_x - x;
        int sqrxy = cur_dx*cur_dx + cur_dy;
        // min(0xe1, sqrxy) without branch
        int diff = 0xe1 - sqrxy;
        pout[x] = sqrxy + (diff & (diff >> 31));
        // pout[x] = sqrxy > 0xe1 ? 0xe1 : sqrxy;
      }
    }
  }
}
