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
 * @file distance_transform_nn.cc
 * @brief Re-implement of cv::distanceTransform(), single-threadded, with a
 *        threshold of 16 and LUT. Instead of distance value, the output is
 *        a matrix of 10-bit elements, 8 bits for the nearest neighbor and 2
 *        bits for x & y sign. If the 8-bit part is 0xff, the distance value
 *        is larger than 16, and should be considered as the outlier.
 */

#include <stdint.h>
#include <math.h>
#include <stdio.h>

// LUT for dt stage1
static uint8_t lut_fw[64];   // {sign[5:5], last_state[4:0]} -> cur_state
static int16_t lut_bw[1024]; // {in[9:5], cur_state[4:0]} -> {output, next_state}

// LUT for intersect calculation in dt stage2
static float lut_sqr_sub[1024];   // {a[9:5], b[4:0]} -> a*a - b*b
static float lut_sqrt_add[1024];  // {a[9:5], b[4:0]} -> sqrt(a*a + b*b)
static float lut_div[1024];       // {a[9:0]} -> 1.0 / a

// LUT for storage
static uint16_t lut_dxdy_xysign[1024];
static float lut_dxdy2dt[1024];

void init_dt_nn(void)
{
  const int8_t min_val = -16;
  for (int i = 0; i < 64; i++) {
    int is_pix = i & 0x20;
    int8_t cur_dy = ((i | 0xe0) + 0x10) ^ 0xf0;
    cur_dy = is_pix ? 0 : (cur_dy <= min_val) ? min_val : cur_dy - 1;
    lut_fw[i] = cur_dy & 0x1f;
  }
  for (int8_t cur_dy = min_val; cur_dy < -min_val; cur_dy++) {
    for (int8_t pout = min_val; pout <= 0; pout++) {
      int lutindex = (cur_dy & 0x1f) | ((pout & 0x1f) << 5);
      int8_t chos_dy, next_dy;
      if (cur_dy > min_val) {
        chos_dy = (cur_dy + pout < 0) ? cur_dy : pout;
        next_dy = chos_dy + 1;
      } else {
        chos_dy = pout;
        next_dy = (chos_dy > min_val) ? chos_dy + 1 : chos_dy;
      }
      int16_t val = ((int16_t)next_dy & 0x1f) | (((int16_t)chos_dy & 0x1f) << 8);
      lut_bw[lutindex] = val;
    }
  }
  for (int8_t cur_dy = min_val; cur_dy < -min_val; cur_dy++) {
    for (int8_t last_dy = min_val; last_dy < -min_val; last_dy++) {
      int lutindex = ((cur_dy & 0x1f) << 5) | (last_dy & 0x1f);
      float sqr_add = sqrt(cur_dy*cur_dy + last_dy*last_dy);
      float sqr_sub = cur_dy*cur_dy - last_dy*last_dy;
      lut_sqrt_add[lutindex] = sqr_add;
      lut_sqr_sub[lutindex] = sqr_sub;
    }
  }
  lut_div[0] = 1.0;
  for (int i = 1; i < 1024; i++) {
    lut_div[i] = 1.0 / i;
  }
  for (int i = 0; i < 1024; i++) {
    int16_t sign = (i & 0x200) | ((i << 4) & 0x100);
    int dx = i >> 5, dy = i & 0x1f;
    if (dx & 0x10) dx = 32 - dx;
    if (dy & 0x10) dy = 32 - dy;
    lut_dxdy_xysign[i] = sign | (dx << 4) | dy;
  }
  for (int i = 0; i < 1024; i++) {
    int dx = (i & 0xf0) >> 4, dy = i & 0x0f;
    // if (i & 0x200) dx = -dx;
    // if (i & 0x100) dy = -dy;
    lut_dxdy2dt[i] = sqrt(dx*dx + dy*dy);
  }
}

void distance_transform_nn(uint8_t *input, int width, int height, 
  uint8_t *xy_mantissa, uint8_t *xy_sign)
{
  uint8_t tmpdy[width];

  // starge 1: iterate by column to find delta y
  for (int x = 0; x < width; x += 4) {
    uint8_t *pin = input + x;
    uint8_t *pout = xy_mantissa + x;
    uint32_t cur_dy = 0x10101010;
    for (int y = 0; y < height; y++, pin += width, pout += width) {
      uint32_t index32 = (*(uint32_t*)pin & 0x20202020) | cur_dy;
      *(uint32_t*)pout = 
      cur_dy = lut_fw[index32 & 0xff] |
              (lut_fw[(index32 >> 8) & 0xff] << 8) |
              (lut_fw[(index32 >> 16) & 0xff] << 16) |
              (lut_fw[(index32 >> 24) & 0xff] << 24);
    }
    uint8_t *pinout = xy_mantissa + width*(height-1) + x;
    for (int y = 0; y < height; y++, pinout -= width) {
      uint32_t last32 = *(uint32_t*)pinout;
      int i1 = (cur_dy & 0x1f) | ((last32 & 0x1f) << 5);
      int i2 = ((cur_dy >> 8) & 0x1f) | (((last32 >> 8) & 0x1f) << 5);
      int i3 = ((cur_dy >> 16) & 0x1f) | (((last32 >> 16) & 0x1f) << 5);
      int i4 = ((cur_dy >> 24) & 0x1f) | (((last32 >> 24) & 0x1f) << 5);
      int16_t val1 = lut_bw[i1], val2 = lut_bw[i2], val3 = lut_bw[i3], val4 = lut_bw[i4];
      cur_dy = (val1 & 0xff) | ((val2 & 0xff) << 8) | ((val3 & 0xff) << 16) | ((val4 & 0xff) << 24);
      *(uint32_t*)pinout = (val1 >> 8) | (val2 & 0xff00) | ((val3 & 0xff00) << 8) | ((val4 & 0xff00) << 16);
    }
  }

  // stage 2: iterate by row and compute distance transform
  uint8_t *pinout = xy_mantissa;
  uint8_t *poutxy_sign = xy_sign;
  float istack[width];
  int16_t vstack[width];
  for (int y = 0; y < height; y++, 
    pinout += width, poutxy_sign += (width/4)
  ) {
    int sp = 0;
    vstack[0] = 0;
    istack[0] = -1e10;

    // calculate a list of intersects in the stack
    for (int16_t x = 0; x < width; x++) {
      float i_s;
      uint8_t cur_y = pinout[x];
      if (cur_y == 16) {
        continue;
      } else if (sp != 0) {
        int16_t last_x = vstack[sp];
        uint8_t last_y = pinout[last_x];
        int lutindex = ((cur_y) << 5) | (last_y);
        i_s = lut_sqr_sub[lutindex] * lut_div[x-last_x] + (x+last_x);
        while (i_s <= istack[sp]) {
          sp--;
          last_x = vstack[sp];
          last_y = pinout[last_x];
          lutindex = ((cur_y) << 5) | (last_y);
          i_s = lut_sqr_sub[lutindex] * lut_div[x-last_x] + (x+last_x);
        }
      } else {
        i_s = -1e10;
      }
      sp++;
      vstack[sp] = x;
      istack[sp] = i_s;
      tmpdy[sp] = cur_y;
    }

    // fill in buffer according to the intersects
    if (sp == 0) {
      for (int x = 0; x < width; x++) {
        pinout[x] = 0xff;
      }
    } else {
      int i_s_x = vstack[sp];
      int cur_dy = tmpdy[sp];
      for (int x = width-1; x >= 0; x--) {
        int x2 = x << 1;
        while (x2 < istack[sp]) {
          sp--;
          i_s_x = vstack[sp];
          cur_dy = tmpdy[sp];
        }
        int cur_dx = i_s_x - x;
        if (-16 < cur_dx && cur_dx < 16) {
          int lutindex = ((cur_dx & 0x1f) << 5) | (cur_dy);
          uint16_t dxdy_sign = lut_dxdy_xysign[lutindex];
          pinout[x] = dxdy_sign & 0xff;
          poutxy_sign[x/4] |= (dxdy_sign & 0x300) >> ((4-(x&0x3)) << 1);
        } else {
          pinout[x] = 0xff;
        }
      }
    }
  }
}

void obtain_dt_nn(uint8_t *dxdy, uint8_t *sign, float *out, int width, int height)
{
  uint8_t *pin = dxdy, *pinsign = sign;
  float *pout = out;
  for (int y = 0; y < height; y++,
    pin += width, pout += width, pinsign += (width >> 2)
  ) {
    for (int x = 0; x < width; x++) {
      int lutindex = ((pinsign[x>>2] >> ((x&3)<<1)) & 0x3) << 8;
      lutindex |= (pin[x] & 0xff);
      pout[x] = lut_dxdy2dt[pin[x]];
    }
  }
}
