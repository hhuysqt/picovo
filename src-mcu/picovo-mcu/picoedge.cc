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

#include "picoedge.h"
#include "picovo_config.h"
#include <string.h>
#include <stdlib.h>

static uint8_t tmpbuf[IMG_WIDTH*IMG_HEIGHT/8];

// LUT for feature selection
static uint8_t lut_feat_sel[64] __attribute__((section(".itcm_data"))); // { curindex[5:2], lastindex[1:0] }

void init_picoedge(void)
{
  for (int i = 0; i < 64; i++) {
    uint32_t curindex = i;
    uint8_t mask = 0;
    for (int j = 0; j < 4; j++, curindex >>= 1) {
      if ((curindex & 0x7) == 0x3) {
        mask |= (1 << j);
      }
    }
    lut_feat_sel[i] = mask;
  }
}

void retain_edge_cv8u(const uint8_t *input, uint8_t *output, int width, int height)
{
  const uint8_t *pbitmap = input;
  uint8_t *pout = output;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x += 8) {
      uint8_t bitmap = *pbitmap;
      for (int i = 0; i < 8; i++) {
        *pout = (bitmap & 0x1) ? 0xff : 0;
        pout++;
        bitmap >>= 1;
      }
      pbitmap++;
    }
  }
}

/**
 * @fn _abs_quad_sub8
 * @brief Quad abs(uint8_t - uint8_t)
 * @param a8 packed quad uint8_t
 * @param b8 packed quad uint8_t
 * @return packed quad abs(a8-b8)
 */
static inline uint32_t _abs_quad_sub8(uint32_t a8, uint32_t b8)
{
  uint32_t res8_1 = 0, res8_2 = 0, resabs = 0;
  asm volatile ("usub8 %0, %1, %2" : "=r"(res8_1) : "r"(a8), "r"(b8));
  asm volatile ("usub8 %0, %2, %1" : "=r"(res8_2) : "r"(a8), "r"(b8));
  asm volatile ("sel %0, %1, %2" : "=r"(resabs) : "r"(res8_2), "r"(res8_1));
  return resabs;
}

/**
 * @fn _sign_of_quad_sub8
 * @brief Sign bit of 4 (u8 - u8). Each bit: 1 : >=0, 0 : <0
 * @param a8 packed quad uint8_t
 * @param b8 packed quad uint8_t
 * @return 4 sign bits of (a8-b8)
 */
static inline uint8_t _sign_of_quad_sub8(uint32_t a8, uint32_t b8)
{
  uint32_t res8, APSR;
  asm volatile ("usub8 %0, %1, %2" : "=r"(res8) : "r"(a8), "r"(b8));
  asm volatile ("mrs %0, APSR" : "=r"(APSR));
  // return APSR.GE
  return (APSR >> 16) & 0xf;
}

void picoedge(const uint8_t *input, uint8_t *output, int width, int height,
  uint32_t thre_i)
{
  memset(tmpbuf, 0, sizeof(tmpbuf));

  uint32_t thre = thre_i | (thre_i << 8);
  thre |= (thre << 16);

  // stage 1: iterate by row
  const uint8_t *pimg = input;
  uint8_t *pedge = (uint8_t*)tmpbuf;
  for (int y = 0; y < height; y++, pimg += width, pedge += (width/8)) {
    uint32_t *prow = (uint32_t*)pimg;
    uint32_t last4 = prow[0] << 16;
    uint32_t last_grad = 0;
    uint32_t lastindex = 0;
    uint8_t *pedgerow = pedge;
    for (int x = 0; x < width; x += 4, prow++) {
      // process in a batchsize of 4 pixel
      uint32_t pix4 = *prow;
      uint32_t curr_grad = _abs_quad_sub8(pix4, (pix4 << 16) | (last4 >> 16));
      uint8_t currindex = 
        (_sign_of_quad_sub8(curr_grad, (curr_grad << 8) | (last_grad >> 24)) << 2) |
                                            // bit [5:2]
        ((lastindex >> 4) & 0x3);           // bit [1:0]
      uint8_t curmask = lut_feat_sel[currindex];
      uint8_t maskout = _sign_of_quad_sub8(curr_grad, thre);
      curmask &= maskout;
      if (x & 0x4) {
        *pedgerow |= (curmask << 4);
        pedgerow++;
      } else {
        *pedgerow = curmask;
      }
      last4 = pix4;
      last_grad = curr_grad;
      lastindex = currindex;
    }
    // reduce 1 pixel
    uint32_t *pedgerow32 = (uint32_t*)pedge;
    uint32_t lastedgerow = pedgerow32[0];
    for (int x = 32; x < width; x += 32) {
      uint32_t nextedgerow = pedgerow32[1];
      pedgerow32[0] = (lastedgerow >> 2) | (nextedgerow << 30);
      lastedgerow = nextedgerow;
      pedgerow32++;
    }
    pedgerow32[0] = (lastedgerow >> 2);
  }

  // stage 2: iterate by column
  pimg = input;
  pedge = (uint8_t*)tmpbuf + (width/8 * 6); // line 6
  for (int x = 0; x < width; x += 4, pimg += 4) {
    uint32_t *pcol = (uint32_t*)pimg;
    uint8_t *pedgecol = pedge;
    uint32_t colbuf[2] = { pcol[0], pcol[width] };
    uint8_t histbuf[2] = { 0, 0 };
    pcol += 2*width;
    uint32_t last_grad = 0;
    for (int y = 2; y < height-1; y++, pcol += (width/4), pedgecol += (width/8)) {
      int bufindex = y & 0x1;
      uint32_t col_2 = pcol[0];

      uint32_t curr_grad = _abs_quad_sub8(colbuf[bufindex], col_2);
      uint8_t raw_chos = _sign_of_quad_sub8(curr_grad, last_grad);
      uint8_t maskout = _sign_of_quad_sub8(curr_grad, thre);
      uint8_t chos_feats = maskout & (~raw_chos) & histbuf[0] & histbuf[1];

      pedgecol[0] |= (chos_feats << (x & 4));

      colbuf[bufindex] = col_2;
      histbuf[bufindex] = raw_chos;
      last_grad = curr_grad;
    }
    pedge += ((x & 4) >> 2);
  }

  // stage 3: filter out pepper and salt within 3x3 region
  uint32_t lastline[width/32];
  memcpy(lastline, tmpbuf, width/8);
  uint32_t *pedge32 = (uint32_t*)(tmpbuf + (width/8)); // line 1
  for (int y = 1; y < height-1; y++) {
    uint32_t lastodd16 = 0;
    /**
     * For each pixel, perform a 3x3 convolution followed by a threshold of 2.
     * The convolution kernel is 3x3 of all ones, which can be efficiently calculated using
     * bit-wise SIMD.
     */
    for (int x = 0; x < width/32; x++, pedge32++) {
      // process with 32-bit patch
      const uint32_t MASK0 = 0x1;
      uint32_t a1 = pedge32[-width/32], a2 = pedge32[0], a3 = pedge32[width/32];
      uint32_t nexteven16 = (pedge32[1-width/32] & MASK0) + (pedge32[1] & MASK0) + (pedge32[1+width/32] & MASK0);

      // split even and odd bits
      const uint32_t MASK1 = 0x55555555;
      uint32_t even16 = (a1 & MASK1) + (a2 & MASK1) + (a3 & MASK1);
      uint32_t odd16 = ((a1>>1) & MASK1) + ((a2>>1) & MASK1) + ((a3>>1) & MASK1);

      // split into 8 elements per 32 bits
      const uint32_t MASK2 = 0x33333333;
      uint32_t sum8_0 = (((odd16<<2) | (lastodd16>>30)) & MASK2) + (even16 & MASK2) + (odd16 & MASK2);
      uint32_t sum8_1 = (even16 & MASK2) + (odd16 & MASK2) + ((even16>>2) & MASK2);
      uint32_t sum8_2 = (odd16 & MASK2) + ((even16>>2) & MASK2) + ((odd16>>2) & MASK2);
      uint32_t sum8_3 = ((even16>>2) & MASK2) + ((odd16>>2) & MASK2) + (((even16>>4) | (nexteven16<<28)) & MASK2);

      uint32_t MASK3 = 0x11111111;
      uint32_t chos0 = ((sum8_0>>1) | (sum8_0>>2) | (sum8_0>>3)) & MASK3;
      MASK3 <<= 1;
      uint32_t chos1 = (sum8_1 | (sum8_1>>1) | (sum8_1>>2)) & MASK3;
      MASK3 <<= 1;
      uint32_t chos2 = ((sum8_2<<1) | sum8_2 | (sum8_2>>1)) & MASK3;
      MASK3 <<= 1;
      uint32_t chos3 = ((sum8_3<<2) | (sum8_3<<1) | sum8_3) & MASK3;
      uint32_t totalmask = chos0 | chos1 | chos2 | chos3;

      pedge32[-width/32] = lastline[x];
      lastline[x] = a2 & totalmask;

      lastodd16 = odd16;
    }
  }
  memcpy(tmpbuf+(width/8)*(height-2), lastline, width/8);
  memcpy(output, tmpbuf, width*height/8);
}
