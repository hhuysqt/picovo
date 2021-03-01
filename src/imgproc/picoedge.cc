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
 * @file picoedge.cc
 * @brief A very simple edge detection algorithm that simplifies Canny.
 *        Refer to the MCU port of picoedge.cc for further optimization.
 */

#include "picoedge.h"
#include <string.h>
#include <stdlib.h>

#include <mmintrin.h>
#include <tmmintrin.h>

static uint8_t tmpbuf[1300*400];

// LUT for feature selection
uint8_t lut_feat_sel[64]; // { curindex[5:2], lastindex[1:0] }

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

void picoedge(const uint8_t *input, uint8_t *output, int width, int height,
  uint32_t thre_i, bool is_do_vertical_scan)
{
  memset(tmpbuf, 0, width*height);

  // Only 32-bit SIMD is evaluated and will be ported to ARM Cortex M4/7

  uint32_t thre = thre_i | (thre_i << 8);
  thre |= (thre << 16);
  __m64 thre_mask = _mm_set_pi32(0, thre);

  // stage 1: iterate by row
  const uint8_t *pimg = input;
  uint8_t *pedge = (uint8_t*)tmpbuf;
  for (int y = 0; y < height; y++, pimg += width, pedge += (width/8)) {
    uint32_t *prow = (uint32_t*)pimg;
    uint32_t last4 = prow[0] << 16;
    uint32_t lastdiff4 = 0;
    uint32_t lastindex = 0;
    uint8_t *pedgerow = pedge;
    for (int x = 0; x < width; x += 4, prow++) {
      // process in a batchsize of 4 pixel
      uint32_t pix4 = *prow;
      __m64 curr_grad = _mm_abs_pi8(_mm_sub_pi8(_mm_set_pi32(0, pix4), _mm_set_pi32(0, (pix4 << 16) | (last4 >> 16))));
      uint32_t tmp_grad = _m_to_int(curr_grad);
      __m64 last_grad = _mm_set_pi32(0, (tmp_grad << 8) | (lastdiff4 >> 24));
      uint32_t diff_grad = _m_to_int(_mm_sub_pi8(last_grad, curr_grad));

      // Such bit masks can be obtained by APSR.GE bits in ARMv7-M
      uint32_t currindex = 
        ((diff_grad & (1L << 31)) >> 26) |  // bit 5
        ((diff_grad & (1L << 23)) >> 19) |  // bit 4
        ((diff_grad & (1L << 15)) >> 12) |  // bit 3
        ((diff_grad & (1L <<  7)) >> 5 ) |  // bit 2
        ((lastindex >> 4) & 0x3);           // bit 1 & 0
      uint8_t curmask = lut_feat_sel[currindex];
      uint32_t diff_thre = _m_to_int(_mm_sub_pi8(thre_mask, curr_grad));
      uint8_t maskout =
        ((diff_thre & (1L << 31)) >> 28) |  // bit 3
        ((diff_thre & (1L << 23)) >> 21) |  // bit 2
        ((diff_thre & (1L << 15)) >> 14) |  // bit 1
        ((diff_thre & (1L <<  7)) >> 7 );   // bit 0
      curmask &= maskout;
      if (x & 0x4) {
        *pedgerow |= (curmask << 4);
        pedgerow++;
      } else {
        *pedgerow = curmask;
      }
      last4 = pix4;
      lastdiff4 = tmp_grad;
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

  // Vertical scan may be skipped when doing stereo matching.
  if (is_do_vertical_scan) {
  // stage 2: iterate by column
  pimg = input;
  pedge = (uint8_t*)tmpbuf + (width/8 * 6); // line 6
  for (int x = 0; x < width; x += 4, pimg += 4) {
    uint32_t *pcol = (uint32_t*)pimg;
    uint8_t *pedgecol = pedge;
    uint32_t colbuf[2] = { pcol[0], pcol[width] };
    uint8_t histbuf[2] = { 0, 0 };
    pcol += 2*width;
    __m64 lastgrad = _mm_set1_pi32(0);
    for (int y = 2; y < height-1; y++, pcol += (width/4), pedgecol += (width/8)) {
      int bufindex = y & 0x1;
      uint32_t col_2 = pcol[0];

      __m64 grad_1 = _mm_abs_pi8(_mm_sub_pi8(_mm_set_pi32(0, colbuf[bufindex]), _mm_set_pi32(0, col_2)));
      uint32_t diffgrad1 = _m_to_int(_mm_sub_pi8(lastgrad, grad_1));
      uint32_t diffthre1 = _m_to_int(_mm_sub_pi8(thre_mask, grad_1));
      uint8_t hist2 =
        ((diffgrad1 & (1L << 31)) >> 28) |  // bit 3
        ((diffgrad1 & (1L << 23)) >> 21) |  // bit 2
        ((diffgrad1 & (1L << 15)) >> 14) |  // bit 1
        ((diffgrad1 & (1L <<  7)) >> 7 );   // bit 0
      uint8_t maskout =
        ((diffthre1 & (1L << 31)) >> 28) |  // bit 3
        ((diffthre1 & (1L << 23)) >> 21) |  // bit 2
        ((diffthre1 & (1L << 15)) >> 14) |  // bit 1
        ((diffthre1 & (1L <<  7)) >> 7 );   // bit 0
      uint8_t chos_feats = maskout & (~hist2) & histbuf[0] & histbuf[1];

      pedgecol[0] |= (chos_feats << (x & 4));

      colbuf[bufindex] = col_2;
      histbuf[bufindex] = hist2;
      lastgrad = grad_1;
    }
    pedge += ((x & 4) >> 2);
  }
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

/**
 * @fn picoedge_slow
 * @brief Slow edition of picoedge only for code comprehension.
 */
void picoedge_slow(const uint8_t *input, uint8_t *output, int width, int height,
  uint32_t thre_i)
{
  // The output width should be aligned to 32 pixels.
  int w_align = ((width + 31) & (~31)) / 8;
  memset(output, 0, w_align*height);

  // horizontal scan
  for (int y = 2; y < height-2; y++) {
    const uint8_t *pin = input + y*width;
    uint8_t *pout = output + y*w_align;
    uint32_t hist = 1 << 1;
    uint32_t lastgrad = 0;
    for (int x = 0; x < width-2; x++) {
      uint32_t pix0 = pin[x], pix2 = pin[x+2];
      uint32_t grad = abs((int32_t)pix2 - (int32_t)pix0);
      if (grad < lastgrad) {
        if (hist == 0 && grad > thre_i) {
          pout[x/8] |= (1 << (x&7));
        }
        hist = (hist >> 1) | (1 << 1);
      } else {
        hist = (hist >> 1);
      }
      lastgrad = grad;
    }
  }

  // vertical scan
  for (int x = 2; x < width-2; x++) {
    uint32_t hist = 1 << 1;
    uint32_t lastgrad = 0;
    for (int y = 0; y < height-2; y++) {
      const uint8_t *pin = input + y*width;
      uint8_t *pout = output + y*w_align;
      uint32_t pix0 = pin[x], pix2 = pin[x+2*width];
      uint32_t grad = abs((int32_t)pix2 - (int32_t)pix0);
      if (grad < lastgrad) {
        if (hist == 0 && grad > thre_i) {
          pout[x/8] |= (1 << (x&7));
        }
        hist = (hist >> 1) | (1 << 1);
      } else {
        hist = (hist >> 1);
      }
      lastgrad = grad;
    }
  }

  // filter out salt-and-pepper
  memcpy(tmpbuf, output, w_align*height);
  for (int y = 2; y < height-2; y++) {
    uint8_t *pout = output + y*w_align, *ptmp = tmpbuf + y*w_align;
    for (int x = 2; x < width-2; x++) {
      int cnt = 0;
      if (pout[(x-1)/8-w_align] & (1 << ((x-1)&7))) cnt++;
      if (pout[(x  )/8-w_align] & (1 << ((x  )&7))) cnt++;
      if (pout[(x+1)/8-w_align] & (1 << ((x+1)&7))) cnt++;
      if (pout[(x-1)/8        ] & (1 << ((x-1)&7))) cnt++;
      if (pout[(x  )/8        ] & (1 << ((x  )&7))) cnt++;
      if (pout[(x+1)/8        ] & (1 << ((x+1)&7))) cnt++;
      if (pout[(x-1)/8+w_align] & (1 << ((x-1)&7))) cnt++;
      if (pout[(x  )/8+w_align] & (1 << ((x  )&7))) cnt++;
      if (pout[(x+1)/8+w_align] & (1 << ((x+1)&7))) cnt++;
      if (cnt < 2)
        ptmp[x/8] &= ~(1 << (x&7));
    }
  }
  memcpy(output, tmpbuf, w_align*height);
}