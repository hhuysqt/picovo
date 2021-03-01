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

#ifndef SIMPLE_EDGE_DETECTION
#define SIMPLE_EDGE_DETECTION

#include <stdint.h>

/**
 * @fn init_picoedge
 * @brief Initialize the internal LUT
 */
void init_picoedge(void);

/**
 * @fn picoedge
 * @brief a simple edge detection
 * @param input input CV_8U buffer
 * @param output output bitmap buffer
 * @param width image width
 * @param height image height
 * @param thre_i Threshold of gradient of a feature point
 */
void picoedge(const uint8_t *input, uint8_t *output, int width, int height,
  uint32_t thre_i = 10);

/**
 * @fn retain_edge_cv8u
 * @brief retain CV_8U image from the bitmap
 * @param input input CV_8U buffer
 * @param output output bitmap buffer
 * @param width image width
 * @param height image height
 */
void retain_edge_cv8u(const uint8_t *input, uint8_t *output, int width, int height);

#endif