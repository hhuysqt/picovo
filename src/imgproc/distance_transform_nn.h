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

#ifndef _MY_DISTANCE_TRANSFORM_NN_H_
#define _MY_DISTANCE_TRANSFORM_NN_H_

#include <stdint.h>

/**
 * @fn init_dt
 * @brief Initialization of Look-Up-Tables for DT.
 */
void init_dt_nn(void);

/**
 * @fn distance_transform_less_than_16
 * @brief Perform the DT with LUT
 * @param input Input edge image: 8bit, 0 or 0xff
 * @param width image width
 * @param height image height
 * @param xy_mantissa 4 bits for each positive delta x and delta y. 
 *        Total size: width x height Bytes.
 * @param xy_sign packed 2 bits for the sign of x and y.
 *        Total size: width x height / 4 Bytes.
 */
void distance_transform_nn(uint8_t *input, int width, int height, 
  uint8_t *xy_mantissa, uint8_t *xy_sign);

/**
 * @fn obtain_dt
 * @brief Recover the DT image from xy and sign
 * @param dxdy delta x and delta y
 * @param sign packed 2-bit sign
 * @param out Output image
 * @param width image width
 * @param height image height
 */
void obtain_dt_nn(uint8_t *dxdy, uint8_t *sign, float *out, int width, int height);

#endif