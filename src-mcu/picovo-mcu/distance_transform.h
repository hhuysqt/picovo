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

#ifndef _MY_DISTANCE_TRANSFORM_H_
#define _MY_DISTANCE_TRANSFORM_H_

#include <stdint.h>

/**
 * @fn init_dt
 * @brief Initialization of Look-Up-Tables for DT.
 */
void init_dt(void);

/**
 * @fn distance_transform
 * @brief Perform the DT with LUT
 * @param input Input edge bitmap: for each bit, 1 for an edge, 0 for none
 * @param output Pre-allocated output buffer: uint8_t square of distance
 * @param width image width
 * @param height image height
 */
void distance_transform(uint8_t *input, uint8_t *output, int width, int height);

#endif