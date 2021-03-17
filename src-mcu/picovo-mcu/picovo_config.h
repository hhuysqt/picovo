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

#ifndef PICOVO_CONFIG_H
#define PICOVO_CONFIG_H

// image shape
#define IMG_WIDTH   320
#define IMG_HEIGHT  240
#define IMG_SIZE    (IMG_WIDTH*IMG_HEIGHT)

// edge: gradient threshold
#define EDGE_GRAD_THRESHOLD 6
// feature: depth threshold
#define DEPTH_MIN 625
#define DEPTH_MAX 55000

// camera intrinsics
#define CAM_FX 258.44390575
#define CAM_FY 258.44390575
#define CAM_CX 159.32152
#define CAM_CY 127.6569945

// solver parameters
#define SOLVER_HUBER 0.3

// limited number of feature points
#define MAX_NR_FEATURE 6000

// Maximum number of dataset to be processed. Since we set a limit of
// only 2 simultaneous open files in FATFS, we have to buffer the names
// of the dataset folder.
#define MAX_NR_DATASETS 32

// use compressed feature structure
#define USE_COMPRESSED_FEATURE

// use sparse-to-dense tracking
#define SPARSE_TO_DENSE_TRACKING

#endif
