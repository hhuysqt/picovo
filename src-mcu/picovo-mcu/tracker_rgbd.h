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

#ifndef _TRACKER_RGBD_H_
#define _TRACKER_RGBD_H_

#include <stdint.h>
#include "picovo_config.h"
#include "solver_lm_fixpoint.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @fn init_tracker
 * @brief C++ wrapper for tracker initialization
 */
void init_tracker(void);

/**
 * @fn reset_tracker
 * @brief C++ wrapper for tracker reset
 */
void reset_tracker(void);

/**
 * @fn track_frame_rgbd
 * @brief C++ wrapper for the tracker input
 * @param gray CV_8U grayscale image
 * @param depth CV_16U depth image
 */
void track_frame_rgbd(uint8_t *gray, uint16_t *depth);

/**
 * @fn get_current_pose_tum_str
 * @brief C++ wrapper to format the pose into TUM compatible message
 * @param outbuf output buffer
 * @param bufsize buffer size
 */
void get_current_pose_tum_str(char *outbuf, int bufsize);

/**
 * @fn get_stat_str
 * @brief get formated OVO system statistic message
 * @param outbuf output buffer
 * @param bufsize buffer size
 */
void get_stat_str(char *outbuf, int bufsize);

#ifdef __cplusplus
};
#endif

#endif
