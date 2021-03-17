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

#ifndef SOLVER_LM_FIXPOINT_H
#define SOLVER_LM_FIXPOINT_H

#include "picovo_config.h"

// solver core works in C++...
#ifdef __cplusplus

#include <Eigen/Eigen>

typedef Eigen::Matrix<float, 6, 1> Vector6;
typedef Eigen::Matrix<float, 6, 6> Matrix6x6;

/**
 * @struct feature_s
 * @brief The compressed form of feature point, 64-bits in total
 */
struct feature_s
{
  int16_t u_cx_fx; // (u-cx)/fx << 12, Q4.12
  int16_t v_cy_fy; // (v-cy)/fy << 12, Q4.12
  uint16_t zinv;   // 1/z << 12, Q4.12
};

/**
 * @struct coo_3d
 * @brief 3D float coordinate
 */
struct coo_3d
{
  float x, y, z;
};

/**
 * @fn init_solver
 * @brief Initialize solver internal LUTs.
 */
void init_solver(void);

/**
 * @fn solver_track_frame
 * @brief Estimate the relative translation between cur_frame and
 *        ref_frame using Levenberg-Marquard algorithm.
 * @param ref_dt reference distance transform image
 * @param cur_feat buffer of feature points of the current frame
 * @param nr_feat input: NR of features; output: NR of tracked features
 * @param R Input&output, rotation matrix
 * @param T Input&output, translation matrix
 * @return The last residual
 */
float solver_track_frame(
  uint8_t *ref_dt,
  struct feature_s *cur_feat,
  int &nr_feat,
  Eigen::Matrix3f &R, 
  Eigen::Vector3f &T
);

#endif  // __cplusplus
#endif  // SOLVER_LM_FIXPOINT_H