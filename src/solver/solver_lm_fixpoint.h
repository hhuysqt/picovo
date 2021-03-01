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

#ifndef _PICOVO_SOLVER_FIXPOINT_H_
#define _PICOVO_SOLVER_FIXPOINT_H_

#include "solver_base.h"

namespace picovo
{

class solver_lm_fixpoint : public solver_base
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:

  /**
   * @fn _calc_update_lm_fp
   * @brief Calculate Hessian & b using fixed point arithmetic.
   * @param ref_dt DT of the reference frame.
   * @param cur_feat Selected features of the current frame.
   * @param nr_proj Input: Number of points in cur_pcl.
   *                Output: Number of valid projections.
   * @param step Interval for sparse-to-dense calculation.
   * @param R Input: Current rotation matrix.
   * @param T Input: Current translation vector.
   * @param H Output: Hessian matrix, Hx=b
   * @param b Output: steppest descent, Hx=b
   * @return Sum of residual.
   */
  float _calc_update_lm_fp(
    const uint8_t *ref_dt, 
    const struct compressed_feature *cur_feat, 
    int &nr_proj,
    int step,
    const Eigen::Matrix3f &R, 
    const Eigen::Vector3f &T, 
    Matrix6x6 &H,
    Vector6 &b
  );

  /* Look-Up-Table for residual: huber*sqrt(uint8_t) */
  int32_t lut_residual[256];
  /* LUT for 1/(huber*sqrt(uint8_t)) */
  int32_t lut_invres[256];
  /* LUT for gradients: 0.5*f*huber*sqrt(uint8_t) */
  int16_t lut_iu[256];
  int16_t lut_iv[256];

  /* The LUT cannot be calculated before knowing camera intrinsics */
  bool is_lut_prepared;

public:

  /**
   * @fn constructor
   * @param config system config structure
   */
  solver_lm_fixpoint(picovo_config &config)
  {
    std::cout << "Using fixed-point Levenberg-Marquard" << std::endl;

    _set_parameters(config);

    // prepare fixed-point LUT
    for (int i = 0; i < 256; i++) {
      double d = huber_weight * sqrt(i);
      lut_residual[i] = d * (1L << 16);       // 4+16
      lut_invres[i] = (1L << 9) / d;          // 7+9
      lut_iu[i] = 0.3 * fx * d * (1L << 7);   // 12+7
      lut_iv[i] = 0.3 * fy * d * (1L << 7);   // 12+7
    }
  }

  ~solver_lm_fixpoint() {}

  /**
   * @fn track_frame
   * @brief Estimate the relative translation between cur_frame and
   *        ref_frame using Levenberg-Marquard algorithm.
   * @param ref_frame Reference frame
   * @param cur_frame Current frame
   * @param R Input&output, rotation matrix
   * @param T Input&output, translation matrix
   * @return The last residual
   */
  float track_frame(
    std::shared_ptr<frame> &ref_frame,
    std::shared_ptr<frame> &cur_frame,
    Eigen::Matrix3f &R, 
    Eigen::Vector3f &T
  );

};

}

#endif