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

#ifndef _PICOVO_SOLVER_NEAREST_NEIGHBOR_H_
#define _PICOVO_SOLVER_NEAREST_NEIGHBOR_H_

#include "solver_base.h"

namespace picovo
{

class solver_lm_nn : public solver_base
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:

  /**
   * @fn _calc_update_lm
   * @brief Calculate the Jacobian matrix for LM algorithm.
   * @param ref_dt LUT of DT & grad of the reference frame.
   * @param cur_pcl Point cloud of the current frame.
   * @param nr_proj Input: Number of points in cur_pcl.
   *                Output: Number of valid projections.
   * @param step Interval for sparse-to-dense calculation.
   * @param outlier_thrshd Threadshold of a projected outlier.
   * @param R Input: Current rotation matrix.
   * @param T Input: Current translation vector.
   * @param H Output: Hessian matrix, Hx=b
   * @param b Output: steppest descent, Hx=b
   * @return Sum of residual.
   */
  float _calc_update_lm(
    const uint8_t *ref_dt, 
    const uint8_t *ref_dt_sign, 
    const struct coo_3d *cur_pcl, 
    int &nr_proj,
    int step,
    float outlier_thrshd,
    const Eigen::Matrix3f &R, 
    const Eigen::Vector3f &T, 
    Matrix6x6 &H,
    Vector6 &b
  );

  /* Look-Up-Table for gradient and residual */
  struct dt_grad_f _grad_lut[256];

  /* The LUT cannot be calculated before knowing camera intrinsics */
  bool is_lut_prepared;

public:

  /**
   * @fn constructor
   * @param config system config structure
   */
  solver_lm_nn(picovo_config &config)
  {
    std::cout << "Using Levenberg-Marquard with nearest neighbor field" << std::endl;

    _set_parameters(config);

    // prepare LUT of NN to grad
    for (int x = 0; x < 16; x++) {
      for (int y = 0; y < 16; y++) {
        float residual = sqrt(x*x + y*y);
        float huber_res = residual <= huber_weight ? residual : huber_weight;
        int lutindex = (x << 4) | y;

        // Pre-multiply fx, fy and huber.
        _grad_lut[lutindex].Iu = fx * x;
        _grad_lut[lutindex].Iv = fy * y;
        _grad_lut[lutindex].residual = residual;
      }
    }
  }

  ~solver_lm_nn() {}

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