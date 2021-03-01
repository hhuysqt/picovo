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

#ifndef _PICOVO_SOLVER_LM_IC_H_
#define _PICOVO_SOLVER_LM_IC_H_

#include "solver_base.h"

namespace picovo
{

class solver_lm_ic : public solver_base
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  /**
   * @fn _calc_update_lm_ic
   * @brief Calculate the Jacobian & Hessian matrix for LM algorithm.
   * @param ref_dt DT & grad of the reference frame.
   * @param ref_dt_indexes Full-map indexes to the ref_dt
   * @param cur_pcl Point cloud of the current frame.
   * @param nr_proj Input: Number of points in cur_pcl.
   *                Output: Number of valid projections.
   * @param step Interval for sparse-to-dense calculation.
   * @param R Input: Current rotation matrix.
   * @param T Input: Current translation vector.
   * @param H Output: Hessian matrix, Hx=b
   * @param b Output: steppest descent, Hx=b
   * @return Sum of residual.
   */
  float _calc_update_lm_ic(
    const struct dt_grad_ic *ref_dt, 
    int32_t *ref_dt_indexes,
    const struct coo_3d *cur_pcl, 
    int &nr_proj,
    int step,
    const Eigen::Matrix3f &R, 
    const Eigen::Vector3f &T, 
    Matrix6x6 &H,
    Vector6 &b
  );

public:
  solver_lm_ic(picovo_config &config)
  {
    std::cout << "Using Levenberg-Marquard with inverse-compositional method." << std::endl;
    _set_parameters(config);
  }
  ~solver_lm_ic() {}

  /**
   * @fn track_frame
   * @brief Estimate the relative translation between cur_frame and
   *        ref_frame using Levenberg-Marquard with IC pre-calculation.
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
