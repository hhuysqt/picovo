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

#ifndef _PICOVO_SOLVER_GD_H_
#define _PICOVO_SOLVER_GD_H_

#include "solver_base.h"

namespace picovo
{

/**
 * This class implements two Jacobian variations: XYZ and UVZ. 
 * The latter one is derived from the "dual Jacobian" of:
 * "Image gradient-based joint direct visual odometry for stereo camera"
 */
class solver_gd : public solver_base
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:

  /**
   * @fn _calc_update_gd
   * @brief Calculate the Jacobian matrix for gradient descent.
   * @param ref_dt DT & grad of the reference frame.
   * @param cur_pcl Point cloud of the current frame.
   * @param nr_proj Input: Number of points in cur_pcl.
   *                Output: Number of valid projections.
   * @param outlier_thrshd Threadshold of a projected outlier.
   * @param R Input: Current rotation matrix.
   * @param T Input: Current translation vector.
   * @param jacobian Output: Jacobian matrix
   * @return Sum of residual.
   */
  float _calc_update_gd(
    const struct dt_grad_f *ref_dt, 
    const struct coo_3d *cur_pcl, 
    int &nr_proj,
    float outlier_thrshd,
    const Eigen::Matrix3f &R, 
    const Eigen::Vector3f &T, 
    Vector6 &jacobian
  );

  /**
   * @fn _calc_update_gd_uv
   * @brief Calculate the Jacobian matrix for GD algorithm, using the
   *        (u,v) pixel coordinate instead of the transformed (x,y,z).
   *        The reference frame must be initialized by set_keyframe_uv().
   * @param ref_dt The pre-calculated gradients by set_keyframe_uv().
   * @param ref_dt_indexes Full-map indexes to the ref_dt
   * @param cur_pcl Point cloud of the current frame.
   * @param nr_proj Input: Number of points in cur_pcl.
   *                Output: Number of valid projections.
   * @param outlier_thrshd Threadshold of a projected outlier.
   * @param R Input: Current rotation matrix.
   * @param T Input: Current translation vector.
   * @param b Output: steppest descent.
   * @return Sum of residual.
   */
  float _calc_update_gd_uv(
    const struct dt_grad_uv *ref_dt, 
    int32_t *ref_dt_indexes,
    const struct coo_3d *cur_pcl, 
    int &nr_proj,
    float outlier_thrshd,
    const Eigen::Matrix3f &R, 
    const Eigen::Vector3f &T, 
    Vector6 &b
  );

  /* use pre-calculated gradient in keyframe (cache-unfriendly) */
  bool is_pre_calc_kf;

public:

  /**
   * @fn constructor
   * @param config system config structure
   */
  solver_gd(picovo_config &config)
  {
    if (config.solver_type == SOLVER_GD_PRE_JACOBIAN) {
      is_pre_calc_kf = true;
      std::cout << "Using gradient descent with Jacobian pre-calculation" << std::endl;
    } else if (config.solver_type == SOLVER_GD_PRE_GRAD) {
      is_pre_calc_kf = false;
      std::cout << "Using gradient descent with DT gradient pre-calculation" << std::endl;
    } else {
      std::cerr << "Invalid solver type: " << config.solver_type << std::endl;
      exit(-1);
    }

    _set_parameters(config);
  }

  ~solver_gd() {}

  /**
   * @fn track_frame_gd
   * @brief Estimate the relative translation between cur_frame and
   *        ref_frame using gradient descent.
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