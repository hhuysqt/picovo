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

#ifndef _PICOVO_SOLVER_BASE_H_
#define _PICOVO_SOLVER_BASE_H_

#include <vector>
#include <Eigen/Eigen>
#include <opencv2/core.hpp>
#include <memory>
#include <chrono>
#include "sophus/se3.hpp"

#include "picovo_frame.h"
#include "picovo_config.h"

namespace picovo
{

/**
 * @class solver_base
 * @brief base class of various solvers
 */
class solver_base
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

protected:

  void _set_parameters(picovo_config &config)
  {
    fx = config.fx;
    fy = config.fy;
    cx = config.cx;
    cy = config.cy;
    width = config.width;
    height = config.height;
    huber_weight = config.solver_huber_weight;
    lamda = lamda_cfg = config.solver_lamda;
    _outlier_threshold = config.solver_outlier_threshold;
  }

  void _proj_to_pixel(
    const struct coo_3d &cur_pcl, 
    const Eigen::Matrix3f &R, 
    const Eigen::Vector3f &T,
    int &u,
    int &v
  ) {
    Eigen::Vector3f pcur;
    pcur << cur_pcl.x, cur_pcl.y, cur_pcl.z;
    // rigid body tranformation
    Eigen::Vector3f pref = R*pcur + T;
    // camera projection
    const float z_inv = 1.0 / pref[2];
    const float x_z = pref[0] * z_inv, y_z = pref[1] * z_inv;
    const float u_f = x_z * fx + cx;
    const float v_f = y_z * fy + cy;
    u = u_f + 0.5;
    v = v_f + 0.5;
  };

  int _cnt_valid_proj(
    const uint8_t *ref_dt, 
    const struct coo_3d *cur_pcl, 
    int &nr_proj,
    const Eigen::Matrix3f &R, 
    const Eigen::Vector3f &T
  )
  {
    cv::Mat proj_cnt = cv::Mat(cv::Size(width, height), CV_8U);
    memset(proj_cnt.ptr<uint8_t>(), 0, width*height);

    for (int i = 0; i < nr_proj; i++) {
      int u_i, v_i;
      _proj_to_pixel(cur_pcl[i], R, T, u_i, v_i);
      if (u_i <= 1 || v_i <= 1 || u_i >= width-2 || v_i >= height-2) {
        continue;
      }
      const uint pixindex = u_i + v_i*width;
      const uint lutindex = *(ref_dt + pixindex);

      if ((lutindex & 0xf) + (lutindex >> 4) <= 2)
        proj_cnt.ptr<uint8_t>(v_i)[u_i] = 0xff;
    }

    return cv::countNonZero(proj_cnt);
  }

  int _cnt_valid_proj(
    const float *ref_dt, 
    const struct coo_3d *cur_pcl, 
    int &nr_proj,
    const Eigen::Matrix3f &R, 
    const Eigen::Vector3f &T
  )
  {
    cv::Mat proj_cnt = cv::Mat(cv::Size(width, height), CV_8U);
    memset(proj_cnt.ptr<uint8_t>(), 0, width*height);

    for (int i = 0; i < nr_proj; i++) {
      int u_i, v_i;
      _proj_to_pixel(cur_pcl[i], R, T, u_i, v_i);
      if (u_i <= 1 || v_i <= 1 || u_i >= width-2 || v_i >= height-2) {
        continue;
      }
      const uint pixindex = u_i + v_i*width;
      const float residual = *(ref_dt + pixindex);

      if (residual <= 2)
        proj_cnt.ptr<uint8_t>(v_i)[u_i] = 0xff;
    }

    return cv::countNonZero(proj_cnt);
  }

  /* Parameters */
  int width, height;            // image shape
  float fx, fy, cx, cy;         // Camera intrinsics
  float huber_weight;           // Huber weight for optimization
  float _outlier_threshold;     // Outlier threshold of a projecton pixel
  float lamda_cfg, lamda;       // default value of LM lamda

  // statistics
  float stat_residual;
  int stat_iterations;

  // Debug image during tracking
  cv::Mat curr_canny;

public:

  /**
   * @fn track_frame
   * @brief Estimate the relative translation between cur_frame and
   *        ref_frame.
   * @param ref_frame Reference frame
   * @param cur_frame Current frame
   * @param R Input&output, rotation matrix
   * @param T Input&output, translation matrix
   * @return The last residual
   */
  virtual float track_frame(
    std::shared_ptr<frame> &ref_frame,
    std::shared_ptr<frame> &cur_frame,
    Eigen::Matrix3f &R, 
    Eigen::Vector3f &T
  ) = 0;

  /**
   * @fn get_statistics
   * @brief Get statistics of the previous tracking phace
   * @param stat statistic struct
   */
  void get_statistics(struct ovo_stat &stat)
  {
    stat.nr_iterations = stat_iterations;
    stat.residual = stat_residual;
  }

  void reset(void)
  {
    lamda = lamda_cfg;
  }

};
  
} // namespace picovo

#endif