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

#ifndef TRACKER_BASE_H
#define TRACKER_BASE_H

#include "picovo_config.h"

#include "solver_base.h"
#ifdef USE_PANGOLIN
#include "viewer.h"
#endif

namespace picovo
{

enum tracker_state {
  TRACK_IDLE,
  TRACK_OKAY,
  TRACK_LOST
};

class tracker_base
{
protected:

  /* Select a solver: GD, LM, w/wo pre-calculation */
  solver_enum solver_type;
  solver_base *solver;

  /**
   * We only need to manage 4 frames:
   *  the key frame for pose estimation,
   *  a candidate for the new key frame,
   *  the last frame for pose initialization,
   *  and the current frame.
   */
  std::shared_ptr<frame> key_frame;
  std::shared_ptr<frame> key_frame_candidate;
  std::shared_ptr<frame> last_frame;
  std::shared_ptr<frame> curr_frame;

  bool is_just_add_kf;

#ifdef USE_PANGOLIN
  // Pangolin OpenGL viewer
  std::shared_ptr<viewer> my_viewer;
#endif

  // Pose against the keyframe, estimated by solver::track_frame()
  Eigen::Matrix4f pose_vs_keyframe;
  // Pose against the last frame, for new frame pose initialization.
  Eigen::Matrix4f pose_vs_lastframe;

  /**
   * @fn makeup_matrix4f
   * @brief Return the 4x4 translation matrix of R & T.
   * @param R Rotation matrix
   * @param T Translation matrix
   */
  Eigen::Matrix4f makeup_matrix4f(Eigen::Matrix3f &R, Eigen::Vector3f &T)
  {
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3,3>(0,0) = R;
    transform.block<3,1>(0,3) = T;
    return transform;
  }

  /**
   * @fn makeup_matrix4f
   * @brief Return the 4x4 translation matrix of R & T.
   * @param Q Quaternion
   * @param T Translation matrix
   */
  Eigen::Matrix4f makeup_matrix4f(Eigen::Quaternionf &Q, Eigen::Vector3f &T)
  {
    Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
    transform.block<3,3>(0,0) = Q.toRotationMatrix();
    transform.block<3,1>(0,3) = T;
    return transform;
  }

  /**
   * @fn extract_R_T
   * @brief Extract rotation and translation matrix from 4x4 pose matrix
   * @param pose 4x4 pose matrix
   * @param R rotation matrix
   * @param T translation matrix
   */
  void extract_R_T(Eigen::Matrix4f &pose, Eigen::Matrix3f &R, Eigen::Vector3f &T)
  {
    R = pose.block<3,3>(0,0);
    T = pose.block<3,1>(0,3);
  }

  /* Parameters */
  bool is_use_canny;
  bool is_show_imgs;
  bool is_view_animation;
  bool is_view_groundtruth;
  float nr_feat_thre;           // threshold of number of features
  int edge_threshold;           // edge: gradient threshold
  int depth_min, depth_max;     // feature: depth threshold
  int width, height;            // image shape
  float fx, fy, cx, cy;         // Camera intrinsics
  float huber_weight;           // Huber weight for optimization
  float _outlier_threshold;     // Outlier threshold of a projecton pixel

public:

  tracker_base() {}

  ~tracker_base() {}

  /**
   * @fn start
   * @brief start the tracker
   * @param config system config structure
   */
  virtual void start(picovo_config &config) = 0;

  tracker_state curr_state;
};
  
} // namespace picovo

#endif
