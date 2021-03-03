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

#ifndef PICOVO_VIEWER_H
#define PICOVO_VIEWER_H

#include <Eigen/Core>
#include <pangolin/pangolin.h>
#include <vector>
#include <thread>
#include <opencv2/core/core.hpp>
#include "sophus/se3.hpp"
#include "picovo_frame.h"
#include "picovo_config.h"

namespace picovo
{

class viewer
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  viewer(picovo_config &config);

  void close();

  void show_curr_frame(std::shared_ptr<frame> current_frame);
  void add_key_frame(std::shared_ptr<frame> key_frame);
  void play_animation(void);
  void clear_all(void);

private:

  /**
   * @fn viewer_loop
   * @brief viewer thread
   */
  void viewer_loop();

  /**
   * @fn draw_camera
   * @brief draw camera frame
   * @param frame the camera to draw
   * @param color float[3] RGB color. NULL for default red.
   */
  void draw_camera(std::shared_ptr<frame> frame, const float* color);

  /**
   * @fn draw_pcld
   * @brief draw point cloud
   */
  void draw_pcld();

  /**
   * @fn follow_current_frame
   * @brief set the OpenGL view to follow current frame
   * @param vis_camera Pangolin OpenGL camera
   */
  void follow_current_frame(pangolin::OpenGlRenderState& vis_camera);

  /**
   * @fn go_smoothly_to_pose
   * @brief move the camera smoothly to a pose
   * @param pose pose in world
   * @param vis_camera Pangolin OpenGL camera
   */
  void go_smoothly_to_pose(Eigen::Matrix4f pose, pangolin::OpenGlRenderState& vis_camera);

  std::thread viewer_thread_;

  bool is_curr_frame_updated, is_keyframe_updated;

  // current frame
  std::shared_ptr<frame> curr_frame = nullptr;

  // a list of active keyframes
  std::vector<std::shared_ptr<struct frame> > active_keyframes_;

  // data mutexes
  std::mutex current_lock;
  std::mutex kf_lock;
  std::mutex running_lock;

  // camera intrinsics
  float fx, fy, cx, cy;

  int win_width, win_height;
  bool is_show_img, is_fullscreen;

  // display information
  cv::Mat cur_rgb;
  cv::Mat bar_mat;
  std::string solver_name;

  // for animation
  Eigen::Matrix4f view_pose;
  std::mutex animation_lock;
  bool is_start_animation;
  int ani_kf_index;
  double alpha;
};

}

#endif
