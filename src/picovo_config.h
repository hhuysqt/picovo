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

#include <string>
#include <iostream>
#include <vector>
#include <opencv2/core.hpp>

namespace picovo
{

/**
 * @enum tracker_enum
 * @brief Choose a tracker type
 */
enum tracker_enum {
  TRACKER_RGBD_DATASET = 1,
  TRACKER_STEREO_DATASET = 2
};

/**
 * @enum solver_enum
 * @brief Choose an optimization solver. They use different datastructures,
 *        and have different speed and accuracy. Basically, a Gradient
 *        Descent (GD) is faster in each iteration than a Levenberg-Marquard
 *        (LM), but has slower convergence rate and less accuracy.
 */
enum solver_enum {
  SOLVER_GD_PRE_GRAD = 1,   // Gradient Descent with DT gradient pre-calculation
  SOLVER_GD_PRE_JACOBIAN,   // GD with Jacobian pre-calculation
  SOLVER_LM_BASE,           // Basic Levenberg-Marquard solver
  SOLVER_LM_PRE_GRAD,       // LM with DT gradient pre-calculation
  SOLVER_LM_PRE_JACOBIAN,   // LM with Jacobian pre-calculation
  SOLVER_LM_NN,             // LM using nearest neighbor
  SOLVER_LM_IC,             // LM using inverse-compositional method
  SOLVER_LM_FIX_POINT       // LM using fixed-point arithmetic
};

struct picovo_config
{
  // image shape
  int width, height;

  // feature parameters
  int edge_threshold, depth_min, depth_max;

  // camera intrinsics
  float fx, fy, cx, cy;

  // solver parameters. better keep default.
  float solver_huber_weight, solver_outlier_threshold, solver_lamda;

  // tracker & solver selection
  int tracker_type;
  int solver_type;

  std::vector<std::string> dataset_folder;

  // visualization control
  bool is_pause;
  bool is_use_canny;
  bool is_show_imgs;
  bool is_view_animation;
  bool is_view_fullscreen;
  bool is_view_show_imgs;
  bool is_view_groundtruth;
  bool is_capture_video;
  std::string capture_video_name;
  int capture_framerate;
  int viewer_height, viewer_width;

  /**
   * @fn constructor
   * @brief default settings without a config file
   */
  picovo_config()
  {
    width = 640;
    height = 480;
    fx = 517.306408;
    fy = 516.469215;
    cx = 318.643040;
    cy = 255.313989;
    edge_threshold = 5;
    depth_min = 625;
    depth_max = 5500;
    solver_huber_weight = 0.3;
    solver_outlier_threshold = 14;
    solver_lamda = 3.2;
    is_pause = false;
    is_use_canny = false;
    is_show_imgs = false;

    tracker_type = TRACKER_RGBD_DATASET;
    solver_type = SOLVER_LM_FIX_POINT;
    dataset_folder.clear();
  }

  /**
   * @fn constructor
   * @brief Settings without a config file
   */
  picovo_config(const std::string& filename)
  {
    cv::FileStorage configfile(filename, cv::FileStorage::READ);
    if (!configfile.isOpened()) {
      std::cerr << "Failed to open config file: " << filename << std::endl;
      exit(-1);
    }
    std::cout << "Reading config file: " << filename << std::endl;

    cv::read(configfile["width"], width, 640);
    cv::read(configfile["height"], height, 480);

    cv::read(configfile["fx"], fx, 516.8878115);
    cv::read(configfile["fy"], fy, 516.8878115);
    cv::read(configfile["cx"], cx, 318.643040);
    cv::read(configfile["cy"], cy, 255.313989);

    cv::read(configfile["edge.threshold"], edge_threshold, 5);
    cv::read(configfile["feature.depth.min"], depth_min, 625);
    cv::read(configfile["feature.depth.max"], depth_max, 55000);

    cv::read(configfile["tracker.type"], tracker_type, TRACKER_RGBD_DATASET);
    cv::read(configfile["tracker.is_pause"], is_pause, false);
    cv::read(configfile["tracker.use_canny"], is_use_canny, false);
    cv::read(configfile["tracker.show_imgs"], is_show_imgs, false);

    cv::read(configfile["solver.type"], solver_type, SOLVER_LM_FIX_POINT);
    // These paramters are better kept default.
    cv::read(configfile["solver.lamda"], solver_lamda, 3.2);
    cv::read(configfile["solver.huber_weight"], solver_huber_weight, 0.3);
    cv::read(configfile["solver.outlier_threshold"], solver_outlier_threshold, 14);

    cv::read(configfile["viewer.do_animation"], is_view_animation, false);
    cv::read(configfile["viewer.full_screen"], is_view_fullscreen, false);
    cv::read(configfile["viewer.show_imgs"], is_view_show_imgs, false);
    cv::read(configfile["viewer.groundtruth"], is_view_groundtruth, false);
    cv::read(configfile["viewer.width"], viewer_width, 1024);
    cv::read(configfile["viewer.height"], viewer_height, 768);
    cv::read(configfile["viewer.capture_video"], is_capture_video, false);
    if (is_capture_video) {
      cv::read(configfile["viewer.capture_name"], capture_video_name, "out.mp4");
      cv::read(configfile["viewer.capture_framerate"], capture_framerate, 30);
    }
    dataset_folder.clear();
    cv::FileNode fn = configfile["dataset_folders"];
    if (fn.type() != cv::FileNode::SEQ) {
      std::cerr << "dataset_folders is not a sequence..." << std::endl;
      exit(-1);
    }
    std::cout << "Found datasets:" << std::endl;
    for (cv::FileNodeIterator it = fn.begin(); it != fn.end(); it++) {
      std::string ds = (std::string)*it;
      std::cout << ds << std::endl;
      dataset_folder.push_back(ds);
    }
  }

  ~picovo_config() {}
};

struct picovo_stat {
  double pre_process_us;
  double track_us;
  double residual;
  int nr_features;
  int nr_iterations;

  picovo_stat& operator += (const picovo_stat &b) {
    this->pre_process_us += b.pre_process_us;
    this->track_us += b.track_us;
    this->residual += b.residual;
    this->nr_features += b.nr_features;
    this->nr_iterations += b.nr_iterations;
    return *this;
  }
};

}

#endif
