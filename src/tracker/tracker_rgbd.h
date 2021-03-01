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

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Eigen>

#include "tracker_base.h"
#include "picovo_frame.h"

namespace picovo
{

class tracker_rgbd : public tracker_base
{
private:

  /**
   * @fn initialize
   * @brief initialize with config
   * @param config system config structure
   */
  void _initialize(picovo_config &config);

  /**
   * @fn add_curr_frame
   * @brief Setup this->curr_frame for multiple solvers
   * @param timestamp double timestamp in seconds
   * @param img_gray CV_8U grayscale image
   * @param img_depth CV_16U depth image
   */
  void _add_curr_frame(double timestamp, cv::Mat img_gray, cv::Mat img_depth);

  /**
   * @fn setup_key_frame
   * @brief Setup this->key_frame for multiple solvers
   */
  void _setup_key_frame(void);

  /**
   * @fn track_one_frame
   * @brief Track a new frame
   */
  void _track_curr_frame(void);

  // LUT for inverse-compositional method
  float lut_nn_ic[256];

public:

  tracker_rgbd() {}

  ~tracker_rgbd() {}

  /**
   * @fn start
   * @brief start the tracker
   * @param config system config structure
   */
  void start(picovo_config &config);

};

}

#endif
