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

#ifndef _PICOVO_FRAME_H_
#define _PICOVO_FRAME_H_

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <Eigen/Eigen>

#include <opencv2/opencv.hpp>

#include "distance_transform.h"

namespace picovo
{

typedef Eigen::Matrix<float, 6, 1> Vector6;
typedef Eigen::Matrix<float, 6, 6> Matrix6x6;

/**
 * @struct dt_grad_f
 * @brief The pixel value and gradient of the distance-transform
 *        image, for the normal forward warp algorithm.
 */
struct dt_grad_f
{
  float Iu, Iv, residual;
};

/**
 * @struct compressed_feature
 * @brief The compressed form of feature point, 64-bits in total
 */
struct compressed_feature
{
  int16_t u_cx_fx; // (u-cx)/fx << 12, Q4.12
  int16_t v_cy_fy; // (v-cy)/fy << 12, Q4.12
  uint16_t zinv;   // 1/z << 12, Q4.12
};

/**
 * @struct dt_grad_uv
 * @brief The precalculated Hessian & Steppest-descent parts using
 *  the modified Jacobian formulas, where pixel coordinate (u,v) is
 *  used rather than the projected 3D coordinate (x,y,z). Since the
 *  3D X and Y is not used and (u,v) is just on the projected frame,
 *  most part of the Hessian matrix and Steppest-descent vector can
 *  be pre-calculated. The only uncalculated part is the projected Z
 *  which can be easily be calculated using VMLA instructions.
 *  However, on PC, due to cache-miss, it is sub-optimal.
 *  Refer to:
 *  "Image gradient-based joint direct visual odometry for stereo camera"
 */ 
struct dt_grad_uv
{
  // Hessian parts
  // float H33, H34, H35, H44, H45, H55;

  // Jacobian parts
  float b0, b1, b2, b3, b4, b5;

  float residual;

  /*  The layout of the Hessian matrix and Steppest-descent vector:
   *  H =                                                   b=
   *     / H00/Z2  H01/Z2  H02/Z2  H03/Z   H04/Z   H05/Z  \   / b0/Z \
   *     | --      H11/Z2  H12/Z2  H13/Z   H14/Z   H15/Z  |   | b1/Z |
   *     | --      --      H22/Z2  H23/Z   H24/Z   H25/Z  |   | b2/Z |
   *     | --      --      --      H33     H34     H35    |   | b3   |
   *     | --      --      --      --      H44     H45    |   | b4   |
   *     \ --      --      --      --      --      H55    /   \ b5   /
   * 
   * To reduce memory overhead, Hessian parts are removed.
   */
};

/**
 * @struct dt_grad_ic
 * @brief The pre-calculated Hessian matrix and steppest descent
 *        vector for the inverse-compositional warp algorithm.
 */
struct dt_grad_ic
{
  float H[21];
  float b[6];
  float residual;
};

/**
 * @struct dt_res_nn
 * @brief The distance transform and nearest neighbor of a pixel
 */
struct dt_res_nn
{
  float res;    // DT value
  int dx, dy;   // Nearest neighbor from this pixel
};

/**
 * @struct coo_3d
 * @brief 3D coordinate of a pointcloud element.
 */
struct coo_3d
{
  float x, y, z;
};

/**
 * @struct frame
 * @brief Datastructure of a frame. Since CPP is not a dynamic typed language,
 *        to avoid using xxx_cast, all the needed structs of different algorithms
 *        are provided... Not using Eigen vectors for lower overhead.
 */
struct frame
{
  frame() {
    pcld_buf = NULL;
    dt_bitmap = NULL;
    dt_buf = NULL;
    dt_buf_ic = NULL;
    dt_buf_uv = NULL;
    dtbuf_indexes = NULL;
    dt_to_grad = NULL;
    dt_to_grad_sign = NULL;
    feat = NULL;
    vbobuffer = NULL;
  }

  /**
   * @fn Destructor
   */
  ~frame()
  {
    if (pcld_buf)       free(pcld_buf);
    if (dt_bitmap)      free(dt_bitmap);
    if (dt_buf)         free(dt_buf);
    if (dt_buf_ic)      free(dt_buf_ic);
    if (dt_buf_uv)      free(dt_buf_uv);
    if (dtbuf_indexes)  free(dtbuf_indexes);
    if (dt_to_grad)     free(dt_to_grad);
    if (dt_to_grad_sign) free(dt_to_grad_sign);
    if (feat)           free(feat);
    if (vbobuffer)      free(vbobuffer);
    // printf("Destroy frame %15lf\n", timestamp);
  }

  /* All parameters are public. No need for getter-setters. */

  // Timestamp as the frame ID
  double timestamp;

  /* These images are for debug purpose */
  cv::Mat img_rgb;    // Original colorful image
  cv::Mat img_gray;   // Original grayscale image
  cv::Mat img_depth;  // Original depth image
  cv::Mat img_canny;  // Canny edge image
  cv::Mat img_dt;     // Distance transform of Canny edge

  /* Common parameters */
  int nr_pcld;                // number of points in the point cloud
  int nr_tracked;             // number of tracked points

  /* DT bitmap buffer */
  uint8_t *dt_bitmap;

  /* feature buffers */
  struct coo_3d *pcld_buf;    // common point cloud buffer
  struct compressed_feature *feat;  // fixed-point arithmetic edition

  /* These are for gradient pre-calculation */
  struct dt_grad_f *dt_buf;     // DT residual & gradient buffer for normal
                                // forward warp algorithm. 

  /* These are for Jacobian pre-calculation */
  int32_t *dtbuf_indexes; // Full-map indexes to the following dt_buf_'s
                          // Outlier if < 0; good if == 0, else in the buffer
  struct dt_grad_uv *dt_buf_uv; // Precalculated part of the Hessian & b
  struct dt_grad_ic *dt_buf_ic; // Hessian & b for the inverse-compositional
                                // warp algorithm

  /* These are for the nearest neighbor algorithm */
  uint8_t *dt_to_grad;          // Full-map indexes to the gradient LUT
  uint8_t *dt_to_grad_sign;     // Full-map signs

  /**
   * Pose in the world coordinate system.
   * Left multiplying by it transfers a point coordinate in the camera
   * coordinate system to that in the world coordinate system.
   */
  Eigen::Matrix4f pose;

  // for viewer
  double time_elapsed, track_time;

  // VBO in OpenGL viewer
  unsigned int vbo;
  float *vbobuffer;
};
  
} // namespace picovo

#endif
