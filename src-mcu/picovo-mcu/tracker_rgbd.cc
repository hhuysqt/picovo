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

#include "tracker_rgbd.h"
#include "picoedge.h"
#include "distance_transform.h"
#include <stdio.h>
#include <string.h>

#include "sophus/se3.hpp"
#include "main.h"

// Poses of frames
static Eigen::Matrix4f curr_pose, last_pose, keyf_pose, cand_pose;
// Pose against the keyframe, estimated by solver_track_frame()
static Eigen::Matrix4f pose_vs_keyframe;
// Pose against the last frame, for new frame pose initialization.
static Eigen::Matrix4f pose_vs_lastframe;

// whether tracking is started
static bool is_first_frame;
// edge bitmap of keyframe candidate
static uint8_t kf_edge[IMG_WIDTH*IMG_HEIGHT/8];
// distance transform image
static uint8_t dt_buf[IMG_WIDTH*IMG_HEIGHT];
// threshold of number of features
float nr_feat_thre;

// statistics
extern int stat_iterations;
int stat_nr_features, stat_it1, stat_it2;
float stat_resdl;
bool stat_is_new_kf;
int stat_ed_time;
int stat_feat_time;

extern TIM_HandleTypeDef htim2;

/* private: */

/**
 * @fn makeup_matrix4f
 * @brief Return the 4x4 translation matrix of R & T.
 * @param R Rotation matrix
 * @param T Translation matrix
 */
static Eigen::Matrix4f makeup_matrix4f(Eigen::Matrix3f &R, Eigen::Vector3f &T)
{
  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
  transform.block<3,3>(0,0) = R;
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
static void extract_R_T(Eigen::Matrix4f &pose, Eigen::Matrix3f &R, Eigen::Vector3f &T)
{
  R = pose.block<3,3>(0,0);
  T = pose.block<3,1>(0,3);
}

/* public: */

void init_tracker(void)
{
  init_solver();
  init_picoedge();
  init_dt();
  reset_tracker();
}

void reset_tracker(void)
{
  curr_pose = Eigen::Matrix4f::Identity();
  pose_vs_keyframe = Eigen::Matrix4f::Identity();
  pose_vs_lastframe = Eigen::Matrix4f::Identity();
  is_first_frame = true;
}

void track_frame_rgbd(uint8_t *gray, uint16_t *depth)
{
  // memory reuse: replace the grayscale image with edge & feature structs
  uint32_t start_tim = htim2.Instance->CNT;
  picoedge(gray, gray, IMG_WIDTH, IMG_HEIGHT, EDGE_GRAD_THRESHOLD);
  uint32_t end_tim = htim2.Instance->CNT;
  stat_ed_time = end_tim - start_tim;

  if (is_first_frame) {
    // setup keyframe
    memcpy(kf_edge, gray, sizeof(kf_edge));
    distance_transform(kf_edge, dt_buf, IMG_WIDTH, IMG_HEIGHT);
    keyf_pose = curr_pose;
    last_pose = curr_pose;
    cand_pose = curr_pose;
    nr_feat_thre = 3000;  // guess initial feature number
    stat_resdl = 0;
    stat_is_new_kf = false;
    // Only for gdb debug: `dump binary memory xxx.bin depth (depth+38400)`
    retain_edge_cv8u(gray, (uint8_t*)depth, IMG_WIDTH, IMG_HEIGHT);
    is_first_frame = false;
  } else {
    // select feature points
    struct feature_s *pfeat = (struct feature_s*)(gray + IMG_WIDTH*IMG_HEIGHT/8);
    const double invfx = 1.0 / CAM_FX, invfy = 1.0 / CAM_FY;
    const int32_t invfx_i = invfx * 0x10000, invfy_i = invfy * 0x10000;
    const int32_t cx_i = CAM_CX * 0x1000, cy_i = CAM_CY * 0x1000;
    int nr_chosen = 0;
    // start from line 1
    uint32_t *pedge = (uint32_t*)(gray + IMG_WIDTH/8);
    uint16_t *drow = depth + IMG_WIDTH;
    for (int y = 1; y < IMG_HEIGHT-1; y++, drow += IMG_WIDTH) {
      for (int x = 0; x < IMG_WIDTH; x += 32, pedge++) {
        uint32_t e32 = *pedge;
        while (e32 != 0) {
          int offset = __builtin_ctz(e32);
          e32 &= ~(1 << offset);
          int cur_x = x + offset;
          uint16_t depth = drow[cur_x];
          if (depth > DEPTH_MIN && depth < DEPTH_MAX) {
            pfeat[nr_chosen].u_cx_fx = (((cur_x << 12) - cx_i) * invfx_i) >> 16;  // Q4.12
            pfeat[nr_chosen].v_cy_fy = (((y << 12) - cy_i) * invfy_i) >> 16;  // Q4.12
            pfeat[nr_chosen].zinv = (5000 << 12) / depth;  // Q4.12
            nr_chosen++;
          }
        }
      }
    }
    stat_nr_features = nr_chosen;
    uint32_t end_tim2 = htim2.Instance->CNT;
    stat_feat_time = end_tim2 - end_tim;

    // Track this frame. Assume a uniform speed model for pose initialization
    Eigen::Matrix4f new_pose_vs_keyframe = pose_vs_keyframe * pose_vs_lastframe;
    Eigen::Matrix3f new_R2kf;
    Eigen::Vector3f new_T2kf;
    extract_R_T(new_pose_vs_keyframe, new_R2kf, new_T2kf);

    int nr_tracked = nr_chosen;
    float err = solver_track_frame(dt_buf, pfeat, nr_tracked, new_R2kf, new_T2kf);
    stat_it1 = stat_iterations;

    Sophus::SE3f delta(makeup_matrix4f(new_R2kf, new_T2kf));
    float se3norm = delta.log().norm();

    if (err > 1.1 || nr_tracked < nr_chosen * 0.66 || se3norm > 0.25
    ) {
      distance_transform(kf_edge, dt_buf, IMG_WIDTH, IMG_HEIGHT);
      keyf_pose = cand_pose;

      // Eigen::Matrix4f new_pose = 
      //   key_frame->pose * makeup_matrix4f(new_R2kf, new_T2kf);
      // new_pose_vs_keyframe = key_frame->pose.inverse() * new_pose;
      new_pose_vs_keyframe = Eigen::Matrix4f::Identity();
      extract_R_T(new_pose_vs_keyframe, new_R2kf, new_T2kf);

      // retrack this frame
      nr_tracked = nr_chosen;
      err = solver_track_frame(dt_buf, pfeat, nr_tracked, new_R2kf, new_T2kf);
      stat_it2 = stat_iterations;
    } else {
      stat_it2 = 0;
    }

    auto tmppose_vs_keyframe = makeup_matrix4f(new_R2kf, new_T2kf);
    curr_pose = keyf_pose * tmppose_vs_keyframe;
    if (err < 1.1) {
      pose_vs_keyframe = tmppose_vs_keyframe;
      pose_vs_lastframe = last_pose.inverse() * curr_pose;
    } else {
      pose_vs_keyframe = pose_vs_keyframe * pose_vs_lastframe;
    }

    // Apply a low-pass filter to frame::nr_pcld
    const float alpha = 0.7;
    nr_feat_thre = alpha * nr_feat_thre + (1-alpha) * nr_chosen;
    // If the new frame has enough features, it can be the keyframe candidate.
    const float beta = 0.7;
    if (nr_chosen > beta * nr_feat_thre) {
      memcpy(kf_edge, gray, sizeof(kf_edge));
      cand_pose = curr_pose;
    }

    last_pose = curr_pose;
    stat_resdl = err;
  }
}

void get_current_pose_tum_str(char *outbuf, int bufsize)
{
  Eigen::Matrix3f R = curr_pose.block<3,3>(0,0);
  Eigen::Vector3f T = curr_pose.block<3,1>(0,3);
  const Eigen::Quaternionf Qf(R);
  snprintf(outbuf, bufsize, "%9f %9f %9f %9f %9f %9f %9f\n", 
    T(0), T(1), T(2), Qf.x(), Qf.y(), Qf.z(), Qf.w());
}

void get_stat_str(char *outbuf, int bufsize)
{
  snprintf(outbuf, bufsize, "%d, %9f, %d, %d, %d, %d\n",
    stat_nr_features,
    stat_resdl, 
    stat_it1,
    stat_it2,
    stat_ed_time,
    stat_feat_time
  );
}
