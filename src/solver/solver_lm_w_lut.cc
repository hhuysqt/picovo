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

#include "solver_lm_w_lut.h"

using namespace picovo;

// #define DEBUG_OPTIMIZATION

float solver_lm_w_lut::track_frame(
  std::shared_ptr<frame> &ref_frame,
  std::shared_ptr<frame> &cur_frame,
  Eigen::Matrix3f &R, 
  Eigen::Vector3f &T
)
{
#ifdef DEBUG_OPTIMIZATION
  curr_canny = ref_frame->img_canny;
#endif

  // projection from cur_frame to ref_dt
  uint8_t *ref_dt_lut = ref_frame->dt_to_grad;
  struct coo_3d *cur_pcl = cur_frame->pcld_buf;
  int nr_proj = cur_frame->nr_pcld;

  // estimated SE3
  Sophus::SE3f ref2cur(R,T);
  cur_frame->nr_tracked = nr_proj;
  Matrix6x6 H;
  Vector6 b;

  float last_min_resdl;
  last_min_resdl = _calc_update_lm(ref_dt_lut, cur_pcl, 
    cur_frame->nr_tracked, 4, huber_weight * _outlier_threshold, R, T, H, b);

  if (lamda > 3) lamda = 3.2;

  int its = 0;
  uint32_t res_inc_history = 0;
  for (; its < 100; its++) {
    H += (1+lamda) * Matrix6x6::Identity();
    Vector6 inc = H.ldlt().solve(b);

    Sophus::SE3f new_ref2cur = Sophus::SE3f::exp(inc) * ref2cur;
    int featoffset = its < 4 ? its : its < 6 ? its-4 : 0;
    int stepsize = its < 4 ? 4 : its < 6 ? 2 : 1;
    cur_frame->nr_tracked = nr_proj & 0xfffffffc;
    float cur_resdl;
    cur_resdl = _calc_update_lm(ref_dt_lut, cur_pcl + featoffset, 
      cur_frame->nr_tracked, stepsize, huber_weight * _outlier_threshold,
      new_ref2cur.rotationMatrix(), new_ref2cur.translation(),
      H, b);

    if (cur_resdl < last_min_resdl) {
      // Set a more radical step size.
      lamda = lamda < 0.3 ? 0 : lamda/2;
      // Check convergence
      if (cur_resdl / last_min_resdl > 0.999f) {
        ref2cur = new_ref2cur;
        break;
      }
      last_min_resdl = cur_resdl;
      res_inc_history = (res_inc_history << 1) | 0;
    } else if (its > 6 && (res_inc_history & 0xf) == 0xf) {
#ifdef DEBUG_OPTIMIZATION
      std::cout << "Residual increases for more than 4 times. Break."
        << std::endl;
#endif
      break;
    } else if (lamda > 5000 || inc.dot(inc) < 1e-16) {
#ifdef DEBUG_OPTIMIZATION
      std::cout << "Not continuing with small increment. " <<
        its << " iterations." << std::endl;
#endif
      break;
    } else {
      res_inc_history = (res_inc_history << 1) | 0x1;
      // Set a more conservative step size.
      // If the residual is trembling, be even more conservative.
      lamda = lamda < 0.3 ? 0.4 : lamda * ((res_inc_history & 0x7) == 0x5 ? 4 : 2);
    }
#ifdef DEBUG_OPTIMIZATION
    printf(" lamda: %f, hist: %x, resdl: %f vs %f\n", lamda, res_inc_history & 0xf, 
      cur_resdl, last_min_resdl);
#endif
    ref2cur = new_ref2cur;
  }

  // Return results
  R = ref2cur.rotationMatrix();
  T = ref2cur.translation();

  // int valid_cnt = _cnt_valid_proj(ref_dt_lut, cur_pcl, nr_proj, R, T);
  // if (cur_frame->nr_tracked < cur_frame->nr_pcld * 0.66 ||
  //     valid_cnt < ref_frame->nr_pcld * 0.66
  // ) {
  //   is_kf_needed = true;
  // } else {
  //   is_kf_needed = false;
  // }

#ifdef DEBUG_OPTIMIZATION
  std::cout << "R: " 
    << R.col(0).transpose() << " "
    << R.col(1).transpose() << " "
    << R.col(2).transpose() << ", "
    "T: " << T.transpose() << std::endl;
  
  // std::cout << "\n  projected: " << valid_cnt << " vs " << ref_frame->nr_pcld
  //   << std::endl;
  getchar();
#endif

  stat_iterations = its;
  stat_residual = last_min_resdl;
  return last_min_resdl;
}

float solver_lm_w_lut::_calc_update_lm(
  const uint8_t *ref_dt, 
  const struct coo_3d *cur_pcl, 
  int &nr_proj,
  int step,
  float outlier_thrshd,
  const Eigen::Matrix3f &R, 
  const Eigen::Vector3f &T, 
  Matrix6x6 &H,
  Vector6 &b
)
{
#ifdef DEBUG_OPTIMIZATION
  cv::Mat proj_ref;
  cv::cvtColor(curr_canny, proj_ref, cv::COLOR_GRAY2BGR);
#endif

  float total_res = 0;
  int nr_valid_proj = 0;

  H.setZero();
  b.setZero();

  for (int i = 0; i < nr_proj; i += step) {
    Eigen::Vector3f pcur;
    pcur << cur_pcl[i].x, cur_pcl[i].y, cur_pcl[i].z;
    // rigid body tranformation
    Eigen::Vector3f pref = R*pcur + T;
    // camera projection
    const float z_inv = 1.0 / pref[2];
    const float x_z = pref[0] * z_inv, y_z = pref[1] * z_inv;
    const float u = x_z * fx + cx;
    const float v = y_z * fy + cy;
    if (u <= 1 || v <= 1 || u >= width-2 || v >= height-2) {
      // out of image boundary
      continue;
    }
    const int u_i = u;
    const int v_i = v;
    const uint pixindex = u_i + v_i*width;
    const uint8_t *pcurdist = ref_dt + pixindex;
    const uint sqr_dist = *pcurdist;
    if (sqr_dist >= 0xe1) {
      // Skip this outlier
      continue;
    }

# ifdef DEBUG_OPTIMIZATION
    cv::circle(proj_ref, cv::Point(u, v), 2, cv::Scalar(0,255,0));
# endif

    if (sqr_dist == 0) {
      nr_valid_proj++;
      continue;
    }

    float residual = lut_residual[sqr_dist];
    uint iuindex = ((uint)(*(pcurdist-1)) << 8) | (*(pcurdist+1));
    uint ivindex = ((uint)(*(pcurdist-width)) << 8) | (*(pcurdist+width));
    float Iu = lut_gradient[iuindex];
    float Iv = lut_gradient[ivindex];

    // Optimized Jacobian calculation. Note that the Huber, fx and fy have
    // been multiplied in Iu & Iv before.
    Vector6 J;
    float Iu_x_z_Iv_y_z = Iu * x_z + Iv * y_z;
    J[0] = Iu * z_inv;
    J[1] = Iv * z_inv;
    J[2] =  -Iu_x_z_Iv_y_z * z_inv;
    J[3] = -(Iu_x_z_Iv_y_z * y_z + Iv);
    J[4] =  (Iu_x_z_Iv_y_z * x_z + Iu);
    J[5] = -Iu * y_z + Iv * x_z;

    H += J*J.transpose() / residual;
    b += J;

    total_res += residual;
    nr_valid_proj++;
  }

  H /= nr_valid_proj;
  b /= nr_valid_proj;

  nr_proj = nr_valid_proj;

# ifdef DEBUG_OPTIMIZATION
  std::cout << "  H= " << H.row(0) << " " << H.row(1)  << " " << H.row(2)
    << ", " << H.row(3) << ", " << H.row(4)  << ", " << H.row(5)
    << "\n b= " << b.transpose() << "\n " 
    << nr_proj << " projections, sum residual " << total_res << std::endl;

  cv::imshow("proj res", proj_ref);
  cv::waitKey(10);
# endif

  return total_res / nr_valid_proj;
}
