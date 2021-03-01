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

#include "solver_lm_ic.h"

using namespace picovo;

// #define DEBUG_OPTIMIZATION

float solver_lm_ic::track_frame(
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
  struct dt_grad_ic *ref_dt = ref_frame->dt_buf_ic;
  int32_t *ref_dt_indexes = ref_frame->dtbuf_indexes;
  struct coo_3d *cur_pcl = cur_frame->pcld_buf;
  int nr_proj = cur_frame->nr_pcld;

  // estimated SE3
  asm volatile("": : :"memory");
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      double rij = R(i,j);
      if (rij > -0.000001 && rij < 0.000001) {
        R(i,j) = 0.0;
      } else if (rij > 1.0) {
        R(i,j) = 0.999999;
      }
    }
  }
  asm volatile("": : :"memory");
  Sophus::SE3f ref2cur(R,T);
  cur_frame->nr_tracked = nr_proj;
  Matrix6x6 H;
  Vector6 b;

  float last_min_resdl;
  last_min_resdl = _calc_update_lm_ic(ref_dt, ref_dt_indexes, cur_pcl, 
    cur_frame->nr_tracked, 4, R, T, H, b);

  if (lamda > 3) lamda = 3.2;

  int its = 0;
  uint32_t res_inc_history = 0;
  for (; its < 100; its++) {
    H += (1+lamda) * Matrix6x6::Identity();
    Vector6 inc = H.ldlt().solve(b);
    if (!std::isnormal(inc[0])) {
      // skip the following procedures
      return 10;
    }

    Sophus::SE3f new_ref2cur = Sophus::SE3f::exp(inc) * ref2cur;
    int featoffset = its < 4 ? its : its < 6 ? its-4 : 0;
    int stepsize = its < 4 ? 4 : its < 6 ? 2 : 1;
    cur_frame->nr_tracked = nr_proj & 0xfffffffc;
    float cur_resdl;
    cur_resdl = _calc_update_lm_ic(ref_dt, ref_dt_indexes, cur_pcl + featoffset, 
    cur_frame->nr_tracked, stepsize,
      new_ref2cur.rotationMatrix(), new_ref2cur.translation(),
      H, b);

    if (cur_resdl < last_min_resdl) {
      // Set a more radical step size.
      lamda = lamda < 0.5 ? 0.4 : lamda/2;
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
      lamda = lamda < 0.5 ? 0.8 : lamda * ((res_inc_history & 0x7) == 0x5 ? 4 : 2);
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

#ifdef DEBUG_OPTIMIZATION
  std::cout << "R: " 
    << R.col(0).transpose() << " "
    << R.col(1).transpose() << " "
    << R.col(2).transpose() << ", "
    "T: " << T.transpose() << std::endl;
  // getchar();
#endif

  stat_iterations = its;
  stat_residual = last_min_resdl;
  return last_min_resdl;
}

float solver_lm_ic::_calc_update_lm_ic(
  const struct dt_grad_ic *ref_dt, 
  int32_t *ref_dt_indexes,
  const struct coo_3d *cur_pcl, 
  int &nr_proj,
  int step,
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

  float H_tmp[21] = { 0 }, b_tmp[6] = { 0 };

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
    const int u_i = static_cast<int>(u + 0.5);
    const int v_i = static_cast<int>(v + 0.5);
    int32_t proj_index = ref_dt_indexes[v_i*width + u_i];
    if (proj_index < 0) {
      // outlier
      continue;
    }

# ifdef DEBUG_OPTIMIZATION
    cv::circle(proj_ref, cv::Point(u, v), 1, cv::Scalar(0,255,0));
# endif

    if (proj_index == 0) {
      nr_valid_proj++;
      continue;
    }

    // Jacobian and Hessian matrixes are all pre-calculated
    const struct dt_grad_ic *pbuf = ref_dt + proj_index;
    for (int i = 0; i < 21; i++) {
      H_tmp[i] += pbuf->H[i];
    }
    for (int i = 0; i < 6; i++) {
      b_tmp[i] += pbuf->b[i];
    }

    float residual = pbuf->residual;
    total_res += residual;
    nr_valid_proj++;
  }

  H(0,0) =          H_tmp[0] / nr_valid_proj;
  H(0,1) = H(1,0) = H_tmp[1] / nr_valid_proj;
  H(0,2) = H(2,0) = H_tmp[2] / nr_valid_proj;
  H(0,3) = H(3,0) = H_tmp[3] / nr_valid_proj;
  H(0,4) = H(4,0) = H_tmp[4] / nr_valid_proj;
  H(0,5) = H(5,0) = H_tmp[5] / nr_valid_proj;
  H(1,1) =          H_tmp[6] / nr_valid_proj;
  H(1,2) = H(2,1) = H_tmp[7] / nr_valid_proj;
  H(1,3) = H(3,1) = H_tmp[8] / nr_valid_proj;
  H(1,4) = H(4,1) = H_tmp[9] / nr_valid_proj;
  H(1,5) = H(5,1) = H_tmp[10] / nr_valid_proj;
  H(2,2) =          H_tmp[11] / nr_valid_proj;
  H(2,3) = H(3,2) = H_tmp[12] / nr_valid_proj;
  H(2,4) = H(4,2) = H_tmp[13] / nr_valid_proj;
  H(2,5) = H(5,2) = H_tmp[14] / nr_valid_proj;
  H(3,3) =          H_tmp[15] / nr_valid_proj;
  H(3,4) = H(4,3) = H_tmp[16] / nr_valid_proj;
  H(3,5) = H(5,3) = H_tmp[17] / nr_valid_proj;
  H(4,4) =          H_tmp[18] / nr_valid_proj;
  H(4,5) = H(5,4) = H_tmp[19] / nr_valid_proj;
  H(5,5) =          H_tmp[20] / nr_valid_proj;
  b(0) = b_tmp[0] / nr_valid_proj;
  b(1) = b_tmp[1] / nr_valid_proj;
  b(2) = b_tmp[2] / nr_valid_proj;
  b(3) = b_tmp[3] / nr_valid_proj;
  b(4) = b_tmp[4] / nr_valid_proj;
  b(5) = b_tmp[5] / nr_valid_proj;

  nr_proj = nr_valid_proj;

# ifdef DEBUG_OPTIMIZATION
  std::cout << "  H= " << H.row(0) << " " << H.row(1)  << " " << H.row(2)
    << ", " << H.row(3) << ", " << H.row(4)  << ", " << H.row(5)
    << "\n b= " << b.transpose() << "\n " 
    << nr_proj << " projections, sum residual " << total_res << std::endl;

  cv::imshow("proj res", proj_ref);
  cv::waitKey(5);
# endif

  return total_res / nr_valid_proj;
}
