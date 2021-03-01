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

#include "solver_lm_fixpoint.h"

using namespace picovo;

// #define DEBUG_OPTIMIZATION

float solver_lm_fixpoint::track_frame(
  std::shared_ptr<frame> &ref_frame,
  std::shared_ptr<frame> &cur_frame,
  Eigen::Matrix3f &R, 
  Eigen::Vector3f &T
)
{
#ifdef DEBUG_OPTIMIZATION
  curr_canny = 255-ref_frame->img_canny;
  cv::Mat dttmp;
  cv::distanceTransform(curr_canny, dttmp, cv::DIST_L2, cv::DIST_MASK_PRECISE);
  cv::threshold(dttmp, dttmp, 15, 1.0, cv::THRESH_TRUNC);
  cv::normalize(dttmp, dttmp, 0, 1, cv::NORM_MINMAX);
  cv::imshow("dt", dttmp);
#endif

  // projection from cur_frame to ref_dt
  uint8_t *ref_dt_lut = ref_frame->dt_to_grad;
  struct compressed_feature *pfeat = cur_frame->feat;
  int nr_proj = cur_frame->nr_pcld;

  // estimated SE3
  Sophus::SE3f ref2cur(R,T);
  Matrix6x6 H;
  Vector6 b;

  float last_min_resdl;
  cur_frame->nr_tracked = nr_proj;
  last_min_resdl = _calc_update_lm_fp(ref_dt_lut, pfeat, 
    cur_frame->nr_tracked, 4, R, T, H, b);

  if (lamda > 3) lamda = 3.2;

  int its = 0;
  uint32_t res_inc_history = 0;
  for (; its < 100; its++) {
    H += (1+lamda) * Matrix6x6::Identity();
    Vector6 inc;
    inc = H.ldlt().solve(b);

    Sophus::SE3f new_ref2cur = Sophus::SE3f::exp(inc) * ref2cur;
    float cur_resdl;

    int featoffset = its < 4 ? its : its < 6 ? its-4 : 0;
    int stepsize = its < 4 ? 4 : its < 6 ? 2 : 1;
    cur_frame->nr_tracked = nr_proj & 0xfffffffc;
    cur_resdl = _calc_update_lm_fp(ref_dt_lut, pfeat + featoffset, 
      cur_frame->nr_tracked, stepsize,
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

#ifdef DEBUG_OPTIMIZATION
  std::cout << "R: " 
    << R.row(0) << " " << R.row(1) << " " << R.row(2) << ", "
    "T: " << T.transpose() << std::endl;

  cv::waitKey();
#endif

  stat_iterations = its;
  stat_residual = last_min_resdl;
  return last_min_resdl;
}

float solver_lm_fixpoint::_calc_update_lm_fp(
  const uint8_t *ref_dt, 
  const struct compressed_feature *pfeat, 
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
  Matrix6x6 H_1;
  Vector6 b_1;
  H_1.setZero();
  b_1.setZero();
#endif

  int32_t total_res = 0;
  int nr_valid_proj = 0;

  /* Parts of the Hessian matrix, 32+0 bits */
  int32_t H00 = 0, H01 = 0, H02 = 0, H03 = 0, H04 = 0, H05 = 0,
                   H11 = 0, H12 = 0, H13 = 0, H14 = 0, H15 = 0,
                            H22 = 0, H23 = 0, H24 = 0, H25 = 0,
                                     H33 = 0, H34 = 0, H35 = 0,
                                              H44 = 0, H45 = 0,
                                                       H55 = 0;
  /* Parts of the steppest descent, 21+11 bits */
  int32_t b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0;

  // Q15 convertion for R & T. Rij & Ti should all be smaller than 1.
  int16_t ri00 = R(0,0) * ((1L << 15) - 1);
  int16_t ri01 = R(0,1) * ((1L << 15) - 1);
  int16_t ri02 = R(0,2) * ((1L << 15) - 1);
  int16_t ri10 = R(1,0) * ((1L << 15) - 1);
  int16_t ri11 = R(1,1) * ((1L << 15) - 1);
  int16_t ri12 = R(1,2) * ((1L << 15) - 1);
  int16_t ri20 = R(2,0) * ((1L << 15) - 1);
  int16_t ri21 = R(2,1) * ((1L << 15) - 1);
  int16_t ri22 = R(2,2) * ((1L << 15) - 1);
  int16_t ti0 = T(0) * (1L << 15);
  int16_t ti1 = T(1) * (1L << 15);
  int16_t ti2 = T(2) * (1L << 15);

  // Fixed-point convertion for camera intrinsics
  int32_t fx_i = fx * (1L << 6);  // Q10.6
  int32_t fy_i = fy * (1L << 6);  // Q10.6
  int32_t cx_i = (cx + 0.5) * (1L << 18); // Q9.18
  int32_t cy_i = (cy + 0.5) * (1L << 18); // Q9.18

  for (int i = 0; i < nr_proj; i += step) {
    auto feature = pfeat[i];
    int16_t ucxfx = feature.u_cx_fx; // Q4.12
    int16_t vcyfy = feature.v_cy_fy; // Q4.12
    int16_t zinv =  feature.zinv;    // Q4.12
    // RP+T: 1+27 bits. res0 & res1: almost (-0.5,0.5), res2: close to 1
    int32_t res0 = (int32_t)ri00 * ucxfx + (int32_t)ri01 * vcyfy + ((int32_t)ri02 << 12) + (int32_t)ti0 * zinv;
    int32_t res1 = (int32_t)ri10 * ucxfx + (int32_t)ri11 * vcyfy + ((int32_t)ri12 << 12) + (int32_t)ti1 * zinv;
    int32_t res2 = (int32_t)ri20 * ucxfx + (int32_t)ri21 * vcyfy + ((int32_t)ri22 << 12) + (int32_t)ti2 * zinv;
    int32_t inv_res2 = (1U << 31) / (res2 >> 11); // 31-16 = 15
    res0 >>= 15, res1 >>= 15;
    int32_t x_z_i = (res0 * inv_res2) >> 15;  // Q4.12
    int32_t y_z_i = (res1 * inv_res2) >> 15;  // Q4.12
    int32_t u = cx_i, v = cy_i;
    u += fx_i * x_z_i; u >>= 18;
    v += fy_i * y_z_i; v >>= 18;

    if (u <= 1 || v <= 1 || u >= width-2 || v >= height-2) {
      // out of image boundary
      continue;
    }
    const uint pixindex = u + v*width;
    const uint8_t *pcurdist = ref_dt + pixindex;
    const uint sqr_dist = *pcurdist;
    if (sqr_dist >= 0xe1) {
      // Skip this outlier
      continue;
    }

# ifdef DEBUG_OPTIMIZATION
    cv::circle(proj_ref, cv::Point(u, v), 1, cv::Scalar(0,0,250));
# endif

    if (sqr_dist == 0) {
      nr_valid_proj++;
      continue;
    }

    int32_t residual = lut_residual[sqr_dist];  // 4+16 bits
    int32_t invres = lut_invres[sqr_dist];  // 7+9 bits
    // grad: 9+7 bits
    int16_t Iu_i = (lut_iu[*(pcurdist-1)] - lut_iu[*(pcurdist+1)]);
    int16_t Iv_i = (lut_iv[*(pcurdist-width)] - lut_iv[*(pcurdist+width)]);
    int16_t z_inv = ((inv_res2>>2) * zinv) >> 14; // 15-2+12 - 14=11

# ifdef DEBUG_OPTIMIZATION
    {
      float Iu = Iu_i / 128.0, Iv = Iv_i / 128.0;
      float x_z = x_z_i / 4096.0, y_z = y_z_i / 4096.0;
      float z_invf = z_inv / 2048.0;
      float Iu_x_z_Iv_y_z = Iu * x_z + Iv * y_z;
      Vector6 J;
      J[0] = Iu * z_invf;
      J[1] = Iv * z_invf;
      J[2] =  -Iu_x_z_Iv_y_z * z_invf;
      J[3] = -(Iu_x_z_Iv_y_z * y_z + Iv);
      J[4] =  (Iu_x_z_Iv_y_z * x_z + Iu);
      J[5] = -Iu * y_z + Iv * x_z;

      float rf = residual / 65536.0;
      H_1 += J*J.transpose() / rf;
      b_1 += J;
    }
#endif

    // Optimized Jacobian calculation. Note that the Huber, fx and fy have
    // been multiplied in Iu & Iv before.
    int16_t J[6];
    int16_t Iu_x_z_Iv_y_z = (Iu_i * x_z_i + Iv_i * y_z_i) >> 12;  // 7+12-12=7

    // First, retain Q9.2 for vector b.
    J[0] = ((int32_t)Iu_i * z_inv) >> 16;  // 7+11 - 16 = 2
    J[1] = ((int32_t)Iv_i * z_inv) >> 16;
    J[2] = (-(int32_t)Iu_x_z_Iv_y_z * z_inv) >> 16;
    J[3] = (-(Iu_x_z_Iv_y_z * y_z_i + ((int32_t)Iv_i << 12))) >> 17; // 7+12 - 17 = 2
    J[4] =   (Iu_x_z_Iv_y_z * x_z_i + ((int32_t)Iu_i << 12)) >> 17;
    J[5] = (-Iu_i * y_z_i + Iv_i * x_z_i) >> 17;
    b0 += J[0], b1 += J[1], b2 += J[2],
    b3 += J[3], b4 += J[4], b5 += J[5];

    // Next, retian Q9.1 J/residual
    int16_t Jinvres[6];
    Jinvres[0] = (J[0] * invres) >> 10, Jinvres[1] = (J[1] * invres) >> 10,
    Jinvres[2] = (J[2] * invres) >> 10, Jinvres[3] = (J[3] * invres) >> 10,
    Jinvres[4] = (J[4] * invres) >> 10, Jinvres[5] = (J[5] * invres) >> 10;

    // Finally, obtain Q29.3 Hessian parts: J*J^T / residual
    H00 += (J[0] * Jinvres[0]);
    H01 += (J[0] * Jinvres[1]);
    H02 += (J[0] * Jinvres[2]);
    H03 += (J[0] * Jinvres[3]);
    H04 += (J[0] * Jinvres[4]);
    H05 += (J[0] * Jinvres[5]);
    H11 += (J[1] * Jinvres[1]);
    H12 += (J[1] * Jinvres[2]);
    H13 += (J[1] * Jinvres[3]);
    H14 += (J[1] * Jinvres[4]);
    H15 += (J[1] * Jinvres[5]);
    H22 += (J[2] * Jinvres[2]);
    H23 += (J[2] * Jinvres[3]);
    H24 += (J[2] * Jinvres[4]);
    H25 += (J[2] * Jinvres[5]);
    H33 += (J[3] * Jinvres[3]);
    H34 += (J[3] * Jinvres[4]);
    H35 += (J[3] * Jinvres[5]);
    H44 += (J[4] * Jinvres[4]);
    H45 += (J[4] * Jinvres[5]);
    H55 += (J[5] * Jinvres[5]);

    total_res += residual;
    nr_valid_proj++;
  }
# ifdef DEBUG_OPTIMIZATION
  H_1 /= nr_valid_proj;
  b_1 /= nr_valid_proj;
  std::cout << "--floating point:\n  H=\n  " 
    << H_1.row(0) << "\n  " << H_1.row(1)  << "\n  " << H_1.row(2) << "\n  "
    << H_1.row(3) << "\n  " << H_1.row(4)  << "\n  " << H_1.row(5)
    << "\n  b= " << b_1.transpose() << std::endl;
  // printf("b: %08x %08x %08x %08x %08x %08x\n", b0, b1, b2, b3, b4, b5);
#endif

  // recover floating point Hessian & b from uint32_t
  float dividor = 1.0 / 8.0 / nr_valid_proj;
  H(0,0) = H00 * dividor;
  H(0,1) = H(1,0) = H01 * dividor;
  H(0,2) = H(2,0) = H02 * dividor;
  H(0,3) = H(3,0) = H03 * dividor;
  H(0,4) = H(4,0) = H04 * dividor;
  H(0,5) = H(5,0) = H05 * dividor;
  H(1,1) = H11 * dividor;
  H(1,2) = H(2,1) = H12 * dividor;
  H(1,3) = H(3,1) = H13 * dividor;
  H(1,4) = H(4,1) = H14 * dividor;
  H(1,5) = H(5,1) = H15 * dividor;
  H(2,2) = H22 * dividor;
  H(2,3) = H(3,2) = H23 * dividor;
  H(2,4) = H(4,2) = H24 * dividor;
  H(2,5) = H(5,2) = H25 * dividor;
  H(3,3) = H33 * dividor;
  H(3,4) = H(4,3) = H34 * dividor;
  H(3,5) = H(5,3) = H35 * dividor;
  H(4,4) = H44 * dividor;
  H(4,5) = H(5,4) = H45 * dividor;
  H(5,5) = H55 * dividor;
  dividor = 1.0 / 4.0 / nr_valid_proj;
  b(0) = b0 * dividor;
  b(1) = b1 * dividor;
  b(2) = b2 * dividor;
  b(3) = b3 * dividor;
  b(4) = b4 * dividor;
  b(5) = b5 * dividor;

# ifdef DEBUG_OPTIMIZATION
  std::cout << "--fixed point:\n  H=\n  " 
    << H.row(0) << "\n  " << H.row(1)  << "\n  " << H.row(2) << "\n  "
    << H.row(3) << "\n  " << H.row(4)  << "\n  " << H.row(5)
    << "\n  b= " << b.transpose() << std::endl;

  Matrix6x6 H_2 = H_1 - H;
  Vector6 b_2 = b_1 - b;
  std::cout << "---difference:\n  H=\n  " 
    << H_2.row(0) << "\n  " << H_2.row(1)  << "\n  " << H_2.row(2) << "\n  "
    << H_2.row(3) << "\n  " << H_2.row(4)  << "\n  " << H_2.row(5)
    << "\n  b= " << b_2.transpose() << std::endl;
  std::cout << nr_valid_proj << " projections, sum residual " 
    << total_res / 65536.0 << std::endl;

  cv::imshow("proj res", proj_ref);
  cv::waitKey();
  // getchar();
# endif

  nr_proj = nr_valid_proj;

  return total_res / 65536.0 / nr_valid_proj;
}
