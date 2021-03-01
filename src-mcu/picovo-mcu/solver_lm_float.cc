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
#include "sophus/se3.hpp"

// private:

// Look-Up-Table for residual: huber*sqrt(uint8_t)
static float lut_residual[256] __attribute__((section(".itcm_data")));
// LUT for 1/(huber*sqrt(uint8_t))
static float lut_invres[256] __attribute__((section(".itcm_data")));
// LUT for gradients: 0.5*f*huber*sqrt(uint8_t)
static float lut_iu[256] __attribute__((section(".itcm_data")));
static float lut_iv[256] __attribute__((section(".itcm_data")));
// the LM lamda
static float lamda;

int stat_iterations;

/**
 * @fn _calc_update_lm_fp
 * @brief Calculate Hessian & b using fixed point arithmetic.
 * @param ref_dt DT of the reference frame.
 * @param cur_feat Selected features of the current frame.
 * @param nr_proj Input: Number of points in cur_pcl.
 *                Output: Number of valid projections.
 * @param R Input: Current rotation matrix.
 * @param T Input: Current translation vector.
 * @param H Output: Hessian matrix, Hx=b
 * @param b Output: steppest descent, Hx=b
 * @return Sum of residual.
 */
static float _calc_update_lm_fp(
  const uint8_t *ref_dt, 
  const struct feature_s *pfeat, 
  int &nr_proj,
  int step,
  const Eigen::Matrix3f &R, 
  const Eigen::Vector3f &T, 
  Matrix6x6 &H,
  Vector6 &b
)
{
  float total_res = 0;
  int nr_valid_proj = 0;

  /* Parts of the Hessian matrix, 32+0 bits */
  float H00 = 0, H01 = 0, H02 = 0, H03 = 0, H04 = 0, H05 = 0,
                 H11 = 0, H12 = 0, H13 = 0, H14 = 0, H15 = 0,
                          H22 = 0, H23 = 0, H24 = 0, H25 = 0,
                                   H33 = 0, H34 = 0, H35 = 0,
                                            H44 = 0, H45 = 0,
                                                     H55 = 0;
  /* Parts of the steppest descent, 21+11 bits */
  float b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0;

  // Q15 convertion for R & T. Rij & Ti should all be smaller than 1.
  const float ri00 = R(0,0);
  const float ri01 = R(0,1);
  const float ri02 = R(0,2);
  const float ri10 = R(1,0);
  const float ri11 = R(1,1);
  const float ri12 = R(1,2);
  const float ri20 = R(2,0);
  const float ri21 = R(2,1);
  const float ri22 = R(2,2);
  const float ti0 = T(0);
  const float ti1 = T(1);
  const float ti2 = T(2);

  for (int i = 0; i < nr_proj; i += step) {
    auto feature = pfeat[i];
    float ucxfx = feature.u_cx_fx / 4096.0; // Q4.12
    float vcyfy = feature.v_cy_fy / 4096.0; // Q4.12
    float zinv =  feature.zinv / 4096.0;    // Q4.12
    float res0 = ri00 * ucxfx + ri01 * vcyfy + ri02 + ti0 * zinv;
    float res1 = ri10 * ucxfx + ri11 * vcyfy + ri12 + ti1 * zinv;
    float res2 = ri20 * ucxfx + ri21 * vcyfy + ri22 + ti2 * zinv;
    float inv_res2 = 1.0 / res2;
    float x_z = res0 * inv_res2, y_z = res1 * inv_res2;
    float u_f = CAM_FX * x_z + CAM_CX;
    float v_f = CAM_FY * y_z + CAM_CY;
    int32_t u = u_f, v = v_f;

    if (u <= 1 || v <= 1 || u >= IMG_WIDTH-2 || v >= IMG_HEIGHT-2) {
      // out of image boundary
      continue;
    }
    const uint pixindex = u + v*IMG_WIDTH;
    const uint8_t *pcurdist = ref_dt + pixindex;
    const uint sqr_dist = *pcurdist;
    if (sqr_dist >= 0xe1) {
      // Skip this outlier
      continue;
    }

    if (sqr_dist == 0) {
      nr_valid_proj++;
      continue;
    }

    float residual = lut_residual[sqr_dist];
    float invres = lut_invres[sqr_dist];
    // grad
    float Iu = (lut_iu[*(pcurdist-1)] - lut_iu[*(pcurdist+1)]);
    float Iv = (lut_iv[*(pcurdist-IMG_WIDTH)] - lut_iv[*(pcurdist+IMG_WIDTH)]);
    float z_inv = inv_res2 * zinv;

    // Optimized Jacobian calculation. Note that the Huber, fx and fy have
    // been multiplied in Iu & Iv before.
    float J[6];
    float Iu_x_z_Iv_y_z = (Iu * x_z + Iv * y_z);

    // First, retain vector b.
    J[0] = Iu * z_inv;
    J[1] = Iv * z_inv;
    J[2] = -Iu_x_z_Iv_y_z * z_inv;
    J[3] = -(Iu_x_z_Iv_y_z * y_z + Iv);
    J[4] =  (Iu_x_z_Iv_y_z * x_z + Iu);
    J[5] = -Iu * y_z + Iv * x_z;
    b0 += J[0], b1 += J[1], b2 += J[2],
    b3 += J[3], b4 += J[4], b5 += J[5];

    // Next, retian J/residual
    float Jinvres[6];
    Jinvres[0] = J[0] * invres, Jinvres[1] = J[1] * invres,
    Jinvres[2] = J[2] * invres, Jinvres[3] = J[3] * invres,
    Jinvres[4] = J[4] * invres, Jinvres[5] = J[5] * invres;

    // Finally, obtain Hessian parts: J*J^T / residual
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

  // recover floating point Hessian & b from uint32_t
  H(0,0) = H00;
  H(0,1) = H(1,0) = H01;
  H(0,2) = H(2,0) = H02;
  H(0,3) = H(3,0) = H03;
  H(0,4) = H(4,0) = H04;
  H(0,5) = H(5,0) = H05;
  H(1,1) = H11;
  H(1,2) = H(2,1) = H12;
  H(1,3) = H(3,1) = H13;
  H(1,4) = H(4,1) = H14;
  H(1,5) = H(5,1) = H15;
  H(2,2) = H22;
  H(2,3) = H(3,2) = H23;
  H(2,4) = H(4,2) = H24;
  H(2,5) = H(5,2) = H25;
  H(3,3) = H33;
  H(3,4) = H(4,3) = H34;
  H(3,5) = H(5,3) = H35;
  H(4,4) = H44;
  H(4,5) = H(5,4) = H45;
  H(5,5) = H55;
  b(0) = b0;
  b(1) = b1;
  b(2) = b2;
  b(3) = b3;
  b(4) = b4;
  b(5) = b5;

  nr_proj = nr_valid_proj;

  return total_res / nr_valid_proj;
}

// public:

void init_solver(void)
{
  // prepare fixed-point LUT
  for (int i = 0; i < 256; i++) {
    double d = SOLVER_HUBER * sqrt(i);
    lut_residual[i] = d;
    lut_invres[i] = 1.0 / d;
    lut_iu[i] = 0.3 * CAM_FX * d;
    lut_iv[i] = 0.3 * CAM_FY * d;
  }

  lamda = 3.2;
  stat_iterations = 0;
}

float solver_track_frame(
  uint8_t *ref_dt,
  struct feature_s *cur_feat,
  int &nr_feat,
  Eigen::Matrix3f &R, 
  Eigen::Vector3f &T
)
{
  // projection from cur_frame to ref_dt
  // estimated SE3
  Sophus::SE3f ref2cur(R,T);
  Matrix6x6 H;
  Vector6 b;

  float last_min_resdl;
  int nr_tracked = nr_feat;
  last_min_resdl = _calc_update_lm_fp(ref_dt, cur_feat, 
    nr_tracked, 4, R, T, H, b);

  if (lamda > 3) lamda = 3.2;

  int its = 0;
  uint32_t res_inc_history = 0;
  for (; its < 100; its++) {
    float tmpadder = 1+lamda;
    for (int i = 0; i < 6; i++)
      H(i,i) += tmpadder;
    Vector6 inc;
    inc = H.ldlt().solve(b);

    Sophus::SE3f new_ref2cur = Sophus::SE3f::exp(inc) * ref2cur;
    float cur_resdl;

    int featoffset = its < 4 ? its : its < 6 ? its-4 : 0;
    int stepsize = its < 4 ? 4 : its < 6 ? 2 : 1;
    nr_tracked = nr_feat & 0xfffffffc;
    cur_resdl = _calc_update_lm_fp(ref_dt, cur_feat + featoffset, nr_tracked, stepsize,
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
      break;
    } else if (lamda > 5000 || inc.dot(inc) < 1e-16) {
      break;
    } else {
      res_inc_history = (res_inc_history << 1) | 0x1;
      // Set a more conservative step size.
      // If the residual is trembling, be even more conservative.
      lamda = lamda < 0.3 ? 0.4 : lamda * ((res_inc_history & 0x7) == 0x5 ? 4 : 2);
    }
    ref2cur = new_ref2cur;
  }

  // Return results
  R = ref2cur.rotationMatrix();
  T = ref2cur.translation();
  nr_feat = nr_tracked;
  stat_iterations = its;

  return last_min_resdl;
}
