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

#include "solver_gd.h"
#include "solver_lm.h"
#include "solver_lm_w_lut.h"
#include "solver_lm_nn.h"
#include "solver_lm_fixpoint.h"
#include "solver_lm_ic.h"

#include "picoedge.h"
#include "distance_transform.h"
#include "distance_transform_nn.h"

#include <libgen.h>
#include <string.h>

using namespace picovo;

void tracker_rgbd::_initialize(picovo_config &config)
{
  fx = config.fx;
  fy = config.fy;
  cx = config.cx;
  cy = config.cy;
  edge_threshold = config.edge_threshold;
  depth_max = config.depth_max;
  depth_min = config.depth_min;
  width = config.width;
  height = config.height;
  huber_weight = config.solver_huber_weight;
  _outlier_threshold = config.solver_outlier_threshold;
  is_use_canny = config.is_use_canny;
  if (is_use_canny) {
    std::cout << "Use basic Canny edges." << std::endl;
  } else {
    std::cout << "Use PicoEdge." << std::endl;
  }
  is_show_imgs = config.is_show_imgs;
  if (is_show_imgs) {
    std::cout << "show images" << std::endl;
  }
  is_view_animation = config.is_view_animation;

  switch (config.tracker_type) {
    case TRACKER_RGBD_DATASET: {
      if (config.dataset_folder.size() == 0) {
        std::cerr << "Invalid dataset..." << std::endl;
        exit(-1);
      }
      break;
    }
    default: {
      std::cerr << "Unrecognize tracker type: " << config.tracker_type << std::endl;
      exit(-1);
    }
  }

  solver_type = (solver_enum)config.solver_type;
  switch (solver_type) {
    case SOLVER_GD_PRE_GRAD:
    case SOLVER_GD_PRE_JACOBIAN:
      solver = new solver_gd(config);
      break;
    case SOLVER_LM_BASE:
      solver = new solver_lm_w_lut(config);
      break;
    case SOLVER_LM_PRE_GRAD:
    case SOLVER_LM_PRE_JACOBIAN:
      solver = new solver_lm(config);
      break;
    case SOLVER_LM_NN:
      solver = new solver_lm_nn(config);
      break;
    case SOLVER_LM_IC:
      solver = new solver_lm_ic(config);
      for (int x = 0; x < 16; x++) {
        for (int y = 0; y < 16; y++) {
          int lutindex = (x << 4) | y;
          lut_nn_ic[lutindex] = sqrtf(x*x + y*y);
        }
      }
      break;
    case SOLVER_LM_FIX_POINT:
    default:
      solver = new solver_lm_fixpoint(config);
      break;
  }

  init_dt();
  init_dt_nn();
  init_picoedge();

#ifdef USE_PANGOLIN
  my_viewer = std::shared_ptr<viewer>(new viewer(config));
#endif
}

void get_bitmap(uint8_t *input, uint8_t *output, int width, int height)
{
  uint8_t *pin = input, *pout = output;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x += 8, pin += 8, pout++) {
      uint8_t bits = (pin[0] & 0x1) | (pin[1] & 0x2) | (pin[2] & 0x4) | (pin[3] & 0x8) |
        (pin[4] & 0x10) | (pin[5] & 0x20) | (pin[6] & 0x40) | (pin[7] & 0x80);
      pout[0] = bits;
    }
  }
}

void tracker_rgbd::_add_curr_frame(double timestamp, cv::Mat img_gray, cv::Mat img_depth)
{
  // basic settings
  std::shared_ptr<frame> new_frame(new frame());
  // TODO: resize image
  new_frame->timestamp = timestamp;
  new_frame->img_gray = img_gray;
  new_frame->img_depth = img_depth;
  new_frame->pose = Eigen::Matrix4f::Identity();

  // feature detection
  new_frame->dt_bitmap = (uint8_t*)malloc(width*height/8);
  // only for imshow() debug
  cv::Mat &img_canny = new_frame->img_canny;
  if (is_use_canny) {
    cv::Canny(img_gray, img_canny, 150, 80, 3, false);
    get_bitmap(img_canny.ptr<uint8_t>(), new_frame->dt_bitmap, width, height);
  } else {
    picoedge(img_gray.ptr<uint8_t>(), new_frame->dt_bitmap, width, height, edge_threshold);
    img_canny.create(cv::Size(width,height), CV_8U);
    retain_edge_cv8u(new_frame->dt_bitmap, img_canny.ptr<uint8_t>(), width, height);
  }
  if (is_show_imgs) {
    cv::imshow("edge", 255-img_canny);
  }

  // select feature points
  int nrplt = cv::countNonZero(img_canny);
  if (solver_type == SOLVER_LM_FIX_POINT) {
    // use compressed feature points
    new_frame->feat = (struct compressed_feature*)malloc(sizeof(struct compressed_feature)*nrplt);
  } else {
    // use 3D feature points
    new_frame->pcld_buf = (struct coo_3d*)malloc(sizeof(struct coo_3d) * nrplt);
  }
  double invfx = 1.0 / fx, invfy = 1.0 / fy;
  int32_t invfx_i = invfx * 0x10000, invfy_i = invfy * 0x10000;
  int32_t cx_i = cx * 0x1000, cy_i = cy * 0x1000;
  int nr_chosen = 0;
  // start from line 1
  uint32_t *pedge = (uint32_t*)(new_frame->dt_bitmap + width/8);
  for (int y = 1; y < height-1; y++) {
    uint16_t *drow = img_depth.ptr<uint16_t>(y);
    for (int x = 0; x < width; x += 32, pedge++) {
      uint32_t e32 = *pedge;
      while (e32 != 0) {
        int offset = __builtin_ctz(e32);
        e32 &= ~(1 << offset);
        int cur_x = x + offset;
        uint16_t depth = drow[cur_x];
        if (depth > depth_min && depth < depth_max) {
          if (solver_type == SOLVER_LM_FIX_POINT) {
            new_frame->feat[nr_chosen].u_cx_fx = (((cur_x << 12) - cx_i) * invfx_i) >> 16;  // Q4.12
            new_frame->feat[nr_chosen].v_cy_fy = (((y << 12) - cy_i) * invfy_i) >> 16;  // Q4.12
            new_frame->feat[nr_chosen].zinv = (5000 << 12) / depth;  // Q4.12
          } else {
            float z = depth / 5000.0;
            // pixel to camera
            new_frame->pcld_buf[nr_chosen].x = (cur_x - cx) * z / fx;
            new_frame->pcld_buf[nr_chosen].y = (y - cy) * z / fy;
            new_frame->pcld_buf[nr_chosen].z = z;
          }
          nr_chosen++;
        }
      }
    }
  }
  new_frame->nr_pcld = nr_chosen;
  curr_frame = new_frame;
  // printf("Constructed frame %15lf, %d points\n", timestamp, nr_chosen);
}

void tracker_rgbd::_setup_key_frame(void)
{
  cv::Mat &img_dt = key_frame->img_dt;

  switch (solver_type) {
    case SOLVER_GD_PRE_GRAD:
    case SOLVER_LM_PRE_GRAD: {
      // pre-calculate DT gradient
      cv::distanceTransform(255 - key_frame->img_canny, img_dt, cv::DIST_L2, 
          cv::DIST_MASK_PRECISE);
      auto dt_buf = (struct dt_grad_f*)malloc(sizeof(struct dt_grad_f) * width*height);
      assert(dt_buf);
      key_frame->dt_buf = dt_buf;

      for (int y = 1; y < height-1; y++) {
        float *pdtin = img_dt.ptr<float>(y);
        struct dt_grad_f *pdtout = &dt_buf[y*width];
        for (int x = 1; x < width-1; x++) {
          // Calculate huber weighted pixel gradient and residual.
          float residual = pdtin[x];
          float huber_res = residual <= huber_weight ? residual : huber_weight;

          // Pre-multiply fx, fy and huber. See also solver::_calc_update_*().
          pdtout[x].Iu = 0.5 * fx * huber_res * (pdtin[x-1] - pdtin[x+1]);
          pdtout[x].Iv = 0.5 * fy * huber_res * (pdtin[x-width] - pdtin[x+width]);
          pdtout[x].residual = huber_res * residual;
        }
      }
      break;
    }

    case SOLVER_GD_PRE_JACOBIAN:
    case SOLVER_LM_PRE_JACOBIAN: {
      // pre-calculate Jacobian matrix
      cv::distanceTransform(255 - key_frame->img_canny, img_dt, cv::DIST_L2, 
          cv::DIST_MASK_PRECISE);
      int imgsize = width*height;
      int max_buf_pixs = imgsize * 2/3;

      key_frame->dtbuf_indexes = (int32_t*)malloc(sizeof(int32_t) * imgsize);
      key_frame->dt_buf_uv = (struct dt_grad_uv*)malloc(sizeof(struct dt_grad_uv)*max_buf_pixs);
      assert(key_frame->dtbuf_indexes);
      assert(key_frame->dt_buf_uv);
      memset(key_frame->dtbuf_indexes, 0xff, sizeof(int32_t) * imgsize);

      int bufindex = 0;
      for (int y = 1; y < height-1 && bufindex < max_buf_pixs; y++) {
        float *pdtin = key_frame->img_dt.ptr<float>(y);
        int32_t *pdt_index = key_frame->dtbuf_indexes + y*width;
        for (int x = 1; x < width-1 && bufindex < max_buf_pixs; x++) {
          float residual = pdtin[x];
          if (residual > _outlier_threshold) {
            continue;
          } else if (residual < 0.999) {
            pdt_index[x] = 0;
            continue;
          }

          register float huber_res = residual <= huber_weight ? residual : huber_weight;
          register float Iu = 0.5 * huber_res * (pdtin[x-1] - pdtin[x+1]);
          register float Iv = 0.5 * huber_res * (pdtin[x-width] - pdtin[x+width]);
          register float u_cx = x - cx, v_cy = y - cy;
          register float iu_ucx_iv_vcy = Iu * u_cx + Iv * v_cy;

          // Pre-calculate Jacobian matrix. See also solver::_calc_update_*_uv().
          struct dt_grad_uv *pout = &key_frame->dt_buf_uv[bufindex];
          pout->b0 = fx * Iu;
          pout->b1 = fy * Iv;
          pout->b2 = -iu_ucx_iv_vcy;
          pout->b3 = -(iu_ucx_iv_vcy * v_cy / fy + Iv*fy);
          pout->b4 =  (iu_ucx_iv_vcy * u_cx / fx + Iu*fx);
          pout->b5 = -Iu*v_cy * fx/fy + Iv*u_cx * fy/fx;
          pout->residual = huber_res * residual;

          pdt_index[x] = bufindex;
          bufindex++;
        }
      }
      break;
    }

    case SOLVER_LM_NN: {
      // calculate nearest neighbor field. See also solver_lm_nn::_calc_update_lm().
      key_frame->dt_to_grad = (uint8_t*)malloc(width*height);
      key_frame->dt_to_grad_sign = (uint8_t*)malloc(width*height/4);
      assert(key_frame->dt_to_grad);
      assert(key_frame->dt_to_grad_sign);
      memset(key_frame->dt_to_grad_sign, 0, width*height/4);
      distance_transform_nn(key_frame->img_canny.ptr<uint8_t>(),
        width, height, key_frame->dt_to_grad, key_frame->dt_to_grad_sign);
      break;
    }

    case SOLVER_LM_IC: {
      // inverse compositional: precalculate Jacobian and Hessian.

      // First obtain the nearest neighbor field.
      key_frame->dt_to_grad = (uint8_t*)malloc(width*height);
      key_frame->dt_to_grad_sign = (uint8_t*)malloc(width*height/4);
      assert(key_frame->dt_to_grad);
      assert(key_frame->dt_to_grad_sign);
      memset(key_frame->dt_to_grad_sign, 0, width*height/4);
      distance_transform_nn(key_frame->img_canny.ptr<uint8_t>(),
        width, height, key_frame->dt_to_grad, key_frame->dt_to_grad_sign);

      // Next pre-calulate the Jacobian and Hessian.
      int imgsize = width*height;
      int max_buf_pixs = imgsize * 2/3;
      key_frame->dtbuf_indexes = (int32_t*)malloc(sizeof(int32_t) * imgsize);
      key_frame->dt_buf_ic = (struct dt_grad_ic*)malloc(sizeof(struct dt_grad_ic)*max_buf_pixs);
      assert(key_frame->dtbuf_indexes);
      assert(key_frame->dt_buf_ic);
      memset(key_frame->dtbuf_indexes, 0xff, sizeof(int32_t) * imgsize);

      int bufindex = 0;
      for (int y = 1; y < height-1 && bufindex < max_buf_pixs; y++) {
        uint8_t *pdtnn = key_frame->dt_to_grad + y*width;
        uint8_t *pdtnn_sign = key_frame->dt_to_grad_sign + y*width/4;
        int32_t *pdt_index = key_frame->dtbuf_indexes + y*width;
        for (int x = 1; x < width-1 && bufindex < max_buf_pixs; x++) {
          uint lutindex = pdtnn[x];
          if (lutindex == 0xff) {
            continue;
          } else if (lutindex == 0) {
            pdt_index[x] = 0;
            continue;
          }

          // find the nearest neighbor depth
          uint8_t sign = pdtnn_sign[x>>2] >> ((x&0x3) << 1);
          int nn_x = sign & 0x2 ? x - (lutindex >> 4) : x + (lutindex >> 4);
          int nn_y = sign & 0x1 ? y - (lutindex & 0xf) : y + (lutindex & 0xf);
          uint16_t depth = key_frame->img_depth.ptr<uint16_t>(nn_y)[nn_x];
          if (depth > depth_min && depth < depth_max) {
            float zinv = 5000.0 / depth;
            float residual = lut_nn_ic[lutindex];
            float huber_res = residual <= huber_weight ? residual : huber_weight;
            float Iu = 0.5 * huber_res * (lut_nn_ic[pdtnn[x-1]] - lut_nn_ic[pdtnn[x+1]]);
            float Iv = 0.5 * huber_res * (lut_nn_ic[pdtnn[x-width]] - lut_nn_ic[pdtnn[x+width]]);
            float u_cx = x - cx, v_cy = y - cy;
            float iu_ucx_iv_vcy = Iu * u_cx + Iv * v_cy;

            // Pre-calculate Jacobian matrix. See also solver::_calc_update_*_uv().
            Vector6 J;
            J[0] = fx * Iu * zinv;
            J[1] = fy * Iv * zinv;
            J[2] = -iu_ucx_iv_vcy * zinv;
            J[3] = -(iu_ucx_iv_vcy * v_cy / fy + Iv*fy);
            J[4] =  (iu_ucx_iv_vcy * u_cx / fx + Iu*fx);
            J[5] = -Iu*v_cy * fx/fy + Iv*u_cx * fy/fx;
            Matrix6x6 H = J*J.transpose() / (residual * huber_res);

            struct dt_grad_ic *pout = &key_frame->dt_buf_ic[bufindex];
            pout->H[0] = H(0,0);
            pout->H[1] = H(0,1);
            pout->H[2] = H(0,2);
            pout->H[3] = H(0,3);
            pout->H[4] = H(0,4);
            pout->H[5] = H(0,5);
            pout->H[6] = H(1,1);
            pout->H[7] = H(1,2);
            pout->H[8] = H(1,3);
            pout->H[9] = H(1,4);
            pout->H[10] = H(1,5);
            pout->H[11] = H(2,2);
            pout->H[12] = H(2,3);
            pout->H[13] = H(2,4);
            pout->H[14] = H(2,5);
            pout->H[15] = H(3,3);
            pout->H[16] = H(3,4);
            pout->H[17] = H(3,5);
            pout->H[18] = H(4,4);
            pout->H[19] = H(4,5);
            pout->H[20] = H(5,5);
            pout->b[0] = J(0);
            pout->b[1] = J(1);
            pout->b[2] = J(2);
            pout->b[3] = J(3);
            pout->b[4] = J(4);
            pout->b[5] = J(5);
            pout->residual = residual;

            pdt_index[x] = bufindex;
            bufindex++;
          }
        }
      }
      break;
    }

    case SOLVER_LM_BASE:
    case SOLVER_LM_FIX_POINT:
    default: {
      // calculate uint8_t square distance transform. See also solver_lm_w_lut
      // and solver_lm_fixpoint
      key_frame->dt_to_grad = (uint8_t*)malloc(width*height);
      assert(key_frame->dt_to_grad);
      distance_transform(key_frame->dt_bitmap, key_frame->dt_to_grad, width, height);
      break;
    }
  }

#ifdef USE_PANGOLIN
  my_viewer->add_key_frame(key_frame);
#endif
}

void tracker_rgbd::start(picovo_config &config)
{
  _initialize(config);

  std::ofstream summary_file("summary.csv");
  if (!summary_file.is_open()) {
    std::cerr << "Failed to open summary.csv" << std::endl;
    exit(-1);
  }
  summary_file << "dataset, preprocess(us), track(us), total(us), nr_features, residual, iterations" << std::endl;

  if (config.tracker_type == TRACKER_RGBD_DATASET) {
    // Input with dataset
    for (auto ds : config.dataset_folder) {
      std::cout << "Tracking " << ds << "..." << std::endl;
      std::string dataset_dir = ds;
      dataset_dir += "/";
      std::string a_f = dataset_dir + "associate.txt";
      std::ifstream assoc_file(a_f);
      if (!assoc_file) {
        std::cerr << "Failed to open " << a_f << std::endl;
        exit(-1);
      }

      std::string out_bn(basename((char*)ds.c_str()));
      std::string outposename("poses-" + out_bn + ".txt");
      std::string outstatname("stat-" + out_bn + ".csv");
      std::cout << "output: " << outposename << ", " << outstatname << std::endl;
      std::ofstream pose_file(outposename), stat_file(outstatname);
      if (!pose_file.is_open() || !stat_file.is_open()) {
        std::cerr << "Failed to create output file." << std::endl;
        exit(-1);
      }

      bool is_first = true;
      double starttime = 0;
      curr_state = TRACK_IDLE;
#ifdef USE_PANGOLIN
      my_viewer->clear_all();
#endif

      solver->reset();
      double nr_frames = 0;
      struct ovo_stat summary_stat = { 0 };

      while (!assoc_file.eof()) {
        std::string rgbid, rgbfile, depthid, depthfile;
        assoc_file >> rgbid >> rgbfile >> depthid >> depthfile;
        if (rgbfile.length() < 1) {
          continue;
        }
        double curtime = std::stod(rgbid);
        if (is_first) {
          starttime = curtime;
          is_first = false;
        }
        // std::cout << rgbid << " ";
        cv::Mat rgb = cv::imread(dataset_dir + rgbfile);
        cv::Mat gray;// = cv::imread(dataset_dir + rgbfile, cv::IMREAD_GRAYSCALE);
        cv::cvtColor(rgb, gray, cv::COLOR_BGRA2GRAY);
        cv::Mat d = cv::imread(dataset_dir + depthfile, cv::IMREAD_UNCHANGED);

        if (gray.cols != width || gray.rows != height) {
          cv::Size newsize = cv::Size(width,height);
          cv::resize(rgb, rgb, newsize);
          cv::resize(gray, gray, newsize);
          cv::resize(d, d, newsize, 0, 0, cv::INTER_NEAREST);
        }

        auto start_time = std::chrono::high_resolution_clock::now();
        _add_curr_frame(curtime, gray, d);
        curr_frame->img_rgb = rgb.clone();
        auto start_track_time = std::chrono::high_resolution_clock::now();
        _track_curr_frame();
        auto end_time2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> pre_process_us = start_track_time - start_time;
        std::chrono::duration<double> track_us = end_time2 - start_track_time;

        // output statistics
        struct ovo_stat stat;
        solver->get_statistics(stat);
        stat.nr_features = curr_frame->nr_pcld;
        stat.pre_process_us = pre_process_us.count();
        stat.track_us = track_us.count();
        summary_stat += stat;
        stat_file << stat.pre_process_us << ", " << stat.track_us << ", " << stat.nr_features << ", "
          << stat.residual << ", " << stat.nr_iterations << std::endl;

        curr_frame->track_time = stat.pre_process_us + stat.track_us;
        curr_frame->time_elapsed = curtime - starttime;
#ifdef USE_PANGOLIN
        my_viewer->show_curr_frame(curr_frame);
#endif

        // output pose results
        Eigen::Matrix4f &pose = curr_frame->pose;
        Eigen::Matrix3f R = pose.block<3,3>(0,0);
        Eigen::Vector3f T = pose.block<3,1>(0,3);
        const Eigen::Quaternionf Qf(R);
        pose_file << std::fixed << rgbid << " " << std::setprecision(9) 
          << T[0] << " " << T[1] << " " << T[2] << " " 
          <<  Qf.x() << " " << Qf.y() << " " << Qf.z() << " " << Qf.w() 
          << std::endl;

        if (is_show_imgs) {
          cv::putText(rgb, std::to_string(curtime - starttime), cv::Point(5,25),
            cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255), 2);
          cv::imshow("cur", rgb);
        } else {
          int cur_nr = nr_frames;
          printf("\r%4d, %4.4lf", cur_nr, curtime - starttime);
          fflush(stdout);
        }

        if (config.is_pause)
          cv::waitKey(0);
        else
          cv::waitKey(1);

        nr_frames += 1.0;
      }
      // summary_file
      std::stringstream str;
      str 
        << out_bn << ", "
        << summary_stat.pre_process_us / nr_frames << ", " 
        << summary_stat.track_us / nr_frames << ", " 
        << (summary_stat.pre_process_us+summary_stat.track_us) / nr_frames << ", " 
        << summary_stat.nr_features / nr_frames << ", "
        << summary_stat.residual / nr_frames << ", " 
        << summary_stat.nr_iterations / nr_frames;
      std::cout << "\n" << str.str() << std::endl;
      summary_file << str.str() << std::endl;
      std::cout << "Done" << std::endl;

#ifdef USE_PANGOLIN
      if (is_view_animation) {
        my_viewer->play_animation();
      }
#endif
    }
  } else {
    // TODO: input with physical RGBD camera
  }
  cv::destroyAllWindows();
#ifdef USE_PANGOLIN
  my_viewer->close();
#endif
}

void tracker_rgbd::_track_curr_frame(void)
{
  if (curr_state != TRACK_OKAY) {
    // The first frame
    std::cout << "Tracking start" << std::endl;
    pose_vs_keyframe = Eigen::Matrix4f::Identity();
    pose_vs_lastframe = Eigen::Matrix4f::Identity();
    key_frame = curr_frame;
    key_frame_candidate = nullptr;
    is_just_add_kf = true;
    _setup_key_frame();
    last_frame = curr_frame;
    nr_feat_thre = curr_frame->nr_pcld;
    if (is_show_imgs) {
      cv::imshow("key_frame", 255-key_frame->img_canny);
    }

    curr_state = TRACK_OKAY;
  } else {
    // Track a new frame
    // Initial pose: assume a uniform speed model
    Eigen::Matrix4f new_pose_vs_keyframe = pose_vs_keyframe * pose_vs_lastframe;
    Eigen::Matrix3f new_R2kf;
    Eigen::Vector3f new_T2kf;
    extract_R_T(new_pose_vs_keyframe, new_R2kf, new_T2kf);

    float err = solver->track_frame(key_frame, curr_frame, new_R2kf, new_T2kf);

    if (err > 9) {
      // error too large. assume tracking lost
      std::cout << "Tracking lost... Restart..." << std::endl;
      curr_frame->pose = last_frame->pose;
      pose_vs_keyframe = Eigen::Matrix4f::Identity();
      pose_vs_lastframe = Eigen::Matrix4f::Identity();
      key_frame = curr_frame;
      key_frame_candidate = nullptr;
      is_just_add_kf = true;
      _setup_key_frame();
      last_frame = curr_frame;
      nr_feat_thre = curr_frame->nr_pcld;
    } else {
      Sophus::SE3f delta(makeup_matrix4f(new_R2kf, new_T2kf));
      float se3norm = delta.log().norm();
      // std::cout << "-- err: " << err << "," << se3norm << std::endl;

      if (key_frame_candidate &&
        (err > 1.1 || curr_frame->nr_tracked < curr_frame->nr_pcld * 0.66 || se3norm > 0.25)
      ) {
        // std::cout << " => Insert a key_frame!" << std::endl;
        // Eigen::Matrix4f new_pose = 
        //   key_frame->pose * makeup_matrix4f(new_R2kf, new_T2kf);

        key_frame = key_frame_candidate;
        _setup_key_frame();

        // new_pose_vs_keyframe = key_frame->pose.inverse() * new_pose;
        new_pose_vs_keyframe = Eigen::Matrix4f::Identity();
        extract_R_T(new_pose_vs_keyframe, new_R2kf, new_T2kf);
        if (is_show_imgs) {
          cv::imshow("key_frame", 255-key_frame->img_canny);
        }

        // retrack this frame
        err = solver->track_frame(key_frame, curr_frame, new_R2kf, new_T2kf);

        is_just_add_kf = true;
      } else {
        is_just_add_kf = false;
      }

      auto tmppose_vs_keyframe = makeup_matrix4f(new_R2kf, new_T2kf);
      curr_frame->pose = key_frame->pose * tmppose_vs_keyframe;
      if (err < 1.1) {
        pose_vs_keyframe = tmppose_vs_keyframe;
        pose_vs_lastframe = last_frame->pose.inverse() * curr_frame->pose;
      } else {
        pose_vs_keyframe = pose_vs_keyframe * pose_vs_lastframe;
      }

      // Apply a low-pass filter to frame::nr_pcld
      const float alpha = 0.7;
      nr_feat_thre = alpha * nr_feat_thre + (1-alpha) * curr_frame->nr_pcld;
      // If the new frame has enough features, it can be the key_frame candidate.
      const float beta = 0.7;
      if (curr_frame->nr_pcld > beta * nr_feat_thre) {
        key_frame_candidate = curr_frame;
      }

      last_frame = curr_frame;
    }
  }
}
