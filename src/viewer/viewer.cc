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

#include "viewer.h"

#include <unistd.h>

using namespace picovo;

viewer::viewer(picovo_config &config)
{
  is_running = true;
  fx = config.fx;
  fy = config.fy;
  cx = config.cx;
  cy = config.cy;
  win_width = config.viewer_width;
  win_height = config.viewer_height;
  is_show_img = config.is_view_show_imgs;
  is_fullscreen = config.is_view_fullscreen;
  viewer_thread_ = std::thread(std::bind(&viewer::viewer_loop, this));
  is_capture_video = config.is_capture_video;
  if (is_capture_video) {
    capture_name = config.capture_video_name;
    capture_fps = config.capture_framerate;
    capture_thread_ = std::thread(std::bind(&viewer::capture_loop, this));
  }
  is_curr_frame_updated = false;
  is_keyframe_updated = false;
  is_start_animation = false;
  view_pose = Eigen::Matrix4f::Identity();
  alpha = 0.01;

  bar_mat.create(cv::Size(win_width, win_height/10), CV_8UC3);
  switch (config.solver_type) {
  case SOLVER_GD_PRE_GRAD:
    solver_name = "GD with DT gradient LUT"; break;
  case SOLVER_GD_PRE_JACOBIAN:
    solver_name = "GD with Jacobian LUT"; break;
  case SOLVER_LM_BASE:
    solver_name = "float-point basic LM"; break;
  case SOLVER_LM_PRE_GRAD:
    solver_name = "float-point LM with gradient LUT"; break;
  case SOLVER_LM_PRE_JACOBIAN:
    solver_name = "float-point LM with Jacobian LUT"; break;
  case SOLVER_LM_FIX_POINT:
    solver_name = "fixed-point LM"; break;
  case SOLVER_LM_NN:
    solver_name = "float-point LM using nearest neighbor"; break;
  case SOLVER_LM_IC:
    solver_name = "float-point LM inverse-compositional LUT"; break;
  }
  if (config.is_use_canny) {
    solver_name += ", Canny";
  } else {
    solver_name += ", PicoEdge";
  }
  char tmpbuf[100];
  snprintf(tmpbuf, sizeof(tmpbuf), ", %dx%d        on Desktop PC", 
    config.width, config.height);
  solver_name += tmpbuf;

  capture_img.create(cv::Size(win_width, win_height), CV_8UC3);
}

void viewer::close()
{
  std::unique_lock<std::mutex> lck(running_lock);
  viewer_thread_.join();
  if (is_capture_video) {
    capture_thread_.join();
  }
}

void viewer::update_curr_frame(std::shared_ptr<frame> current_frame)
{
  std::unique_lock<std::mutex> lck(current_lock);
  if (curr_frame == nullptr) {
    // this is the first frame
    animation_lock.lock();
    ani_kf_index = 0;
    alpha = 0.01;
  }
  curr_frame = current_frame;
  cv::resize(curr_frame->img_rgb, cur_rgb, cv::Size(640,480));
  char tmpbuf[256];
  snprintf(tmpbuf, sizeof(tmpbuf), "%.3lfs: track time %.3fms", 
    curr_frame->time_elapsed, curr_frame->track_time * 1000);
  cv::putText(cur_rgb, tmpbuf, cv::Point(5,25),
    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255), 2);
  is_curr_frame_updated = true;

  memset(bar_mat.data, 0, bar_mat.cols * bar_mat.rows * 3);
  cv::putText(bar_mat, solver_name, cv::Point(5,25),
    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255), 2);
  cv::putText(bar_mat, tmpbuf, cv::Point(5,60),
    cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255), 2);

  traj_estimate.push_back(curr_frame->pose);
}

void viewer::add_key_frame(std::shared_ptr<frame> kf)
{
  std::unique_lock<std::mutex> lck(kf_lock);
  active_keyframes_.push_back(kf);

  // // VBO buffer of float[8]: x,y,z,1,R,G,B,1
  // float *pclbuf = (float*)malloc(8 * kf->nr_pcld * sizeof(float));

  // Eigen::Matrix4f pose = kf->pose;
  // for (int i = 0; i < kf->nr_pcld; i++) {
  //   Eigen::Vector4f pcur;
  //   int u, v;
  //   if (kf->pcld_buf != NULL) {
  //     float x = kf->pcld_buf[i].x, y = kf->pcld_buf[i].y, z = kf->pcld_buf[i].z;
  //     pcur << x, y, z, 1;
  //     u = fx * x/z + cx;
  //     v = fy * y/z + cy;
  //   } else {
  //     float z = 4096.0 / kf->feat[i].zinv;
  //     float x = kf->feat[i].u_cx_fx * z / 4096.0, y = kf->feat[i].v_cy_fy * z / 4096.0;
  //     pcur << x, y, z, 1;
  //     u = fx * x/z + cx;
  //     v = fy * y/z + cy;
  //   }
  //   Eigen::Vector4f pworld = pose * pcur;
  //   auto color = kf->img_rgb.at<cv::Vec3b>(v, u);
  //   pclbuf[i*8 + 0] = pworld[0];
  //   pclbuf[i*8 + 1] = pworld[1];
  //   pclbuf[i*8 + 2] = pworld[2];
  //   pclbuf[i*8 + 3] = 1;
  //   pclbuf[i*8 + 4] = color[2];
  //   pclbuf[i*8 + 5] = color[1];
  //   pclbuf[i*8 + 6] = color[0];
  //   pclbuf[i*8 + 7] = 1;
  // }
  // glGenBuffers(1, &kf->vbo);
  // glBindBuffer(GL_ARRAY_BUFFER, kf->vbo);
  // glBufferData(GL_ARRAY_BUFFER, kf->nr_pcld * 8 * sizeof(float), pclbuf, GL_STATIC_DRAW);
  // glBindBuffer(GL_ARRAY_BUFFER, 0);

  // kf->vbobuffer = pclbuf;
  is_keyframe_updated = true;
}

void viewer::add_groundtruth(Eigen::Matrix4f gt_pose)
{
  std::unique_lock<std::mutex> lck(current_lock);
  traj_groundtruth.push_back(gt_pose);
}

void viewer::play_animation(void)
{
  // trigger animation
  std::cout << "Start animation..." << std::endl;
  is_start_animation = true;
  // wait for animation to stop
  std::unique_lock<std::mutex> lck_ani(animation_lock);
  std::cout << "Animation end..." << std::endl;
  is_start_animation = false;
  usleep(2000000);
}

void viewer::clear_all(void)
{
  std::unique_lock<std::mutex> lck(kf_lock);
  std::unique_lock<std::mutex> lck2(current_lock);
  // for (auto kf : active_keyframes_) {
  //   glDeleteBuffers(1, &kf->vbo);
  // }
  traj_estimate.clear();
  traj_groundtruth.clear();
  active_keyframes_.clear();
  curr_frame = nullptr;
  animation_lock.unlock();
}

void viewer::viewer_loop()
{
  std::unique_lock<std::mutex> lck(running_lock);

  pangolin::CreateWindowAndBind("PicoVO viewer", win_width, win_height);
  if (is_fullscreen) {
    pangolin::ToggleFullscreen();
  }
  glewInit();
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::OpenGlRenderState vis_camera(
    pangolin::ProjectionMatrix(win_width, win_height, 500, 500, win_width/2, win_height/2, 0.1, 1000),
    pangolin::ModelViewLookAt(0, -1.5, -1.5, 0, -1, 0, 0.0, -1.0, 0.0));

  // Add named OpenGL viewport to window and provide 3D Handler
  pangolin::View& vis_display = pangolin::CreateDisplay()
    .SetBounds(0.0, 1.0, 0.0, 1.0, -(float)win_width / (float)win_height)
    .SetHandler(new pangolin::Handler3D(vis_camera));

  pangolin::View& cv_img_1 = pangolin::Display("image_1")
    .SetBounds(0, 0.3, 0, 0.3, 640.0/480.0)
    .SetLock(pangolin::LockLeft, pangolin::LockBottom);
  pangolin::GlTexture cur_img(640, 480, GL_RGB, false, 0, GL_BGR, GL_UNSIGNED_BYTE);

  pangolin::View& cv_img_disp_bar = pangolin::Display("image_bar")
    .SetBounds(0, 0.1, 0, 1.0, (float)win_width / (float)win_height * 10.0)
    .SetLock(pangolin::LockLeft, pangolin::LockBottom);
  pangolin::GlTexture bar_img(win_width, win_height/10, GL_RGB, false, 0, GL_BGR, GL_UNSIGNED_BYTE);

  const float green[3] = {0, 1, 0};

  int ani_cnt = 0;

  while (!pangolin::ShouldQuit()) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(.0f, .0f, .0f, 1.0f);
    vis_display.Activate(vis_camera);

    if (curr_frame) {
      std::unique_lock<std::mutex> lock(current_lock);
      draw_camera(curr_frame, green);
      float color[3] = {0.2, 1.0, 0.2};
      draw_trajectory(traj_estimate, color);
      color[1] = 0.2, color[0] = 1.0;
      draw_trajectory(traj_groundtruth, color);
      if (!is_start_animation)
        follow_current_frame(vis_camera);
    }

    if (true) {
      draw_pcld();
    }

    if (is_start_animation) {
      if (ani_cnt > 2) {
        ani_cnt = 0;
        ani_kf_index++;
      } else {
        ani_cnt++;
      }
      int nr_kf = active_keyframes_.size();
      if (ani_kf_index >= nr_kf) {
        ani_kf_index = 0;
        animation_lock.unlock();
      } else {
        alpha = 0.01 + 0.02 * ani_kf_index / nr_kf;
        go_smoothly_to_pose(active_keyframes_[ani_kf_index]->pose, vis_camera);
      }
    }

    if (is_show_img && curr_frame != nullptr) {
      // show current image
      std::unique_lock<std::mutex> lck(current_lock);
      cur_img.Upload(cur_rgb.data, GL_BGR, GL_UNSIGNED_BYTE);
      cv_img_1.Activate();
      glColor3f(1.0,1.0,1.0);
      cur_img.RenderToViewportFlipY();
    } else {
      // show debug info
      std::unique_lock<std::mutex> lck(current_lock);
      bar_img.Upload(bar_mat.data, GL_BGR, GL_UNSIGNED_BYTE);
      cv_img_disp_bar.Activate();
      glColor3f(1,1,1);
      bar_img.RenderToViewportFlipY();
    }

    pangolin::FinishFrame();

    // capture pictures
    {
      cv::Mat tmp(capture_img.size(), CV_8UC3);
      glReadPixels(0, 0, win_width, win_height, GL_BGR, GL_UNSIGNED_BYTE, tmp.ptr());
      cv::flip(tmp, tmp, 0);
      tmp.copyTo(capture_img);
    }

    usleep(10000);
  }
  pangolin::QuitAll();

  // std::cout << "Stop viewer" << std::endl;
  is_running = false;
}

void viewer::capture_loop()
{
  std::cout << "Saving video to " << capture_name << "." << std::endl;

  cv::VideoWriter video_out;
  video_out.open(capture_name, cv::VideoWriter::fourcc('a','v','c','1'), 
    capture_fps, cv::Size(win_width, win_height), true);
  if (!video_out.isOpened()) {
    std::cerr << "Failed to open video output..." << std::endl;
    return;
  }
  usleep(100000);
  while (is_running) {
    // cv::imshow("capture", capture_img);
    video_out.write(capture_img);
    usleep(1000000/capture_fps);
  }
  std::cout << "catpure end" << std::endl;
}

void viewer::follow_current_frame(pangolin::OpenGlRenderState& vis_camera)
{
  go_smoothly_to_pose(curr_frame->pose, vis_camera);
}

void viewer::go_smoothly_to_pose(Eigen::Matrix4f pose, pangolin::OpenGlRenderState& vis_camera)
{
  Eigen::Matrix4f diff_pose = pose * view_pose.inverse();
  Vector6 diff_se3 = Sophus::SE3f(diff_pose).log();
  // double alpha = 0.01;
  view_pose = Sophus::SE3f::exp(alpha * diff_se3).matrix() * view_pose;
  // refine the pose to avoid non orthogonal rotation matrix
  view_pose = Sophus::SE3f::exp(Sophus::SE3f(view_pose).log()).matrix();
  pangolin::OpenGlMatrix m(view_pose);
  vis_camera.Follow(m, true);
}

void viewer::draw_camera(std::shared_ptr<frame> pframe, const float* color)
{
  const float sz = 0.1;
  const int line_width = 2.0;
  const float fx = 400;
  const float fy = 400;
  const float cx = 512;
  const float cy = 384;
  const float width = 1080;
  const float height = 768;

  glPushMatrix();

  Sophus::Matrix4f m = pframe->pose;
  glMultMatrixf((GLfloat*)m.data());

  if (color == NULL) {
    glColor3f(1, 0, 0);
  } else {
    glColor3f(color[0], color[1], color[2]);
  }

  glLineWidth(line_width);
  glBegin(GL_LINES);
  glVertex3f(0, 0, 0);
  glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
  glVertex3f(0, 0, 0);
  glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
  glVertex3f(0, 0, 0);
  glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
  glVertex3f(0, 0, 0);
  glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

  glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);
  glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

  glVertex3f(sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
  glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);

  glVertex3f(sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz);
  glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);

  glVertex3f(sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz);
  glVertex3f(sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz);

  glEnd();
  glPopMatrix();
}

void viewer::draw_pcld()
{
  std::unique_lock<std::mutex> lock(kf_lock);

  // for (auto& kf : active_keyframes_) {
  //   float color[3] = { 0, 0, 1 };
  //   draw_camera(kf, color);
  // }

  glPointSize(1.5);
  glBegin(GL_POINTS);
  for (auto kf : active_keyframes_) {
    Eigen::Matrix4f pose = kf->pose;
    for (int i = 0; i < kf->nr_pcld; i++) {
      Eigen::Vector4f pcur;
      int u, v;
      if (kf->pcld_buf != NULL) {
        float x = kf->pcld_buf[i].x, y = kf->pcld_buf[i].y, z = kf->pcld_buf[i].z;
        pcur << x, y, z, 1;
        u = fx * x/z + cx;
        v = fy * y/z + cy;
      } else {
        float z = 4096.0 / kf->feat[i].zinv;
        float x = kf->feat[i].u_cx_fx * z / 4096.0, y = kf->feat[i].v_cy_fy * z / 4096.0;
        pcur << x, y, z, 1;
        u = fx * x/z + cx;
        v = fy * y/z + cy;
      }
      Eigen::Vector4f pworld = pose * pcur;
      auto color = kf->img_rgb.at<cv::Vec3b>(v, u);
      glColor3f(color[2]/255.0, color[1]/255.0, color[0]/255.0);
      glVertex3d(pworld[0], pworld[1], pworld[2]);
    }
    // Eigen::Matrix4f pose = kf->pose.cast<float>();
    // glPushMatrix();
    // glMultMatrixf((float*)pose.data());
    // glPointSize(2);
    // glBindBuffer(GL_ARRAY_BUFFER, kf->vbo);
    // glEnableClientState(GL_VERTEX_ARRAY);
    // //XYZ1
    // glVertexPointer(3, GL_FLOAT, sizeof(float)*8, 0);
    // glEnableClientState(GL_COLOR_ARRAY);
    // //XYZ1,RGB1
    // glColorPointer(3, GL_FLOAT, sizeof(float)*8, (void *)(sizeof(float) * 4));
    // glDrawArrays(GL_POINTS, 0, kf->nr_pcld-1);
    // glDisableClientState(GL_COLOR_ARRAY);
    // glDisableClientState(GL_VERTEX_ARRAY);
    // glBindBuffer(GL_ARRAY_BUFFER, 0);
    // glPopMatrix();
  }
  glEnd();
}

void viewer::draw_trajectory(std::vector<Eigen::Matrix4f> &traj, float color[3])
{
  if (traj.size() == 0) {
    return;
  }

  glPushMatrix();
  glMultMatrixf((GLfloat*)traj[0].data());
  glColor3f(color[0], color[1], color[2]);
  glLineWidth(3);
  glBegin(GL_LINE_STRIP);
  for (auto traj : traj) {
    glVertex3f(traj(0,3), traj(1,3), traj(2,3));
  }
  glEnd();
  glPopMatrix();
}
