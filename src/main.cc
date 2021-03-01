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

#include <iostream>

#include "tracker_rgbd.h"
#ifdef USE_PANGOLIN
#include "viewer.h"
#endif
#include "picovo_config.h"

#include <opencv2/opencv.hpp>  

int main(int argc, char const *argv[])
{
  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <config file>" << std::endl;
    return -1;
  }

  picovo::picovo_config sys_config(argv[1]);

  if (sys_config.tracker_type == picovo::TRACKER_RGBD_DATASET) {
    picovo::tracker_rgbd *tracker = new picovo::tracker_rgbd();
    tracker->start(sys_config);
  } else {
    // TODO: implement other trackers
  }

  return 0;
}
