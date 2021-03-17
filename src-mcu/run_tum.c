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

#include "main.h"
#include "fatfs.h"
#include "tracker_rgbd.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>

extern TIM_HandleTypeDef htim2;

// global buffer for input grayscale & depth image
static uint8_t _inputbuf[IMG_WIDTH*IMG_HEIGHT*3];
static uint8_t * const pgray = &_inputbuf[0];
static uint16_t * const pdepth = (uint16_t*)(&_inputbuf[IMG_WIDTH*IMG_HEIGHT]);

// Store the datasets to run in datasets.txt in the root directory of SD card, 
// one dataset folder per line.
static char dataset_name_buf[512];
static char *dataset_names[MAX_NR_DATASETS];
static int nr_datasets;

// override
void run_dataset(void)
{
  init_tracker();
  HAL_TIM_Base_Start(&htim2);

  FATFS fs;
  if (f_mount(&fs, "0:", 0) != FR_OK){
    printf("mount failed...\n");
    while (1);
  }
  {
    // grab dataset configs
    FIL dataset_file;
    if (f_open(&dataset_file, "datasets.txt", FA_READ) != FR_OK) {
      myputs("No dataset config file...\n");
      while(1);
    }
    UINT br;
    if (f_read(&dataset_file, dataset_name_buf, sizeof(dataset_name_buf), &br) != FR_OK) {
      myputs("read config error...\n");
      while(1);
    }
    dataset_names[0] = &dataset_name_buf[0];
    nr_datasets = 0;
    for (int i = 0; i < sizeof(dataset_name_buf) && nr_datasets < MAX_NR_DATASETS; i++) {
      if (dataset_name_buf[i] == '\n') {
        dataset_name_buf[i] = 0;
        nr_datasets++;
        dataset_names[nr_datasets] = &dataset_name_buf[i+1];
      }
    }
    f_close(&dataset_file);
  }

  for (int i = 0; i < nr_datasets && strlen(dataset_names[i]) > 0; i++) {
    myputs("\nProcessing dataset "), myputs(dataset_names[i]), myputs("\n");

    reset_tracker();

    #define PRBUFSIZE 256
    char namebuf[PRBUFSIZE];
    namebuf[PRBUFSIZE-1] = 0;
    sprintf(namebuf, "%s/", dataset_names[i]);
    int baselen = strlen(namebuf);

    FIL asc_file, img_file, pose_file, stat_file;
    sprintf(&namebuf[baselen], "associate.txt");
    if (f_open(&asc_file, namebuf, FA_READ) != FR_OK) {
      myputs("open "), myputs(namebuf), myputs(" failed...\n");
      while (1);
    }
    sprintf(&namebuf[baselen], "data.bin");
    if (f_open(&img_file, namebuf, FA_READ) != FR_OK) {
      myputs("open "), myputs(namebuf), myputs(" failed...\n");
      while(1);
    }
    char buf[PRBUFSIZE];
    buf[PRBUFSIZE-1] = 0;
    // result files
    sprintf(buf, "poses-%s.txt", dataset_names[i]);
    if (f_open(&pose_file, buf, FA_WRITE | FA_CREATE_ALWAYS) != FR_OK) {
      myputs("open poses.txt failed...\n");
      while (1);
    }
    sprintf(buf, "stat-%s.csv", dataset_names[i]);
    if (f_open(&stat_file, buf, FA_WRITE | FA_CREATE_ALWAYS) != FR_OK) {
      myputs("open stat.csv failed...\n");
      while (1);
    }

    UINT br;
    int datacnt = 0, lastprlen = 0;
    while (!f_eof(&asc_file)) {
      char *etimestamp = &buf[0];
      int nr_space = 0;
      for (int i = 0; i < PRBUFSIZE-1; i++) {
        char ch;
        if (f_read(&asc_file, &ch, 1, &br) != FR_OK) {
          break;
        }
        if (ch == '\n') {
          buf[i] = 0;
          break;
        } else {
          buf[i] = ch;
          if (ch == ' ' && nr_space == 0) {
            etimestamp = &buf[i+1];
            nr_space++;
          }
        }
      }

      // Compiler barrier for true timer measurement
      #define BARRIER asm volatile("": : :"memory")
      BARRIER;
      uint32_t start_rd_tim = htim2.Instance->CNT;
      BARRIER;
      if (f_read(&img_file, pgray, IMG_WIDTH*IMG_HEIGHT, &br) != FR_OK) {
        myputs("read "), myputs(namebuf), myputs(" failed...\n");
        while(1);
      }
      if (f_read(&img_file, pdepth, IMG_WIDTH*IMG_HEIGHT*2, &br) != FR_OK) {
        myputs("read "), myputs(namebuf), myputs(" failed...\n");
        while(1);
      }
      BARRIER;
      uint32_t start_tim = htim2.Instance->CNT;
      BARRIER;
      track_frame_rgbd(pgray, pdepth);
      BARRIER;
      uint32_t end_tim = htim2.Instance->CNT;
      BARRIER;

      char buf2[16];
      snprintf(buf2, sizeof(buf2), "%d: ", datacnt++);
      get_current_pose_tum_str(etimestamp, PRBUFSIZE - (etimestamp-buf));
      // myputs(buf2), myputs(buf);
      if (f_write(&pose_file, buf, strlen(buf), &br) != FR_OK) {
        myputs("write pose failed...\n");
        while(1);
      }
      sprintf(buf, "stat: %u, %u, ", (unsigned)(start_tim-start_rd_tim), (unsigned)(end_tim-start_tim));
      int len1 = strlen(buf);
      get_stat_str(&buf[len1], PRBUFSIZE-len1);
      char *statbuf = &buf[6];
      if (f_write(&stat_file, statbuf, strlen(statbuf), &br) != FR_OK) {
        myputs("write stat failed...\n");
        while(1);
      }
      int prlen = strlen(buf);
      if (prlen < lastprlen) {
        memset(&buf[prlen-1], ' ', lastprlen - prlen);
        buf[lastprlen-1] = '\r';
        buf[lastprlen] = 0;
      } else {
        buf[prlen-1] = '\r';
        buf[prlen] = 0;
      }
      lastprlen = prlen;
      myputs(buf);
    }
    f_close(&asc_file);
    f_close(&img_file);
    f_close(&stat_file);
    f_close(&pose_file);
  }
  myputs("\nDone\n");
}
