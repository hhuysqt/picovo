#! /usr/bin/python
# Resize the dataset into 320x240, and store the raw data blob for MCU

import cv2
import os, sys, shutil, getopt

def helpmsg(fname):
  print('Usage: ' + fname + " -i <input folder> -o <output folder>")
  sys.exit(-1)

if __name__ == "__main__":
  inputfolder = ''
  outputfolder = ''
  try:
    opts, args = getopt.getopt(sys.argv[1:], "i:o:")
  except getopt.GetoptError:
    helpmsg(sys.argv[0])

  for opt, arg in opts:
    if opt == '-i':
      inputfolder = arg
    elif opt == '-o':
      outputfolder = arg
  if inputfolder.__len__() == 0 or outputfolder.__len__() == 0:
    helpmsg(sys.argv[0])

  print('input folder: ' + inputfolder + ", output folder: " + outputfolder)
  ascfilename = inputfolder + '/associate.txt'
  if not os.path.exists(ascfilename):
    print(ascfilename + ' not exist...')
    sys.exit(-1)
  if not os.path.exists(outputfolder):
    os.mkdir(outputfolder)
  outputfolder = os.path.abspath(outputfolder)

  # FATFS has poor performance in a directory with many files.
  # Thus gather all the data into a binary blob to speed up reading on MCU.
  datablobname = outputfolder + '/data.bin'

  shutil.copy(ascfilename, outputfolder + '/associate.txt')
  with open(ascfilename, "rb") as asefile:
    with open(datablobname, "wb") as datablob:
      lines = asefile.readlines()
      cnt, nr_data = 0, lines.__len__()
      print('%d files' % nr_data)
      last_pct = int(0)
      sys.stdout.write(' 0%')
      sys.stdout.flush()
      for line in lines:
        files = line.strip('\n').split(' ')
        rgbfile, depthfile = files[1], files[3]
        cur_pct = int(cnt/nr_data)
        cnt = cnt+100
        if cur_pct != last_pct:
          last_pct = cur_pct
          sys.stdout.write('\r%2d%%' % cur_pct)
          sys.stdout.flush()
        # print('rgbfile: ' + rgbfile + ', depthfile: ' + depthfile)
        gray = cv2.resize(cv2.imread(inputfolder + '/' + rgbfile, cv2.IMREAD_GRAYSCALE), (320,240))
        depth = cv2.resize(cv2.imread(inputfolder + '/' + depthfile, cv2.IMREAD_UNCHANGED), (320,240))
        gray.tofile(datablob)
        depth.tofile(datablob)

  print("\r100%\nDone.")
