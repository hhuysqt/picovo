cmake_minimum_required(VERSION 2.8)
project(picovo)

# set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_FLAGS_DEBUG "-std=c++11 -Wall -Og -g -pthread -mssse3")
set(CMAKE_CXX_FLAGS_RELEASE  "-std=c++11 -Ofast -g -fopenmp -pthread -mssse3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
# set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

############### dependencies ######################
# Eigen
include_directories("/usr/include/eigen3")

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(THIRD_PARTY_LIBS
  ${OpenCV_LIBS}
  pthread
)

set(PICOVO_SRC_FILES
  src/main.cc
  src/tracker/tracker_rgbd.cc
  src/imgproc/picoedge.cc
  src/imgproc/distance_transform.cc
  src/imgproc/distance_transform_nn.cc
  src/solver/solver_gd.cc
  src/solver/solver_lm.cc
  src/solver/solver_lm_nn.cc
  src/solver/solver_lm_w_lut.cc
  src/solver/solver_lm_ic.cc
  src/solver/solver_lm_fixpoint.cc
)

# pangolin
OPTION(WITH_PANGOLIN_VIEWER "Build with Pangolin viewer?" OFF)
if (WITH_PANGOLIN_VIEWER)
  find_package(Pangolin REQUIRED)
  include_directories(${Pangolin_INCLUDE_DIRS})
  list(APPEND THIRD_PARTY_LIBS ${Pangolin_LIBRARIES})
  list(APPEND PICOVO_SRC_FILES src/viewer/viewer.cc)
  add_definitions(-DUSE_PANGOLIN)
endif()

enable_testing()

############### source and test ######################
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src/tracker)
include_directories(${PROJECT_SOURCE_DIR}/src/imgproc)
include_directories(${PROJECT_SOURCE_DIR}/src/solver)
include_directories(${PROJECT_SOURCE_DIR}/src/viewer)
include_directories(${PROJECT_SOURCE_DIR}/lib)

add_executable(picovo ${PICOVO_SRC_FILES})
target_link_libraries(picovo ${THIRD_PARTY_LIBS})
