cmake_minimum_required (VERSION 2.8)
project(picovo-mcu)

# set(CMAKE_BUILD_TYPE "Debug")
# # set(CMAKE_BUILD_TYPE "Release")

# set(CMAKE_CXX_FLAGS_DEBUG "-std=c++11 -Wall -Og -g -pthread -mssse3")
# set(CMAKE_CXX_FLAGS_RELEASE  "-std=c++11 -Ofast -fopenmp -pthread -mssse3")

enable_language(C CXX ASM)

# Get rid of -rdynamic
SET(BUILD_SHARED_LIBS OFF)
SET(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")
SET(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS "")

set(CMAKE_SYSTEM_NAME "Generic")

set(commonflags "-mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard -Wall -fdata-sections -ffunction-sections -g -Ofast")

set(CMAKE_C_COMPILER    "arm-none-eabi-gcc")
set(CMAKE_C_FLAGS       "${commonflags}")
set(CMAKE_CXX_COMPILER  "arm-none-eabi-g++")
set(CMAKE_CXX_FLAGS     "${commonflags}")
set(CMAKE_ASM_COMPILER  "arm-none-eabi-gcc")
set(CMAKE_ASM_FLAGS     "${commonflags}")
set(CMAKE_OBJCOPY       "arm-none-eabi-objcopy")
set(CMAKE_AR            "arm-none-eabi-ar")
set(CMAKE_RANLIB        "arm-none-eabi-ranlib")

add_definitions(-DSTM32F767xx -DUSE_HAL_DRIVER )

include_directories(drivers/CMSIS/Include)
include_directories(drivers/CMSIS/Device/ST/STM32F7xx/Include)
include_directories(drivers/STM32F7XX_HAL/Inc)
include_directories(drivers/fatfs)
include_directories(drivers)
include_directories(picovo-mcu)
include_directories(${CMAKE_SOURCE_DIR})
include_directories(../lib/)

# Eigen
include_directories("/usr/include/eigen3")

# startup file
set(STARTUP_FILE startup_stm32f767xx.s)
set_source_files_properties(${STARTUP_FILE} PROPERTIES LANGUAGE ASM)
set_source_files_properties(${STARTUP_FILE} PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")
add_library(startup STATIC ${STARTUP_FILE})

# std driver
set(STM32_STDDRIVER_FILES
  drivers/STM32F7XX_HAL/Src/stm32f7xx_hal_cortex.c
  drivers/STM32F7XX_HAL/Src/stm32f7xx_ll_sdmmc.c
  drivers/STM32F7XX_HAL/Src/stm32f7xx_hal_sd.c
  drivers/STM32F7XX_HAL/Src/stm32f7xx_hal_tim.c
  drivers/STM32F7XX_HAL/Src/stm32f7xx_hal_tim_ex.c
  drivers/STM32F7XX_HAL/Src/stm32f7xx_hal_uart.c
  drivers/STM32F7XX_HAL/Src/stm32f7xx_hal_rcc.c
  drivers/STM32F7XX_HAL/Src/stm32f7xx_hal_rcc_ex.c
  drivers/STM32F7XX_HAL/Src/stm32f7xx_hal_flash.c
  drivers/STM32F7XX_HAL/Src/stm32f7xx_hal_flash_ex.c
  drivers/STM32F7XX_HAL/Src/stm32f7xx_hal_gpio.c
  drivers/STM32F7XX_HAL/Src/stm32f7xx_hal_dma.c
  drivers/STM32F7XX_HAL/Src/stm32f7xx_hal_dma_ex.c
  drivers/STM32F7XX_HAL/Src/stm32f7xx_hal_pwr.c
  drivers/STM32F7XX_HAL/Src/stm32f7xx_hal_pwr_ex.c
  drivers/STM32F7XX_HAL/Src/stm32f7xx_hal.c
  drivers/STM32F7XX_HAL/Src/stm32f7xx_hal_i2c.c
  drivers/STM32F7XX_HAL/Src/stm32f7xx_hal_i2c_ex.c
  drivers/STM32F7XX_HAL/Src/stm32f7xx_hal_exti.c
)
add_library(stm32_stddriver STATIC ${STM32_STDDRIVER_FILES})

# fatfs
set(FATFS_FILES
  drivers/fatfs/diskio.c
  drivers/fatfs/ff.c
  drivers/fatfs/ff_gen_drv.c
  drivers/fatfs/option/syscall.c
  drivers/fatfs/option/ccsbcs.c
)
add_library(fatfs STATIC ${FATFS_FILES})

# misc drivers
set(MISC_DRIVER_FILES
  drivers/bsp_driver_sd.c
  drivers/fatfs.c
  drivers/sd_diskio.c
  drivers/system_stm32f7xx.c
)
add_library(misc_driver STATIC ${MISC_DRIVER_FILES})

# picovo algorithm
set(PICOVO_SRC_FILES
  picovo-mcu/tracker_rgbd.cc
  picovo-mcu/distance_transform.cc
  picovo-mcu/picoedge.cc
)
OPTION(USE_FLOAT_SOLVER "Use a floating point solver?" OFF)
if (USE_FLOAT_SOLVER)
  list(APPEND PICOVO_SRC_FILES picovo-mcu/solver_lm_float.cc)
else()
  list(APPEND PICOVO_SRC_FILES picovo-mcu/solver_lm_fixpoint.cc)
endif()

add_library(picovo_core STATIC ${PICOVO_SRC_FILES})

# main
set(TARGET_NAME "picovo_cm7")
# NB! The implementation of weak functions in the libraries must be added here so that it can appear
# in the final linking stage.
set(USER_CSRCS
  main.c
  run_tum.c
  drivers/stm32f7xx_hal_msp.c
  drivers/stm32f7xx_it.c
)
add_executable(${TARGET_NAME} ${USER_CSRCS})
set(MY_LINKER_SCRIPT "${CMAKE_SOURCE_DIR}/scripts/STM32F767ZITx_FLASH.ld")
set(CMAKE_EXE_LINKER_FLAGS "-specs=nosys.specs -Wl,--gc-sections -T ${MY_LINKER_SCRIPT}")
set_target_properties(${TARGET_NAME} PROPERTIES LINK_DEPENDS ${MY_LINKER_SCRIPT})
target_link_libraries(${TARGET_NAME} startup fatfs misc_driver stm32_stddriver picovo_core -lc -lm -lnosys)
add_custom_command(TARGET ${TARGET_NAME}
    POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} -O binary -R .itcm_data ${TARGET_NAME} ${TARGET_NAME}.bin)

