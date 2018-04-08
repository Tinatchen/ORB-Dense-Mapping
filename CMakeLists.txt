cmake_minimum_required(VERSION 2.8)
project(ORB_Dense_SLAM2)

SET(CMAKE_EXPORT_COMPILE_COMMANDS "ON")

SET(CMAKE_BUILD_TYPE Debug)

#IF(NOT CMAKE_BUILD_TYPE)
#  SET(CMAKE_BUILD_TYPE Release)
#ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
# adding for point cloud viewer and mapper
find_package( PCL REQUIRED )

include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

find_package(octomap REQUIRED)
include_directories( ${OCTOMAP_INCLUDE_DIRS} )


include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${PROJECT_SOURCE_DIR}/Thirdparty/sgm/include
${PROJECT_SOURCE_DIR}/Thirdparty/voxblox/include
${PROJECT_SOURCE_DIR}/Thirdparty/voxblox/build
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${PCL_INCLUDE_DIRS}
)

link_directories( ${PCL_LIBRARY_DIRS} )

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc
src/DenseMapping.cc
src/VoxbloxMapping.cpp
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
${PROJECT_SOURCE_DIR}/Thirdparty/sgm/lib/libdeeprob_cv_stereo.so
${PROJECT_SOURCE_DIR}/Thirdparty/voxblox/lib/libvoxblox.so
${PCL_LIBRARIES}
${OCTOMAP_LIBRARIES}
)

# Build examples

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_executable(rgbd_tum 
    Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})

add_executable(rgbd_my
    Examples/RGB-D/rgbd_my.cc)
target_link_libraries(rgbd_my ${PROJECT_NAME})

add_executable(stereo_kitti
Examples/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

add_executable(stereo_zed
        Examples/Stereo/stereo_zed.cc
        )
target_link_libraries(stereo_zed
        ${PROJECT_NAME})

add_executable(stereo_kitti_dense
        Examples/Stereo/stereo_kitti_dense.cc)
target_link_libraries(stereo_kitti_dense ${PROJECT_NAME})

add_executable(stereo_seg_dense_kitti
        Examples/Stereo/stereo_seg_dense_kitti.cpp)
target_link_libraries(stereo_seg_dense_kitti ${PROJECT_NAME})

add_executable(stereo_seg_voxblox_kitti
        Examples/Stereo/stereo_seg_voxblox_kitti.cpp)
target_link_libraries(stereo_seg_voxblox_kitti ${PROJECT_NAME})

add_executable(stereo_fcn_voxblox_kitti
        Examples/Stereo/stereo_fcn_voxblox_kitti.cpp)
target_link_libraries(stereo_fcn_voxblox_kitti ${PROJECT_NAME})

add_executable(compare_sgm_fcn
        Examples/Stereo/compare_sgm_fcn.cpp)
target_link_libraries(compare_sgm_fcn ${PROJECT_NAME})

add_executable(stereo_euroc
Examples/Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc ${PROJECT_NAME})

add_executable(stereo_euroc_dense
        Examples/Stereo/stereo_euroc_dense.cc)
target_link_libraries(stereo_euroc_dense ${PROJECT_NAME})

set(USE_CUDA ON)
if (USE_CUDA)
   set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
   find_package(CUDA REQUIRED)
   include_directories(
           ${PROJECT_SOURCE_DIR}/Thirdparty/cuda_sgm/include
   )
   MESSAGE("Has CUDA")
   set(CUDA_ARCH "-arch=sm_50" CACHE STRING "Value of the NVCC -arch option.")

   set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

   CUDA_ADD_EXECUTABLE(stereo_euroc_dense_cuda
           Examples/Stereo/stereo_euroc_dense_cuda.cpp
           ${CUDA_SRC}
           )
   TARGET_LINK_LIBRARIES(stereo_euroc_dense_cuda
           ${PROJECT_NAME}
           ${PROJECT_SOURCE_DIR}/Thirdparty/cuda_sgm/lib/libsgm.so
           ${CUDA_LIBRARIES} ${CUDA_nppi_LIBRARY}
           )
endif ()

add_executable(test_surface
        test/test_surface.cpp
        )
target_link_libraries(test_surface
        ${PROJECT_NAME}
        )

add_executable(mono_tum
Examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})

add_executable(mono_kitti
Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})

add_executable(mono_apple
        Examples/Monocular/mon_apple.cpp)
target_link_libraries(mono_apple ${PROJECT_NAME})

add_executable(mono_apple_seg
        Examples/Monocular/mon_apple_seg.cpp)
target_link_libraries(mono_apple_seg ${PROJECT_NAME})

add_executable(mono_euroc
Examples/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME})


# Build tools
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/tools)
add_executable(bin_vocabulary
tools/bin_vocabulary.cc)
target_link_libraries(bin_vocabulary ${PROJECT_NAME})
