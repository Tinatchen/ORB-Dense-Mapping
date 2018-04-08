cmake_minimum_required(VERSION 2.8.3)
project(voxblox)


#set(CMAKE_MACOSX_RPATH 0)
add_definitions(-std=c++11 -Wno-sign-compare -Wno-unused-value)
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
#if (EXISTS "${CMAKE_CURRENT_LIST_DIR}/../tools/pybind11/CMakeLists.txt")
#  set(HAVE_PYBIND11 TRUE)
#  message(STATUS "Found pybind11; generating Python bindings")
#endif()

############
# PROTOBUF #
############
# General idea: first check if we have protobuf catkin, then use that.
# Otherwise use system protobuf.
set(PROTO_DEFNS proto/voxblox/Block.proto
        proto/voxblox/Layer.proto)
set(ADDITIONAL_LIBRARIES "")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

message(STATUS "Using system protobuf")
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS ${PROTO_DEFNS})
set(ADDITIONAL_LIBRARIES ${PROTOBUF_LIBRARY})

find_package(Glog REQUIRED)
find_package(Gflags REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
find_package( PCL REQUIRED )
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

include_directories(
        include
        ${EIGEN3_INCLUDE_DIR}
        ${GLOG_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${Gflags_INCLUDE_DIRS}
)
link_directories( ${PCL_LIBRARY_DIRS} )

####################
# SET SOURCE FILES #
####################

set("${PROJECT_NAME}_SRCS"
        src/core/block.cc
        src/core/esdf_map.cc
        src/core/tsdf_map.cc
        src/integrator/esdf_integrator.cc
        src/integrator/esdf_occ_integrator.cc
        src/integrator/tsdf_integrator.cc
        src/integrator/integrator_utils.cc
        src/io/mesh_ply.cc
        src/mesh/marching_cubes.cc
        src/simulation/objects.cc
        src/simulation/simulation_world.cc
        src/utils/camera_model.cc
        src/utils/layer_utils.cc
        src/utils/protobuf_utils.cc
        src/utils/timing.cc
        ${PROTO_SRCS}
        )

set(THIRD_PARTY_LIBS
        ${PROTOBUF_LIBRARIES}
        ${EIGEN3_LIBS}
        ${GLOG_LIBRARY}
        ${PCL_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        )

add_library(${PROJECT_NAME} SHARED
        ${${PROJECT_NAME}_SRCS}
        )
target_link_libraries(${PROJECT_NAME} ${THIRD_PARTY_LIBS})

add_executable(tsdf_to_esdf
        test/tsdf_to_esdf.cc
        )
target_link_libraries(tsdf_to_esdf ${PROJECT_NAME} ${THIRD_PARTY_LIBS})

add_executable(test_load_esdf
        test/test_load_esdf.cc
        )
target_link_libraries(test_load_esdf ${PROJECT_NAME} ${THIRD_PARTY_LIBS})


add_executable(test_voxblox
        test/test_voxblox.cpp
        )
target_link_libraries(test_voxblox
        ${PROJECT_NAME} ${THIRD_PARTY_LIBS})
