cmake_minimum_required(VERSION 2.8)

project(deeprob_cv_stereo_test)

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
MESSAGE(STATUS "OpenCV Include: " ${OpenCV_INCLUDE_DIRS})
LINK_DIRECTORIES(${OpenCV_LIBRARY_DIRS})
MESSAGE(STATUS "OpenCV Library: " ${OpenCV_LIBRARY_DIRS})
ADD_DEFINITIONS(${OpenCV_DEFINITIONS})

add_executable(test_local test_local.cpp test_utility.cpp test_utility.h)
target_link_libraries(test_local deeprob_cv_stereo opencv_core opencv_imgproc opencv_video opencv_highgui)

add_executable(test_sgm test_sgm.cpp test_utility.cpp test_utility.h)
target_link_libraries(test_sgm deeprob_cv_stereo opencv_core opencv_imgproc opencv_video opencv_highgui)

add_executable(test_euroc test_euroc.cpp test_utility.cpp test_utility.h)
target_link_libraries(test_euroc deeprob_cv_stereo opencv_core opencv_imgproc opencv_video opencv_highgui)


install(DIRECTORY include DESTINATION include/${CMAKE_PROJECT_NAME})
