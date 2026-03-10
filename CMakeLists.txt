cmake_minimum_required(VERSION 3.10)
project(OpticalFlow)

# set C++ Standard
set(CMAKE_CXX_STANDARD 17)

# find OpenCV package
find_package( OpenCV 4 REQUIRED )
#find_package(Sophus REQUIRED)
#find_package(Pangolin REQUIRED)

# create execute file
add_executable( tracking_optical_flow src/optical_flow.cpp src/SingleLayer-LucasKanade.cpp src/SingleLayer-ORB_GaussNewton.cpp )

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        #${G2O_INCLUDE_DIRS}
        #${Sophus_INCLUDE_DIRS}
        "/usr/include/eigen3/"
        #${Pangolin_INCLUDE_DIRS}
)

# set Include directory
target_include_directories( tracking_optical_flow PRIVATE ${OpenCV_INCLUDE_DIRS})
# set Link libraries
target_link_libraries( tracking_optical_flow PRIVATE ${OpenCV_LIBS})