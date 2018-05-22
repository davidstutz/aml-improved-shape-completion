cmake_minimum_required(VERSION 3.2)
project(libvoxelizekitti)

set(CMAKE_CXX_FLAGS "--std=gnu++11 ${CMAKE_CXX_FLAGS} -O3 -g")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

find_package(HDF5 COMPONENTS C CXX HL REQUIRED)
find_package(Boost COMPONENTS system filesystem REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(
  ${Boost_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${HDF5_INCLUDE_DIRS}
)
add_executable(icp icp.cpp icp/icp.cpp icp/icpPointToPlane.cpp icp/icpPointToPoint.cpp icp/kdtree.cpp icp/matrix.cpp)
target_link_libraries(icp
  ${Boost_LIBRARIES}
  ${HDF5_CXX_LIBRARIES}
)
add_executable(sample sample.cpp)
target_link_libraries(sample
  ${Boost_LIBRARIES}
  ${HDF5_CXX_LIBRARIES}
)