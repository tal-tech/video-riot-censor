cmake_minimum_required(VERSION 3.10)
project(image_riot_censor)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -frtti")

add_library(image_riot_censor SHARED image_riot.h image_riot.cpp)

target_include_directories(image_riot_censor PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
target_link_libraries(image_riot_censor ${TRITON_INFER_LIBRARY} ${OPENCV_LIBRARY} ${BOOST_LIBRARY} -ldl -lpthread)


