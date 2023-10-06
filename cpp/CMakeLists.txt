cmake_minimum_required(VERSION 3.10)

project(video_riot_censor)
set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/package")
#set(PACKVERSION 2.0.0.1)
######################################################################
#全局编译标志
######################################################################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

set(CUDA_cublas_device_LIBRARY /usr/local/cuda/lib64)
######################################################################
#全局链接和包含
######################################################################

set(TRITON_INFER_DIR ${PROJECT_SOURCE_DIR}/3rdParty/triton_infer/linux/1.0.0)
set(TRITON_INFER_INCLUDE ${TRITON_INFER_DIR}/include)
file(GLOB TRITON_INFER_LIBRARY ${TRITON_INFER_DIR}/lib/*.so)
message(STATUS "found TRITON_INFER_INCLUDE: ${TRITON_INFER_INCLUDE}, TRITON_INFER_LIBRARY:${TRITON_INFER_LIBRARY}")

# 设置opencv库路径(git地址: git@git.100tal.com:jituan_AILab_Alg_AIEngine/opencv.git)
set(OPENCV_DIR ${PROJECT_SOURCE_DIR}/3rdParty/opencv)
set(OPENCV_INCLUDE ${OPENCV_DIR}/include)
file(GLOB OPENCV_LIBRARY ${OPENCV_DIR}/lib/*)
message(STATUS "found OPENCV_INCLUDE:${OPENCV_INCLUDE}, OPENCV_LIBRARY:${OPENCV_LIBRARY}")

# 设置boost库路径(git地址: git@git.100tal.com:jituan_AILab_Alg_AIEngine/boost.git)
set(BOOST_DIR ${PROJECT_SOURCE_DIR}/3rdParty/boost/linux/boost_1_65_1)
set(BOOST_INCLUDE ${BOOST_DIR}/include)
file(GLOB BOOST_LIBRARY ${BOOST_DIR}/lib/*)
message(STATUS "found BOOST_INCLUDE:${BOOST_INCLUDE}, BOOST_LIBRARY:${BOOST_LIBRARY}")
add_definitions(-DBOOST_ALL_DYN_LINK)


include_directories(${TRITON_INFER_INCLUDE} ${OPENCV_INCLUDE} ${BOOST_INCLUDE})

#IF(CMAKE_BUILD_TYPE MATCHES Debug)
set(CMAKE_VERBOSE_MAKEFILE ON)
#ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)

set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/${BUILD_TYPE})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${BUILD_TYPE})

add_subdirectory(src)

add_executable(video_riot_censor demo.cpp)
target_link_libraries(video_riot_censor image_riot_censor ${TRITON_INFER_LIBRARY} ${OPENCV_LIBRARY} ${BOOST_LIBRARY} -ldl -lpthread)


include_directories(${PROJECT_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/3rdParty)
include_directories(${CMAKE_SOURCE_DIR}/3rdParty/opencv/include)
# include_directories(${CMAKE_SOURCE_DIR}/3rdParty/triton_infer/linux/1.0.0/include)
# include_directories(${CMAKE_SOURCE_DIR}/3rdParty/boost/linux/boost_1_65_1/include)
include_directories(${CMAKE_SOURCE_DIR}/ai_model/include)
link_directories(${CMAKE_SOURCE_DIR}/3rdParty/opencv/lib)
link_directories(${CMAKE_SOURCE_DIR}/3rdParty/triton_infer/linux/1.0.0/lib)
link_directories(${CMAKE_SOURCE_DIR}/3rdParty/boost/linux/boost_1_65_1/lib)
link_directories(${CMAKE_SOURCE_DIR}/ai_model/libs/linux/gpu)
link_directories(${CMAKE_SOURCE_DIR}/3rdParty/ubuntu-cuda10)

find_package(Boost REQUIRED)
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif()

#################################################################L#####
#全局编译输出指定
######################################################################

#IF(CMAKE_BUILD_TYPE MATCHES Debug)
set(CMAKE_VERBOSE_MAKEFILE ON)
#ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)

# string(TOLOWER ${CMAKE_BUILD_TYPE} BUILD_TYPE)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/${BUILD_TYPE})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/${BUILD_TYPE})


######################################################################
#子工程目录
######################################################################
# add_subdirectory(base/third_party/dynamic_annotations)
# add_subdirectory(base/third_party/symbolize)
# add_subdirectory(base/third_party/xdg_mime)
# add_subdirectory(base/third_party/libevent)
# add_subdirectory(third_party)
# add_subdirectory(base)

######################################################################
#common目录 获取所有源文件
######################################################################
# aux_source_directory(${CMAKE_SOURCE_DIR}/common DIR_COMMON_SRCS)
######################################################################
#编译对象
######################################################################
# add_executable(MicroserviceDemo
#     demo.cpp
# )

# if(Boost_FOUND)
#     target_link_libraries(MicroserviceDemo ${Boost_LIBRARIES})
# endif()

# target_link_libraries(MicroserviceDemo
#         jsoncpp
#         crypto
#         boost_system
#         opencv_core
#         opencv_imgcodecs
#         pthread
#         ${TRITON_INFER_LIBRARY}
#     )


######################################################################
#程序发布设置
######################################################################

install(DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ DESTINATION bin)
install(DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/ DESTINATION lib)
install(DIRECTORY config/ DESTINATION bin/config)
install(DIRECTORY 3rdParty/ubuntu-cuda10 DESTINATION lib)
install(DIRECTORY 3rdParty/opencv/lib DESTINATION .)
install(DIRECTORY ai_model DESTINATION .)
