cmake_minimum_required(VERSION 2.8.8)

find_package(Caffe)
include_directories(${Caffe_INCLUDE_DIRS})
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY


add_executable(caffeinated_application main.cpp utils.cpp utils.h)
target_link_libraries(caffeinated_application ${Caffe_LIBRARIES})
