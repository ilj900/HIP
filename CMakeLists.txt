cmake_minimum_required(VERSION 3.21)
project(HIP_Playground LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)

add_executable(my_app
        vector_add.hip
        vector_add.h
        main.cpp)

target_include_directories(my_app PRIVATE $ENV{ROCM_PATH}/include)