cmake_minimum_required(VERSION 3.21)
project(MyProject LANGUAGES CXX HIP)      # <- drop HIP from LANGUAGES

set(CMAKE_CXX_STANDARD 17)

add_executable(my_app
        vector_add.hip                # clang treats .hip as HIP source automatically
        vector_add.h
        main.cpp)

target_include_directories(my_app PRIVATE $ENV{ROCM_PATH}/include)