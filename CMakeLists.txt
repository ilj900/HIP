cmake_minimum_required(VERSION 3.21)
project(HIP_Playground LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 23)

include(FetchContent)
enable_testing()

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.17.0
)

FetchContent_MakeAvailable(googletest)

file(GLOB SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.hip")

add_executable(Playground ${SOURCES})

target_include_directories(Playground PRIVATE $ENV{ROCM_PATH}/include)
target_link_libraries(Playground GTest::gtest)