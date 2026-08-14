cmake_minimum_required(VERSION 3.21)
project(HIP_Playground LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 23)

include(cmake/gtest.cmake)

FetchContent_MakeAvailable(googletest)

include(GoogleTest)
enable_testing()

file(GLOB SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/*.hip")

add_executable(Playground ${SOURCES})

target_link_libraries(Playground GTest::gtest)