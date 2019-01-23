cmake_minimum_required(VERSION 3.13)
project(BatchGradientDescent)

set(CMAKE_CXX_STANDARD 17)

# Boost

set(Boost_USE_STATIC_LIBS       ON)
set(Boost_USE_DEBUG_LIBS        OFF)
set(Boost_USE_RELEASE_LIBS      ON)
set(Boost_USE_MULTITHREADED     ON)
set(Boost_USE_STATIC_RUNTIME    OFF)

find_package(Boost 1.65.0 REQUIRED)
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    add_executable(BGD main.cpp BGD.h)
    target_link_libraries(BGD ${Boost_LIBRARIES})
endif()