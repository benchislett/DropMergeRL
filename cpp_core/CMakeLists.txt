cmake_minimum_required(VERSION 3.15)
project(cpp_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Development)
find_package(pybind11 REQUIRED)

add_library(cpp_core MODULE src/core.cpp)
set_target_properties(cpp_core PROPERTIES PREFIX "" OUTPUT_NAME "cpp_core")

target_link_libraries(cpp_core PRIVATE pybind11::module)

target_include_directories(cpp_core PRIVATE ${Python_INCLUDE_DIRS})
