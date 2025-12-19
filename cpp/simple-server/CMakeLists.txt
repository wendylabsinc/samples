cmake_minimum_required(VERSION 3.16)
project(simple-server VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Threads REQUIRED)

# Include directories for header-only libraries
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(simple-server main.cpp)

# Link pthread for httplib
target_link_libraries(simple-server PRIVATE Threads::Threads)
