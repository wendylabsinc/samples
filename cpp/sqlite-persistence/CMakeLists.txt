cmake_minimum_required(VERSION 3.16)
project(sqlite-persistence VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(SQLite3 REQUIRED)

add_executable(sqlite-persistence main.cpp)
target_link_libraries(sqlite-persistence SQLite::SQLite3)
