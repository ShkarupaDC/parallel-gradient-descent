cmake_minimum_required(VERSION 3.7...3.22)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

project(
  gradient_descent
  VERSION 0.1.0
  DESCRIPTION "Parallel computing coursework. Gradient descent"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)

add_executable(gradient_descent src/main.cpp)
target_include_directories(gradient_descent PUBLIC include/gradient_descent
                                                   ${xtensor_INCLUDE_DIRS})
target_link_libraries(gradient_descent PUBLIC xtensor xtensor::optimize)
