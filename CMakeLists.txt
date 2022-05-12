# Cmake version
cmake_minimum_required(VERSION 3.7...3.22)

if(${CMAKE_VERSION} VERSION_LESS 3.12)
  cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

# Project settings
project(
  gradient_descent
  VERSION 0.1.0
  DESCRIPTION "Parallel computing coursework. Gradient descent"
  LANGUAGES CXX)

# Compiler configuration
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wall")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# Executable
set(SOURCES
    src/main.cpp
    src/interface.cpp
    src/core.cpp
    src/sgd.cpp
    src/parallel_sgd.cpp
    src/thread_pool.cpp
    src/early_stopping.cpp
    src/scaler.cpp)
add_executable(${PROJECT_NAME} ${SOURCES})

# External packages (C++ threads, Eigen, Boost program options)
find_package(Threads REQUIRED)
find_package (Eigen3 3.4 REQUIRED NO_MODULE)
find_package(Boost 1.78 REQUIRED COMPONENTS program_options)

# Include headers
set(INCLUDES include src)
target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDES})

# Link external libraries
set(LIBRARIES Threads::Threads Eigen3::Eigen Boost::program_options)
target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBRARIES})
