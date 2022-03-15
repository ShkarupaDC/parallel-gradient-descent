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
    src/utils.cpp
    src/arg_parser.cpp)
add_executable(gradient_descent ${SOURCES})

# C++ threads
find_package(Threads REQUIRED)

# Xtensor
find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)

# Xtensor-BLAS
add_definitions(-DHAVE_CBLAS=1)

if(WIN32)
  find_package(OpenBLAS REQUIRED)
  set(BLAS_LIBRARIES ${CMAKE_INSTALL_PREFIX}${OpenBLAS_LIBRARIES})
else()
  find_package(BLAS REQUIRED)
  find_package(LAPACK REQUIRED)
endif()

message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})
message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})

# Boost
find_package(Boost 1.78 REQUIRED COMPONENTS program_options)

# Include headers
set(INCLUDES include src ${xtensor_INCLUDE_DIRS})
target_include_directories(gradient_descent PUBLIC ${INCLUDES})

# Link external libraries
set(LIBRARIES Threads::Threads xtensor xtensor::optimize ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES}
              Boost::program_options)
target_link_libraries(gradient_descent PUBLIC ${LIBRARIES})
