cmake_minimum_required(VERSION 3.20)
project(pycircuit LANGUAGES C CXX)

option(PYC_BUILD_MLIR_TOOLS "Build MLIR-based pyc tools" ON)
option(PYC_BUILD_CPP_EXAMPLES "Build C++ simulation examples" ON)

if(PYC_BUILD_MLIR_TOOLS)
  add_subdirectory(pyc/mlir)
endif()

if(PYC_BUILD_CPP_EXAMPLES)
  add_subdirectory(examples/cpp)
endif()
