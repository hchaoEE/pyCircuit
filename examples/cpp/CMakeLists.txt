add_executable(pyc_cpp_tb_fifo
  tb_fifo.cpp
)
target_include_directories(pyc_cpp_tb_fifo PRIVATE
  "${PROJECT_SOURCE_DIR}/include"
)
target_compile_features(pyc_cpp_tb_fifo PRIVATE cxx_std_17)

