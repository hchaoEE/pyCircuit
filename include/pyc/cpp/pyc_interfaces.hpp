#pragma once

#include "pyc_bits.hpp"

namespace pyc::cpp {

// Equivalent of `pyc_stream_if.sv`.
template <unsigned Width>
struct pyc_stream_if {
  Wire<1> valid{};
  Wire<1> ready{};
  Wire<Width> data{};
};

// Equivalent of `pyc_mem_if.sv` (explicit signal bundle).
template <unsigned AddrWidth, unsigned DataWidth>
struct pyc_mem_if {
  static_assert(DataWidth > 0 && DataWidth <= 64, "pyc_mem_if supports DataWidth 1..64 in the prototype");
  static_assert((DataWidth % 8) == 0, "pyc_mem_if requires DataWidth divisible by 8 in the prototype");

  static constexpr unsigned StrbWidth = DataWidth / 8;

  Wire<1> req_valid{};
  Wire<1> req_ready{};
  Wire<AddrWidth> req_addr{};
  Wire<1> req_write{};
  Wire<DataWidth> req_wdata{};
  Wire<StrbWidth> req_wstrb{};

  Wire<1> resp_valid{};
  Wire<1> resp_ready{};
  Wire<DataWidth> resp_rdata{};
};

} // namespace pyc::cpp

