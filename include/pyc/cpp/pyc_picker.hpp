#pragma once

#include <cstddef>

#include "pyc_bits.hpp"
#include "pyc_vec.hpp"

namespace pyc::cpp {

template <unsigned Width, std::size_t N>
struct pyc_picker_onehot {
  Wire<N> &sel;
  Vec<Wire<Width>, N> &in_data;
  Wire<Width> &y;

  void eval() {
    std::uint64_t out = 0;
    std::uint64_t s = sel.value();
    for (std::size_t i = 0; i < N; i++) {
      if (s & (std::uint64_t{1} << i)) {
        out |= in_data[i].value();
      }
    }
    y = Wire<Width>(out);
  }
};

} // namespace pyc::cpp

