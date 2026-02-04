#pragma once

#include <cstddef>
#include <cstdint>

#include "pyc_bits.hpp"
#include "pyc_vec.hpp"

namespace pyc::cpp {

constexpr unsigned ct_log2_ceil(unsigned n) {
  if (n <= 1)
    return 1;
  unsigned w = 0;
  unsigned v = n - 1;
  while (v) {
    w++;
    v >>= 1;
  }
  return w;
}

template <unsigned Width, std::size_t N>
class pyc_rr_arb {
public:
  static_assert(N > 0, "pyc_rr_arb requires N > 0");
  static constexpr unsigned SelWidth = ct_log2_ceil(static_cast<unsigned>(N));

  struct Sel {
    bool valid = false;
    std::size_t idx = 0;
  };

  pyc_rr_arb(Wire<1> &clk,
             Wire<1> &rst,
             Wire<N> &in_valid,
             Wire<N> &in_ready,
             Vec<Wire<Width>, N> &in_data,
             Wire<1> &out_valid,
             Wire<1> &out_ready,
             Wire<Width> &out_data,
             Wire<SelWidth> &out_sel)
      : clk(clk), rst(rst), in_valid(in_valid), in_ready(in_ready), in_data(in_data), out_valid(out_valid),
        out_ready(out_ready), out_data(out_data), out_sel(out_sel) {
    eval();
  }

  void eval() {
    const auto sel = select();
    selected = sel;

    if (sel.valid) {
      out_valid = Wire<1>(1);
      out_data = in_data[sel.idx];
      out_sel = Wire<SelWidth>(sel.idx);

      std::uint64_t r = 0;
      if (out_ready.toBool())
        r |= (std::uint64_t{1} << sel.idx);
      in_ready = Wire<N>(r);
    } else {
      out_valid = Wire<1>(0);
      out_data = Wire<Width>(0);
      out_sel = Wire<SelWidth>(0);
      in_ready = Wire<N>(0);
    }
  }

  void tick() {
    bool clkNow = clk.toBool();
    bool posedge = (!clkPrev) && clkNow;
    clkPrev = clkNow;
    if (!posedge)
      return;

    if (rst.toBool()) {
      rr_ptr = 0;
      eval();
      return;
    }

    // Consume when out_valid && out_ready.
    if (selected.valid && out_ready.toBool()) {
      rr_ptr = (selected.idx + 1 >= N) ? 0 : (selected.idx + 1);
    }
    eval();
  }

public:
  Wire<1> &clk;
  Wire<1> &rst;
  Wire<N> &in_valid;
  Wire<N> &in_ready;
  Vec<Wire<Width>, N> &in_data;
  Wire<1> &out_valid;
  Wire<1> &out_ready;
  Wire<Width> &out_data;
  Wire<SelWidth> &out_sel;

  bool clkPrev = false;
  std::size_t rr_ptr = 0;
  Sel selected{};

private:
  Sel select() const {
    std::uint64_t v = in_valid.value();
    if (v == 0)
      return {};

    for (std::size_t off = 0; off < N; off++) {
      std::size_t idx = rr_ptr + off;
      if (idx >= N)
        idx -= N;
      if (v & (std::uint64_t{1} << idx))
        return Sel{true, idx};
    }
    return {};
  }
};

} // namespace pyc::cpp
