#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "pyc_bits.hpp"

namespace pyc::cpp {

template <unsigned AddrWidth, unsigned DataWidth, std::size_t Depth>
class pyc_sram {
public:
  static_assert(DataWidth > 0 && DataWidth <= 64, "pyc_sram supports DataWidth 1..64 in the prototype");
  static_assert((DataWidth % 8) == 0, "pyc_sram requires DataWidth divisible by 8 in the prototype");
  static constexpr unsigned StrbWidth = DataWidth / 8;

  pyc_sram(Wire<1> &clk,
           Wire<1> &rst,
           Wire<1> &req_valid,
           Wire<1> &req_ready,
           Wire<AddrWidth> &req_addr,
           Wire<1> &req_write,
           Wire<DataWidth> &req_wdata,
           Wire<StrbWidth> &req_wstrb,
           Wire<1> &resp_valid,
           Wire<1> &resp_ready,
           Wire<DataWidth> &resp_rdata)
      : clk(clk), rst(rst), req_valid(req_valid), req_ready(req_ready), req_addr(req_addr), req_write(req_write),
        req_wdata(req_wdata), req_wstrb(req_wstrb), resp_valid(resp_valid), resp_ready(resp_ready),
        resp_rdata(resp_rdata) {
    eval();
  }

  void eval() {
    resp_valid = Wire<1>(pending ? 1u : 0u);
    resp_rdata = resp_rdata_r;

    bool ready = !pending || resp_ready.toBool();
    req_ready = Wire<1>(ready ? 1u : 0u);
  }

  void tick() {
    bool clkNow = clk.toBool();
    bool posedge = (!clkPrev) && clkNow;
    clkPrev = clkNow;
    if (!posedge)
      return;

    if (rst.toBool()) {
      pending = false;
      resp_rdata_r = Wire<DataWidth>(0);
      eval();
      return;
    }

    bool fire_resp = pending && resp_ready.toBool();
    bool fire_req = req_valid.toBool() && req_ready.toBool();

    if (fire_resp && !fire_req) {
      pending = false;
    }

    if (fire_req) {
      std::size_t idx = static_cast<std::size_t>(req_addr.value());
      if (idx < Depth) {
        if (req_write.toBool()) {
          writeBytes(idx, req_wdata.value(), req_wstrb.value());
          resp_rdata_r = Wire<DataWidth>(0);
        } else {
          resp_rdata_r = mem_[idx];
        }
      } else {
        resp_rdata_r = Wire<DataWidth>(0);
      }
      pending = true;
    }

    eval();
  }

  // Convenience for testbenches.
  void poke(std::size_t idx, std::uint64_t value) {
    if (idx < Depth)
      mem_[idx] = Wire<DataWidth>(value);
  }
  std::uint64_t peek(std::size_t idx) const { return (idx < Depth) ? mem_[idx].value() : 0; }

private:
  void writeBytes(std::size_t idx, std::uint64_t wdata, std::uint64_t wstrb) {
    std::uint64_t old = mem_[idx].value();
    std::uint64_t v = old;
    for (unsigned i = 0; i < StrbWidth; i++) {
      if (wstrb & (std::uint64_t{1} << i)) {
        std::uint64_t mask = std::uint64_t{0xFF} << (i * 8);
        v = (v & ~mask) | (wdata & mask);
      }
    }
    mem_[idx] = Wire<DataWidth>(v);
  }

public:
  Wire<1> &clk;
  Wire<1> &rst;

  Wire<1> &req_valid;
  Wire<1> &req_ready;
  Wire<AddrWidth> &req_addr;
  Wire<1> &req_write;
  Wire<DataWidth> &req_wdata;
  Wire<StrbWidth> &req_wstrb;

  Wire<1> &resp_valid;
  Wire<1> &resp_ready;
  Wire<DataWidth> &resp_rdata;

  bool clkPrev = false;
  bool pending = false;
  Wire<DataWidth> resp_rdata_r{};

private:
  std::array<Wire<DataWidth>, Depth> mem_{};
};

} // namespace pyc::cpp
