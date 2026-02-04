#pragma once

#include <type_traits>

#include "pyc_bits.hpp"
#include "pyc_handshake.hpp"
#include "pyc_primitives.hpp"

namespace pyc::cpp {

// Ready/valid queue.
//
// - Depth==1: fall-through skid buffer (0-cycle when ready, 1-entry storage when stalled)
// - Depth>1: wraps `pyc_fifo` (1-cycle read latency, prototype)
template <unsigned Width, unsigned Depth>
class pyc_queue {
public:
  pyc_queue(Wire<1> &clk,
            Wire<1> &rst,
            Wire<1> &in_valid,
            Wire<1> &in_ready,
            Wire<Width> &in_data,
            Wire<1> &out_valid,
            Wire<1> &out_ready,
            Wire<Width> &out_data)
      : fifo(clk, rst, in_valid, in_ready, in_data, out_valid, out_ready, out_data) {}

  void eval() { fifo.eval(); }
  void tick() { fifo.tick(); }

  pyc_fifo<Width, Depth> fifo;
};

template <unsigned Width>
class pyc_queue<Width, 1> {
public:
  pyc_queue(Wire<1> &clk,
            Wire<1> &rst,
            Wire<1> &in_valid,
            Wire<1> &in_ready,
            Wire<Width> &in_data,
            Wire<1> &out_valid,
            Wire<1> &out_ready,
            Wire<Width> &out_data)
      : clk(clk), rst(rst), in_valid(in_valid), in_ready(in_ready), in_data(in_data), out_valid(out_valid),
        out_ready(out_ready), out_data(out_data) {
    eval();
  }

  void eval() {
    if (full) {
      out_valid = Wire<1>(1);
      out_data = storage;
      in_ready = Wire<1>(out_ready.toBool() ? 1u : 0u);
    } else {
      out_valid = in_valid;
      out_data = in_data;
      in_ready = Wire<1>(1);
    }
  }

  void tick() {
    bool clkNow = clk.toBool();
    bool posedge = (!clkPrev) && clkNow;
    clkPrev = clkNow;
    if (!posedge)
      return;

    if (rst.toBool()) {
      full = false;
      storage = Wire<Width>(0);
      eval();
      return;
    }

    if (full) {
      if (out_ready.toBool()) {
        if (in_valid.toBool()) {
          storage = in_data;
          full = true;
        } else {
          full = false;
        }
      }
    } else {
      // Empty: if downstream is stalled and we see valid, capture.
      if (in_valid.toBool() && !out_ready.toBool()) {
        storage = in_data;
        full = true;
      }
    }
    eval();
  }

public:
  Wire<1> &clk;
  Wire<1> &rst;
  Wire<1> &in_valid;
  Wire<1> &in_ready;
  Wire<Width> &in_data;
  Wire<1> &out_valid;
  Wire<1> &out_ready;
  Wire<Width> &out_data;

  bool clkPrev = false;
  bool full = false;
  Wire<Width> storage{};
};

} // namespace pyc::cpp
