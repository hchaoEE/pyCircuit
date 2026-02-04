// pyCircuit C++ emission (prototype)
#include <pyc/cpp/pyc_sim.hpp>

namespace pyc::gen {

struct FifoLoopback {
  pyc::cpp::Wire<1> clk{};
  pyc::cpp::Wire<1> rst{};
  pyc::cpp::Wire<1> in_valid{};
  pyc::cpp::Wire<8> in_data{};
  pyc::cpp::Wire<1> out_ready{};
  pyc::cpp::Wire<1> in_ready{};
  pyc::cpp::Wire<1> out_valid{};
  pyc::cpp::Wire<8> out_data{};

  pyc::cpp::Wire<1> v1{};
  pyc::cpp::Wire<1> v2{};
  pyc::cpp::Wire<8> v3{};

  pyc::cpp::pyc_fifo<8, 2> v1_inst;

  FifoLoopback() :
      v1_inst(clk, rst, in_valid, v1, in_data, v2, out_ready, v3) {
    eval();
  }

  void eval() {
    v1_inst.eval();
    in_ready = v1;
    out_valid = v2;
    out_data = v3;
  }

  void tick() {
    // Two-phase update: compute next state for all sequential elements,
    // then commit together. This avoids ordering artifacts between regs.
    // Phase 1: compute.
    v1_inst.tick_compute();
    // Phase 2: commit.
    v1_inst.tick_commit();
  }
};

} // namespace pyc::gen
