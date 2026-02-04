// pyCircuit C++ emission (prototype)
#include <pyc/cpp/pyc_sim.hpp>

namespace pyc::gen {

struct Counter {
  pyc::cpp::Wire<1> clk{};
  pyc::cpp::Wire<1> rst{};
  pyc::cpp::Wire<1> en{};
  pyc::cpp::Wire<8> count{};

  pyc::cpp::Wire<8> v1{};
  pyc::cpp::Wire<8> v2{};
  pyc::cpp::Wire<8> v3{};
  pyc::cpp::Wire<8> v4{};
  pyc::cpp::Wire<8> v5{};
  pyc::cpp::Wire<8> v6{};
  pyc::cpp::Wire<8> v7{};

  pyc::cpp::pyc_reg<8> v5_inst;
  pyc::cpp::pyc_reg<8> v7_inst;

  Counter() :
      v5_inst(clk, rst, en, v4, v4, v5),
      v7_inst(clk, rst, en, v6, v4, v7) {
    eval();
  }

  inline void eval_comb_0() {
    v1 = pyc::cpp::Wire<8>(1ull);
    v2 = pyc::cpp::Wire<8>(0ull);
    v3 = v1;
    v4 = v2;
  }

  void eval() {
    eval_comb_0();
    v6 = (v5 + v3);
    count = v7;
  }

  void tick() {
    // Two-phase update: compute next state for all sequential elements,
    // then commit together. This avoids ordering artifacts between regs.
    // Phase 1: compute.
    v5_inst.tick_compute();
    v7_inst.tick_compute();
    // Phase 2: commit.
    v5_inst.tick_commit();
    v7_inst.tick_commit();
  }
};

} // namespace pyc::gen
