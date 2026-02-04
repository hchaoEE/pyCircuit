// pyCircuit C++ emission (prototype)
#include <pyc/cpp/pyc_sim.hpp>

namespace pyc::gen {

struct MultiClockRegs {
  pyc::cpp::Wire<1> clk_a{};
  pyc::cpp::Wire<1> rst_a{};
  pyc::cpp::Wire<1> clk_b{};
  pyc::cpp::Wire<1> rst_b{};
  pyc::cpp::Wire<8> a_count{};
  pyc::cpp::Wire<8> b_count{};

  pyc::cpp::Wire<1> v1{};
  pyc::cpp::Wire<8> v2{};
  pyc::cpp::Wire<8> v3{};
  pyc::cpp::Wire<1> v4{};
  pyc::cpp::Wire<8> v5{};
  pyc::cpp::Wire<8> v6{};
  pyc::cpp::Wire<8> v7{};
  pyc::cpp::Wire<8> v8{};
  pyc::cpp::Wire<8> v9{};
  pyc::cpp::Wire<8> v10{};
  pyc::cpp::Wire<8> v11{};
  pyc::cpp::Wire<8> v12{};

  pyc::cpp::pyc_reg<8> v7_inst;
  pyc::cpp::pyc_reg<8> v9_inst;
  pyc::cpp::pyc_reg<8> v10_inst;
  pyc::cpp::pyc_reg<8> v12_inst;

  MultiClockRegs() :
      v7_inst(clk_a, rst_a, v4, v5, v5, v7),
      v9_inst(clk_a, rst_a, v4, v8, v5, v9),
      v10_inst(clk_b, rst_b, v4, v5, v5, v10),
      v12_inst(clk_b, rst_b, v4, v11, v5, v12) {
    eval();
  }

  inline void eval_comb_0() {
    v1 = pyc::cpp::Wire<1>(1ull);
    v2 = pyc::cpp::Wire<8>(0ull);
    v3 = pyc::cpp::Wire<8>(1ull);
    v4 = v1;
    v5 = v2;
    v6 = v3;
  }

  void eval() {
    eval_comb_0();
    v8 = (v7 + v6);
    v11 = (v10 + v6);
    a_count = v9;
    b_count = v12;
  }

  void tick() {
    // Two-phase update: compute next state for all sequential elements,
    // then commit together. This avoids ordering artifacts between regs.
    // Phase 1: compute.
    v7_inst.tick_compute();
    v9_inst.tick_compute();
    v10_inst.tick_compute();
    v12_inst.tick_compute();
    // Phase 2: commit.
    v7_inst.tick_commit();
    v9_inst.tick_commit();
    v10_inst.tick_commit();
    v12_inst.tick_commit();
  }
};

} // namespace pyc::gen
