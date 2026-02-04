// pyCircuit C++ emission (prototype)
#include <pyc/cpp/pyc_sim.hpp>

namespace pyc::gen {

struct WireOps {
  pyc::cpp::Wire<1> sys_clk{};
  pyc::cpp::Wire<1> sys_rst{};
  pyc::cpp::Wire<8> a{};
  pyc::cpp::Wire<8> b{};
  pyc::cpp::Wire<1> sel{};
  pyc::cpp::Wire<8> y{};

  pyc::cpp::Wire<8> v1{};
  pyc::cpp::Wire<1> v2{};
  pyc::cpp::Wire<8> v3{};
  pyc::cpp::Wire<8> v4{};
  pyc::cpp::Wire<8> v5{};
  pyc::cpp::Wire<8> v6{};
  pyc::cpp::Wire<1> v7{};
  pyc::cpp::Wire<8> v8{};
  pyc::cpp::Wire<8> v9{};

  pyc::cpp::pyc_reg<8> v9_inst;

  WireOps() :
      v9_inst(sys_clk, sys_rst, v7, v8, v6, v9) {
    eval();
  }

  inline void eval_comb_0() {
    v1 = pyc::cpp::Wire<8>(0ull);
    v2 = pyc::cpp::Wire<1>(1ull);
    v3 = (a & b);
    v4 = (a ^ b);
    v5 = (sel.toBool() ? v3 : v4);
    v6 = v1;
    v7 = v2;
    v8 = v5;
  }

  void eval() {
    eval_comb_0();
    y = v9;
  }

  void tick() {
    // Two-phase update: compute next state for all sequential elements,
    // then commit together. This avoids ordering artifacts between regs.
    // Phase 1: compute.
    v9_inst.tick_compute();
    // Phase 2: commit.
    v9_inst.tick_commit();
  }
};

} // namespace pyc::gen
