// pyCircuit C++ emission (prototype)
#include <pyc/cpp/pyc_sim.hpp>

namespace pyc::gen {

struct JitControlFlow {
  pyc::cpp::Wire<8> a{};
  pyc::cpp::Wire<8> b{};
  pyc::cpp::Wire<8> out{};

  pyc::cpp::Wire<8> v1{};
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


  JitControlFlow() {
    eval();
  }

  inline void eval_comb_0() {
    v1 = pyc::cpp::Wire<8>(2ull);
    v2 = pyc::cpp::Wire<8>(1ull);
    v3 = (a + b);
    v4 = pyc::cpp::Wire<1>((a == b) ? 1u : 0u);
    v5 = (v3 + v2);
    v6 = (v3 + v1);
    v7 = (v4.toBool() ? v5 : v6);
    v8 = (v7 + v2);
    v9 = (v8 + v2);
    v10 = (v9 + v2);
    v11 = (v10 + v2);
    v12 = v11;
  }

  void eval() {
    eval_comb_0();
    out = v12;
  }

  void tick() {
    // Two-phase update: compute next state for all sequential elements,
    // then commit together. This avoids ordering artifacts between regs.
    // Phase 1: compute.
    // Phase 2: commit.
  }
};

} // namespace pyc::gen
