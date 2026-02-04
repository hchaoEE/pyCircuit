// pyCircuit C++ emission (prototype)
#include <pyc/cpp/pyc_sim.hpp>

namespace pyc::gen {

struct JitPipelineVec {
  pyc::cpp::Wire<1> sys_clk{};
  pyc::cpp::Wire<1> sys_rst{};
  pyc::cpp::Wire<16> a{};
  pyc::cpp::Wire<16> b{};
  pyc::cpp::Wire<1> sel{};
  pyc::cpp::Wire<1> tag{};
  pyc::cpp::Wire<16> data{};
  pyc::cpp::Wire<8> lo8{};

  pyc::cpp::Wire<25> v1{};
  pyc::cpp::Wire<1> v2{};
  pyc::cpp::Wire<16> v3{};
  pyc::cpp::Wire<16> v4{};
  pyc::cpp::Wire<16> v5{};
  pyc::cpp::Wire<1> v6{};
  pyc::cpp::Wire<8> v7{};
  pyc::cpp::Wire<25> v8{};
  pyc::cpp::Wire<25> v9{};
  pyc::cpp::Wire<25> v10{};
  pyc::cpp::Wire<25> v11{};
  pyc::cpp::Wire<25> v12{};
  pyc::cpp::Wire<25> v13{};
  pyc::cpp::Wire<25> v14{};
  pyc::cpp::Wire<25> v15{};
  pyc::cpp::Wire<25> v16{};
  pyc::cpp::Wire<1> v17{};
  pyc::cpp::Wire<25> v18{};
  pyc::cpp::Wire<25> v19{};
  pyc::cpp::Wire<25> v20{};
  pyc::cpp::Wire<25> v21{};
  pyc::cpp::Wire<8> v22{};
  pyc::cpp::Wire<16> v23{};
  pyc::cpp::Wire<1> v24{};
  pyc::cpp::Wire<8> v25{};
  pyc::cpp::Wire<16> v26{};
  pyc::cpp::Wire<1> v27{};

  pyc::cpp::pyc_reg<25> v19_inst;
  pyc::cpp::pyc_reg<25> v20_inst;
  pyc::cpp::pyc_reg<25> v21_inst;

  JitPipelineVec() :
      v19_inst(sys_clk, sys_rst, v17, v18, v16, v19),
      v20_inst(sys_clk, sys_rst, v17, v19, v16, v20),
      v21_inst(sys_clk, sys_rst, v17, v20, v16, v21) {
    eval();
  }

  inline void eval_comb_0() {
    v1 = pyc::cpp::Wire<25>(0ull);
    v2 = pyc::cpp::Wire<1>(1ull);
    v3 = (a + b);
    v4 = (a ^ b);
    v5 = (sel.toBool() ? v3 : v4);
    v6 = pyc::cpp::Wire<1>((a == b) ? 1u : 0u);
    v7 = pyc::cpp::extract<8, 16>(v5, 0u);
    v8 = pyc::cpp::zext<25, 8>(v7);
    v9 = (v1 | v8);
    v10 = pyc::cpp::zext<25, 16>(v5);
    v11 = pyc::cpp::Wire<25>(v10.value() << 8ull);
    v12 = (v9 | v11);
    v13 = pyc::cpp::zext<25, 1>(v6);
    v14 = pyc::cpp::Wire<25>(v13.value() << 24ull);
    v15 = (v12 | v14);
    v16 = v1;
    v17 = v2;
    v18 = v15;
  }

  inline void eval_comb_1() {
    v22 = pyc::cpp::extract<8, 25>(v21, 0u);
    v23 = pyc::cpp::extract<16, 25>(v21, 8u);
    v24 = pyc::cpp::extract<1, 25>(v21, 24u);
    v25 = v22;
    v26 = v23;
    v27 = v24;
  }

  void eval() {
    eval_comb_0();
    eval_comb_1();
    tag = v27;
    data = v26;
    lo8 = v25;
  }

  void tick() {
    // Two-phase update: compute next state for all sequential elements,
    // then commit together. This avoids ordering artifacts between regs.
    // Phase 1: compute.
    v19_inst.tick_compute();
    v20_inst.tick_compute();
    v21_inst.tick_compute();
    // Phase 2: commit.
    v19_inst.tick_commit();
    v20_inst.tick_commit();
    v21_inst.tick_commit();
  }
};

} // namespace pyc::gen
