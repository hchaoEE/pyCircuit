"""Case 6: multi_clock — Dual clock-domain design with CDC synchronizer.

Tests: CDC analysis, metastability checking, multi-clock SDC.
"""
from __future__ import annotations

from pycircuit import Circuit, compile, module, u, unsigned


@module
def build(m: Circuit) -> None:
    clk_fast = m.clock("clk_fast")
    rst_fast = m.reset("rst_fast")
    clk_slow = m.clock("clk_slow")
    rst_slow = m.reset("rst_slow")

    data_in = m.input("data_in", width=8)
    capture = m.input("capture", width=1)

    # Fast domain: 8-bit counter + capture register
    fast_cnt = m.out("fast_cnt", clk=clk_fast, rst=rst_fast, width=8, init=u(8, 0))
    fast_cnt.set((fast_cnt.out() + 1)[0:8])

    capt_val = m.out("capt_val", clk=clk_fast, rst=rst_fast, width=8, init=u(8, 0))
    capt_val.set(data_in, when=capture)

    capt_flag = m.out("capt_flag", clk=clk_fast, rst=rst_fast, width=1, init=u(1, 0))
    capt_flag.set(capture | capt_flag.out())

    # CDC: 2-stage synchronizer (fast → slow domain)
    sync1 = m.out("sync1", clk=clk_slow, rst=rst_slow, width=1, init=u(1, 0))
    sync2 = m.out("sync2", clk=clk_slow, rst=rst_slow, width=1, init=u(1, 0))
    sync1.set(capt_flag.out())
    sync2.set(sync1.out())

    sync_data1 = m.out("sdata1", clk=clk_slow, rst=rst_slow, width=8, init=u(8, 0))
    sync_data2 = m.out("sdata2", clk=clk_slow, rst=rst_slow, width=8, init=u(8, 0))
    sync_data1.set(capt_val.out())
    sync_data2.set(sync_data1.out())

    # Slow domain: edge detect + accumulator
    sync2_prev = m.out("s2prev", clk=clk_slow, rst=rst_slow, width=1, init=u(1, 0))
    sync2_prev.set(sync2.out())
    rising_edge = sync2.out() & ~sync2_prev.out()

    slow_acc = m.out("slow_acc", clk=clk_slow, rst=rst_slow, width=16, init=u(16, 0))
    acc_val = slow_acc.out()
    new_acc = (acc_val + (unsigned(sync_data2.out()) + u(16, 0)))[0:16]
    slow_acc.set(new_acc, when=rising_edge)

    m.output("fast_count", fast_cnt)
    m.output("captured", capt_val)
    m.output("sync_valid", sync2)
    m.output("slow_accumulator", slow_acc)


build.__pycircuit_name__ = "multi_clock"

if __name__ == "__main__":
    print(compile(build, name="multi_clock").emit_mlir())
