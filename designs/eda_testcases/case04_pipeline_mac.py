"""Case 4: pipeline_mac — 4-stage pipelined multiply-accumulate.

Tests: DSP inference, pipeline retiming, throughput analysis.
"""
from __future__ import annotations

from pycircuit import Circuit, compile, module, u, unsigned


@module
def build(m: Circuit) -> None:
    clk = m.clock("clk")
    rst = m.reset("rst")

    a_in = m.input("a_in", width=16)
    b_in = m.input("b_in", width=16)
    valid_in = m.input("valid_in", width=1)
    acc_clear = m.input("acc_clear", width=1)
    saturate_mode = m.input("saturate_mode", width=1)

    s1_a = m.out("s1_a", clk=clk, rst=rst, width=16, init=u(16, 0))
    s1_b = m.out("s1_b", clk=clk, rst=rst, width=16, init=u(16, 0))
    s1_v = m.out("s1_v", clk=clk, rst=rst, width=1, init=u(1, 0))
    s1_clr = m.out("s1_clr", clk=clk, rst=rst, width=1, init=u(1, 0))
    s1_sat = m.out("s1_sat", clk=clk, rst=rst, width=1, init=u(1, 0))

    s2_prod = m.out("s2_prod", clk=clk, rst=rst, width=32, init=u(32, 0))
    s2_v = m.out("s2_v", clk=clk, rst=rst, width=1, init=u(1, 0))
    s2_clr = m.out("s2_clr", clk=clk, rst=rst, width=1, init=u(1, 0))
    s2_sat = m.out("s2_sat", clk=clk, rst=rst, width=1, init=u(1, 0))

    s3_sum = m.out("s3_sum", clk=clk, rst=rst, width=40, init=u(40, 0))
    s3_v = m.out("s3_v", clk=clk, rst=rst, width=1, init=u(1, 0))
    s3_sat = m.out("s3_sat", clk=clk, rst=rst, width=1, init=u(1, 0))

    acc = m.out("acc_q", clk=clk, rst=rst, width=40, init=u(40, 0))

    s1_a.set(a_in)
    s1_b.set(b_in)
    s1_v.set(valid_in)
    s1_clr.set(acc_clear)
    s1_sat.set(saturate_mode)

    a_ext = (unsigned(s1_a.out()) + u(32, 0))[0:32]
    b_ext = (unsigned(s1_b.out()) + u(32, 0))[0:32]
    product = (a_ext * b_ext)[0:32]

    s2_prod.set(product)
    s2_v.set(s1_v.out())
    s2_clr.set(s1_clr.out())
    s2_sat.set(s1_sat.out())

    acc_val = acc.out()
    acc_base = u(40, 0) if s2_clr.out() else acc_val
    prod_ext = (unsigned(s2_prod.out()) + u(40, 0))[0:40]
    raw_sum = (acc_base + prod_ext)[0:40]

    max40 = u(40, (1 << 39) - 1)
    overflow = raw_sum[39]
    sat_sum = max40 if (overflow & s2_sat.out()) else raw_sum

    s3_sum.set(sat_sum)
    s3_v.set(s2_v.out())
    s3_sat.set(s2_sat.out())

    acc.set(s3_sum.out(), when=s3_v.out())

    result = s3_sum.out()[0:32]
    m.output("result", result)
    m.output("valid_out", s3_v)
    m.output("accumulator", acc)


build.__pycircuit_name__ = "pipeline_mac"

if __name__ == "__main__":
    print(compile(build, name="pipeline_mac").emit_mlir())
