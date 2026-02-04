from __future__ import annotations

from pycircuit import Module


def build() -> Module:
    m = Module("MultiClockRegs")

    clk_a = m.clock("clk_a")
    rst_a = m.reset("rst_a")
    clk_b = m.clock("clk_b")
    rst_b = m.reset("rst_b")

    en = m.const(1, width=1)
    zero8 = m.const(0, width=8)
    one8 = m.const(1, width=8)

    a0 = m.reg(clk_a, rst_a, en, zero8, zero8)
    a1 = m.add(a0, one8)
    a = m.reg(clk_a, rst_a, en, a1, zero8)

    b0 = m.reg(clk_b, rst_b, en, zero8, zero8)
    b1 = m.add(b0, one8)
    b = m.reg(clk_b, rst_b, en, b1, zero8)

    m.output("a_count", a)
    m.output("b_count", b)
    return m

