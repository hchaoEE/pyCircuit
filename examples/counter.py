from __future__ import annotations

from pycircuit import Module


def build() -> Module:
    m = Module("Counter")

    clk = m.clock("clk")
    rst = m.reset("rst")
    en = m.input("en", width=1)

    one = m.const(1, width=8)
    zero = m.const(0, width=8)
    q0 = m.reg(clk, rst, en, zero, zero)
    q1 = m.add(q0, one)
    q = m.reg(clk, rst, en, q1, zero)

    m.output("count", q)
    return m

