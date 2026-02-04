from __future__ import annotations

from pycircuit import Module


def build() -> Module:
    m = Module("FifoLoopback")

    clk = m.clock("clk")
    rst = m.reset("rst")

    in_valid = m.input("in_valid", width=1)
    in_data = m.input("in_data", width=8)
    out_ready = m.input("out_ready", width=1)

    in_ready, out_valid, out_data = m.fifo(clk, rst, in_valid, in_data, out_ready, depth=2)

    m.output("in_ready", in_ready)
    m.output("out_valid", out_valid)
    m.output("out_data", out_data)
    return m

