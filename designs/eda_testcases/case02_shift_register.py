"""Case 2: shift_register — 32-bit configurable shift register.

Tests: setup/hold timing, scan chain insertion, clock gating.
"""
from __future__ import annotations

from pycircuit import Circuit, compile, module, u, unsigned


@module
def build(m: Circuit) -> None:
    clk = m.clock("clk")
    rst = m.reset("rst")

    din = m.input("din", width=1)
    pdin = m.input("pdin", width=32)
    mode = m.input("mode", width=2)
    shift_en = m.input("shift_en", width=1)

    sr = [
        m.out(f"sr{i}", clk=clk, rst=rst, width=1, init=u(1, 0))
        for i in range(32)
    ]

    for i in range(32):
        cur = sr[i].out()
        left_in = sr[i - 1].out() if i > 0 else din
        right_in = sr[i + 1].out() if i < 31 else din
        circ_left = sr[(i - 1) % 32].out()
        pbit = pdin[i]

        nxt = left_in if mode == u(2, 0) else cur
        nxt = right_in if mode == u(2, 1) else nxt
        nxt = circ_left if mode == u(2, 2) else nxt
        nxt = pbit if mode == u(2, 3) else nxt

        sr[i].set(nxt, when=shift_en)

    dout_bus = sr[0].out()
    for i in range(1, 32):
        dout_bus = m.cat(sr[i], dout_bus)

    m.output("dout", sr[31])
    m.output("pout", dout_bus)


build.__pycircuit_name__ = "shift_register"

if __name__ == "__main__":
    print(compile(build, name="shift_register").emit_mlir())
