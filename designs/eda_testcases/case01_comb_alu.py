"""Case 1: comb_alu — 32-bit pure-combinational ALU (no registers).

Tests: logic synthesis optimization, critical path analysis, area minimization.
"""
from __future__ import annotations

from pycircuit import Circuit, compile, module, u, unsigned


@module
def build(m: Circuit) -> None:
    a = m.input("a", width=32)
    b = m.input("b", width=32)
    op = m.input("op", width=3)

    add_r = (a + b)[0:32]
    sub_r = (a - b)[0:32]
    and_r = a & b
    or_r = a | b
    xor_r = a ^ b

    sa = a[31]
    sb = b[31]
    sd = sa ^ sb
    ult = a < b
    slt_1 = sa if sd else ult
    slt_r = (unsigned(slt_1) + u(32, 0))[0:32]

    sll_r = a
    sh = b[0:5]
    for i in range(5):
        t = (sll_r << (1 << i))[0:32]
        sll_r = t if sh[i] else sll_r

    srl_r = a
    for i in range(5):
        lo = srl_r[(1 << i):32]
        t = (unsigned(lo) + u(32, 0))[0:32]
        srl_r = t if sh[i] else srl_r

    result = add_r
    result = sub_r if op == u(3, 1) else result
    result = and_r if op == u(3, 2) else result
    result = or_r if op == u(3, 3) else result
    result = xor_r if op == u(3, 4) else result
    result = slt_r if op == u(3, 5) else result
    result = sll_r if op == u(3, 6) else result
    result = srl_r if op == u(3, 7) else result

    zero = result == u(32, 0)
    carry = a[31] ^ b[31] ^ result[31]

    m.output("result", result)
    m.output("zero", zero)
    m.output("carry", carry)


build.__pycircuit_name__ = "comb_alu"

if __name__ == "__main__":
    print(compile(build, name="comb_alu").emit_mlir())
