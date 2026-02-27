"""Case 5: sync_fifo — Synchronous FIFO with full/empty/overflow protection.

Tests: memory inference, pointer logic, formal verification.
"""
from __future__ import annotations

from pycircuit import Circuit, compile, module, u, unsigned

DEPTH = 16
ADDR_W = 4
DATA_W = 32


@module
def build(m: Circuit) -> None:
    clk = m.clock("clk")
    rst = m.reset("rst")

    wr_en = m.input("wr_en", width=1)
    wr_data = m.input("wr_data", width=DATA_W)
    rd_en = m.input("rd_en", width=1)

    wptr = m.out("wptr_q", clk=clk, rst=rst, width=ADDR_W + 1, init=u(ADDR_W + 1, 0))
    rptr = m.out("rptr_q", clk=clk, rst=rst, width=ADDR_W + 1, init=u(ADDR_W + 1, 0))

    mem = [
        m.out(f"mem{i}", clk=clk, rst=rst, width=DATA_W, init=u(DATA_W, 0))
        for i in range(DEPTH)
    ]

    wp = wptr.out()
    rp = rptr.out()
    w_addr = wp[0:ADDR_W]
    r_addr = rp[0:ADDR_W]

    full = (wp[ADDR_W] != rp[ADDR_W]) & (w_addr == r_addr)
    empty = wp == rp
    count_raw = (wp - rp)[0:ADDR_W + 1]
    half_full = count_raw >= u(ADDR_W + 1, DEPTH // 2)

    do_wr = wr_en & ~full
    do_rd = rd_en & ~empty

    wptr.set((wp + 1)[0:ADDR_W + 1], when=do_wr)
    rptr.set((rp + 1)[0:ADDR_W + 1], when=do_rd)

    for i in range(DEPTH):
        mem[i].set(wr_data, when=do_wr & (w_addr == u(ADDR_W, i)))

    rd_data = u(DATA_W, 0)
    for i in range(DEPTH):
        rd_data = mem[i].out() if r_addr == u(ADDR_W, i) else rd_data

    m.output("rd_data", rd_data)
    m.output("full", full)
    m.output("empty", empty)
    m.output("half_full", half_full)
    m.output("count", count_raw)
    m.output("overflow", wr_en & full)
    m.output("underflow", rd_en & empty)


build.__pycircuit_name__ = "sync_fifo"

if __name__ == "__main__":
    print(compile(build, name="sync_fifo").emit_mlir())
