"""Case 8: hier_soc — Hierarchical mini-SoC (core + timer + GPIO).

Tests: hierarchical synthesis, module partitioning, interface checking.
"""
from __future__ import annotations

from pycircuit import Circuit, compile, module, u, unsigned


@module
def _timer(m: Circuit) -> None:
    """Programmable countdown timer with interrupt."""
    clk = m.clock("clk")
    rst = m.reset("rst")
    load_val = m.input("load_val", width=16)
    load_en = m.input("load_en", width=1)
    enable = m.input("enable", width=1)

    cnt = m.out("cnt_q", clk=clk, rst=rst, width=16, init=u(16, 0))
    irq_r = m.out("irq_q", clk=clk, rst=rst, width=1, init=u(1, 0))

    cv = cnt.out()
    at_zero = cv == u(16, 0)
    nxt = (cv - 1)[0:16] if (enable & ~at_zero) else cv
    nxt = load_val if load_en else nxt

    cnt.set(nxt)
    irq_r.set(at_zero & enable)

    m.output("count", cnt)
    m.output("irq", irq_r)


@module
def _gpio(m: Circuit) -> None:
    """8-bit GPIO with direction control."""
    clk = m.clock("clk")
    rst = m.reset("rst")
    dir_wr = m.input("dir_wr", width=1)
    dir_data = m.input("dir_data", width=8)
    out_wr = m.input("out_wr", width=1)
    out_data = m.input("out_data", width=8)
    pin_in = m.input("pin_in", width=8)

    direction = m.out("dir_q", clk=clk, rst=rst, width=8, init=u(8, 0))
    out_reg = m.out("out_q", clk=clk, rst=rst, width=8, init=u(8, 0))

    direction.set(dir_data, when=dir_wr)
    out_reg.set(out_data, when=out_wr)

    pin_out = out_reg.out() & direction.out()
    pin_read = (pin_in & ~direction.out()) | pin_out

    m.output("pin_out", pin_out)
    m.output("pin_oe", direction)
    m.output("pin_read", pin_read)


@module
def build(m: Circuit) -> None:
    clk = m.clock("clk")
    rst = m.reset("rst")

    addr = m.input("addr", width=4)
    wdata = m.input("wdata", width=16)
    wen = m.input("wen", width=1)
    ren = m.input("ren", width=1)
    gpio_pin_in = m.input("gpio_pin_in", width=8)

    acc = m.out("acc_q", clk=clk, rst=rst, width=16, init=u(16, 0))
    pc_reg = m.out("pc_q", clk=clk, rst=rst, width=8, init=u(8, 0))
    pc_reg.set((pc_reg.out() + 1)[0:8])

    is_timer_addr = (addr[2:4] == u(2, 1))
    is_gpio_addr = (addr[2:4] == u(2, 2))
    is_core_addr = (addr[2:4] == u(2, 0))

    # Core: simple accumulator
    acc_next = acc.out()
    if is_core_addr & wen:
        if addr[0:2] == u(2, 0):
            acc_next = wdata
        elif addr[0:2] == u(2, 1):
            acc_next = (acc.out() + wdata)[0:16]
        elif addr[0:2] == u(2, 2):
            acc_next = acc.out() & wdata
        elif addr[0:2] == u(2, 3):
            acc_next = acc.out() | wdata
    acc.set(acc_next)

    # Timer instance
    timer_out = _timer(
        m, clk=clk, rst=rst,
        load_val=wdata,
        load_en=is_timer_addr & wen & (addr[0:2] == u(2, 0)),
        enable=is_timer_addr & wen & (addr[0:2] == u(2, 1)),
    )

    # GPIO instance
    gpio_out = _gpio(
        m, clk=clk, rst=rst,
        dir_wr=is_gpio_addr & wen & (addr[0:2] == u(2, 0)),
        dir_data=wdata[0:8],
        out_wr=is_gpio_addr & wen & (addr[0:2] == u(2, 1)),
        out_data=wdata[0:8],
        pin_in=gpio_pin_in,
    )

    m.output("pc", pc_reg)
    m.output("acc_val", acc)


build.__pycircuit_name__ = "hier_soc"

if __name__ == "__main__":
    print(compile(build, name="hier_soc").emit_mlir())
