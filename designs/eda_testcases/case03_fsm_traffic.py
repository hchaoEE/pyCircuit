"""Case 3: fsm_traffic — Traffic light FSM controller.

Tests: FSM extraction, state encoding optimization, coverage analysis.
"""
from __future__ import annotations

from pycircuit import Circuit, compile, module, u, unsigned

S_NS_GREEN = 0
S_NS_YELLOW = 1
S_NS_RED = 2
S_EW_GREEN = 3
S_EW_YELLOW = 4
S_EW_RED = 5

GREEN_TICKS = 10
YELLOW_TICKS = 3


@module
def build(m: Circuit) -> None:
    clk = m.clock("clk")
    rst = m.reset("rst")

    emergency = m.input("emergency", width=1)

    state = m.out("state_q", clk=clk, rst=rst, width=3, init=u(3, S_NS_GREEN))
    timer = m.out("timer_q", clk=clk, rst=rst, width=5, init=u(5, 0))

    st = state.out()
    tmr = timer.out()
    timeout = tmr == u(5, 0)

    nxt_st = st
    nxt_tmr = (tmr - 1)[0:5] if ~timeout else tmr

    if st == u(3, S_NS_GREEN):
        if timeout:
            nxt_st = u(3, S_NS_YELLOW)
            nxt_tmr = u(5, YELLOW_TICKS - 1)
        if emergency:
            nxt_st = u(3, S_NS_YELLOW)
            nxt_tmr = u(5, 1)

    elif st == u(3, S_NS_YELLOW):
        if timeout:
            nxt_st = u(3, S_NS_RED)
            nxt_tmr = u(5, 1)

    elif st == u(3, S_NS_RED):
        if timeout:
            nxt_st = u(3, S_EW_GREEN)
            nxt_tmr = u(5, GREEN_TICKS - 1)

    elif st == u(3, S_EW_GREEN):
        if timeout:
            nxt_st = u(3, S_EW_YELLOW)
            nxt_tmr = u(5, YELLOW_TICKS - 1)
        if emergency:
            nxt_st = u(3, S_EW_YELLOW)
            nxt_tmr = u(5, 1)

    elif st == u(3, S_EW_YELLOW):
        if timeout:
            nxt_st = u(3, S_EW_RED)
            nxt_tmr = u(5, 1)

    elif st == u(3, S_EW_RED):
        if timeout:
            nxt_st = u(3, S_NS_GREEN)
            nxt_tmr = u(5, GREEN_TICKS - 1)

    state.set(nxt_st)
    timer.set(nxt_tmr)

    ns_green = (st == u(3, S_NS_GREEN))
    ns_yellow = (st == u(3, S_NS_YELLOW))
    ew_green = (st == u(3, S_EW_GREEN))
    ew_yellow = (st == u(3, S_EW_YELLOW))

    ns_r = ~ns_green & ~ns_yellow
    ew_r = ~ew_green & ~ew_yellow

    m.output("ns_red", ns_r)
    m.output("ns_yellow", ns_yellow)
    m.output("ns_green", ns_green)
    m.output("ew_red", ew_r)
    m.output("ew_yellow", ew_yellow)
    m.output("ew_green", ew_green)
    m.output("timer_val", timer)


build.__pycircuit_name__ = "fsm_traffic"

if __name__ == "__main__":
    print(compile(build, name="fsm_traffic").emit_mlir())
