"""Case 7: bus_arbiter — 4-port round-robin bus arbiter.

Tests: high fanout optimization, priority encoding, fairness verification.
"""
from __future__ import annotations

from pycircuit import Circuit, compile, module, u, unsigned

NUM_PORTS = 4


@module
def build(m: Circuit) -> None:
    clk = m.clock("clk")
    rst = m.reset("rst")

    req = [m.input(f"req{i}", width=1) for i in range(NUM_PORTS)]
    lock = [m.input(f"lock{i}", width=1) for i in range(NUM_PORTS)]
    data_in = [m.input(f"data{i}", width=32) for i in range(NUM_PORTS)]

    last_grant = m.out("last_grant", clk=clk, rst=rst, width=2, init=u(2, 0))
    active = m.out("active_q", clk=clk, rst=rst, width=1, init=u(1, 0))
    active_port = m.out("active_port", clk=clk, rst=rst, width=2, init=u(2, 0))
    locked = m.out("locked_q", clk=clk, rst=rst, width=1, init=u(1, 0))

    lg = last_grant.out()
    act = active.out()
    ap = active_port.out()
    lck = locked.out()

    any_req = req[0] | req[1] | req[2] | req[3]

    # Fixed-priority encoder (3 > 2 > 1 > 0), rotated by last_grant
    winner = u(2, 0) if req[0] else lg
    winner = u(2, 1) if req[1] else winner
    winner = u(2, 2) if req[2] else winner
    winner = u(2, 3) if req[3] else winner

    # Rotate: prefer port after last_grant for fairness
    for off in range(1, NUM_PORTS):
        for idx in range(NUM_PORTS):
            cand = (lg + u(2, off))[0:2]
            winner = cand if (req[idx] & (cand == u(2, idx))) else winner

    nxt_act = act
    nxt_ap = ap
    nxt_lg = lg
    nxt_lck = lck

    cur_req_held = req[0]
    for i in range(1, NUM_PORTS):
        cur_req_held = req[i] if ap == u(2, i) else cur_req_held

    cur_lock = lock[0]
    for i in range(1, NUM_PORTS):
        cur_lock = lock[i] if ap == u(2, i) else cur_lock

    release = ~cur_req_held & ~lck

    if act:
        nxt_lck = cur_lock
        if release:
            nxt_act = u(1, 0)
    else:
        if any_req:
            nxt_act = u(1, 1)
            nxt_ap = winner
            nxt_lg = winner

    active.set(nxt_act)
    active_port.set(nxt_ap)
    last_grant.set(nxt_lg)
    locked.set(nxt_lck)

    # Grant outputs
    bus_data = u(32, 0)
    for i in range(NUM_PORTS):
        grant_i = act & (ap == u(2, i))
        m.output(f"grant{i}", grant_i)
        bus_data = data_in[i] if grant_i else bus_data

    m.output("bus_data", bus_data)
    m.output("bus_valid", active)


build.__pycircuit_name__ = "bus_arbiter"

if __name__ == "__main__":
    print(compile(build, name="bus_arbiter").emit_mlir())
