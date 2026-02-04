from __future__ import annotations

from pycircuit import Circuit


def build(m: Circuit, STAGES: int = 3) -> None:
    dom = m.domain("sys")
    en = m.const_wire(1, width=1)

    a = m.in_wire("a", width=16)
    b = m.in_wire("b", width=16)
    sel = m.in_wire("sel", width=1)

    # Some combinational logic feeding a multi-field pipeline bus.
    sum_ = a + b
    x = a ^ b
    data = sel.select(sum_, x)
    tag = a.eq(b)
    lo8 = data.slice(lsb=0, width=8)

    fields = m.vec(tag, data, lo8)
    bus = fields.pack()

    # Pipeline the packed bus through STAGES registers.
    for _ in range(STAGES):
        bus = m.reg_domain(dom, en, bus, 0).q

    out_fields = fields.unpack(bus)
    m.output("tag", out_fields[0])
    m.output("data", out_fields[1])
    m.output("lo8", out_fields[2])

