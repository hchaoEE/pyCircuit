from __future__ import annotations

from pycircuit import Circuit, Reg, Wire

from ..util import latch


def build_if_stage(m: Circuit, *, do_if: Wire, ifid_window: Reg, mem_rdata: Wire) -> None:
    latch(m, ifid_window, en=do_if, new=mem_rdata)

