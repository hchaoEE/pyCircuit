from __future__ import annotations

from pycircuit import Circuit, Wire

from ..pipeline import ExMemRegs, MemWbRegs
from ..util import Consts, latch_many


def build_mem_stage(m: Circuit, *, do_mem: Wire, exmem: ExMemRegs, memwb: MemWbRegs, mem_rdata: Wire, consts: Consts) -> None:
    load32 = mem_rdata.trunc(width=32)
    load64 = load32.sext(width=64)
    mem_val = exmem.is_load.select(load64, exmem.alu)
    mem_val = exmem.is_store.select(consts.zero64, mem_val)

    latch_many(
        m,
        do_mem,
        [
            (memwb.op, exmem.op),
            (memwb.len_bytes, exmem.len_bytes),
            (memwb.regdst, exmem.regdst),
            (memwb.value, mem_val),
        ],
    )

