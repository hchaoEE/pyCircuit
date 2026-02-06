from __future__ import annotations

from pycircuit import Circuit, Wire, jit_inline

from ..pipeline import ExMemRegs, MemWbRegs


@jit_inline
def build_mem_stage(
    m: Circuit,
    *,
    do_mem: Wire,
    exmem: ExMemRegs,
    memwb: MemWbRegs,
    mem_rdata: Wire,
    # Store->load forwarding from WB stage (for precise WB-commit stores).
    wb_store_valid: Wire,
    wb_store_addr: Wire,
    wb_store_size: Wire,
    wb_store_wdata: Wire,
) -> Wire:
    mem_val = m.const(0, width=64)
    with m.scope("MEM"):
        # Stage inputs.
        pc = exmem.pc.out()
        window = exmem.window.out()
        op = exmem.op.out()
        len_bytes = exmem.len_bytes.out()
        regdst = exmem.regdst.out()
        alu = exmem.alu.out()
        is_load = exmem.is_load.out()
        is_store = exmem.is_store.out()
        addr = exmem.addr.out()
        size = exmem.size.out()
        wdata = exmem.wdata.out()

        # Combinational.
        load32 = mem_rdata.trunc(width=32)

        # WB-store to MEM-load forwarding (handles the common aligned cases).
        # Note: loads are word (4B); stores may be 4B or 8B.
        if wb_store_valid & is_load:
            # Exact word-aligned match.
            if (wb_store_size == 4) & (wb_store_addr == addr):
                load32 = wb_store_wdata.trunc(width=32)
            # Double-word store overlap (addr or addr+4).
            if (wb_store_size == 8) & (wb_store_addr == addr):
                load32 = wb_store_wdata.trunc(width=32)
            if (wb_store_size == 8) & ((wb_store_addr + 4) == addr):
                load32 = wb_store_wdata[32:64]

        load64 = load32.sext(width=64)
        mem_val = alu
        if is_load:
            mem_val = load64
        if is_store:
            mem_val = 0

        # Pipeline regs: MEM/WB.
        memwb.pc.set(pc, when=do_mem)
        memwb.window.set(window, when=do_mem)
        memwb.op.set(op, when=do_mem)
        memwb.len_bytes.set(len_bytes, when=do_mem)
        memwb.regdst.set(regdst, when=do_mem)
        memwb.value.set(mem_val, when=do_mem)
        memwb.is_store.set(is_store, when=do_mem)
        memwb.size.set(size, when=do_mem)
        memwb.addr.set(addr, when=do_mem)
        memwb.wdata.set(wdata, when=do_mem)

    return mem_val
