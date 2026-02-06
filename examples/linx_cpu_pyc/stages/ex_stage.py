from __future__ import annotations

from pycircuit import Circuit, Wire, jit_inline

from ..isa import (
    OP_ADDTPC,
    OP_ADDI,
    OP_ADDIW,
    OP_ADDW,
    OP_ANDW,
    OP_BSTART_STD_CALL,
    OP_CMP_EQ,
    OP_C_BSTART_COND,
    OP_C_BSTART_STD,
    OP_CSEL,
    OP_C_LWI,
    OP_C_MOVI,
    OP_C_MOVR,
    OP_C_SETC_EQ,
    OP_C_SETC_TGT,
    OP_C_SETRET,
    OP_C_SWI,
    OP_HL_LUI,
    OP_LWI,
    OP_ORW,
    OP_SDI,
    OP_SUBI,
    OP_SWI,
    OP_XORW,
    REG_INVALID,
)
from ..pipeline import ExMemRegs, IdExRegs
from ..util import Consts


@jit_inline
def build_ex_stage(
    m: Circuit,
    *,
    do_ex: Wire,
    idex: IdExRegs,
    exmem: ExMemRegs,
    consts: Consts,
    # Forwarding sources (from MEM and WB stages).
    mem_fwd_valid: Wire,
    mem_fwd_regdst: Wire,
    mem_fwd_value: Wire,
    wb_fwd_valid: Wire,
    wb_fwd_regdst: Wire,
    wb_fwd_value: Wire,
    # Forwarded T/U stack views for this EX stage (after applying older in-flight
    # pushes/clears from WB+MEM).
    t0_fwd: Wire,
    t1_fwd: Wire,
    t2_fwd: Wire,
    t3_fwd: Wire,
    u0_fwd: Wire,
    u1_fwd: Wire,
    u2_fwd: Wire,
    u3_fwd: Wire,
) -> None:
    with m.scope("EX"):
        z1 = consts.zero1
        z4 = consts.zero4
        z64 = consts.zero64

        # Stage inputs.
        pc = idex.pc.out()
        window = idex.window.out()
        op = idex.op.out()
        len_bytes = idex.len_bytes.out()
        regdst = idex.regdst.out()
        srcl = idex.srcl.out()
        srcr = idex.srcr.out()
        srcp = idex.srcp.out()
        srcl_val = idex.srcl_val.out()
        srcr_val = idex.srcr_val.out()
        srcp_val = idex.srcp_val.out()
        imm = idex.imm.out()

        # Operand forwarding (priority: MEM over WB).
        can_fwd_mem = mem_fwd_valid & (mem_fwd_regdst != REG_INVALID) & (mem_fwd_regdst != 0)
        can_fwd_wb = wb_fwd_valid & (wb_fwd_regdst != REG_INVALID) & (wb_fwd_regdst != 0)

        if can_fwd_mem & (mem_fwd_regdst == srcl):
            srcl_val = mem_fwd_value
        if can_fwd_wb & (wb_fwd_regdst == srcl) & (~(can_fwd_mem & (mem_fwd_regdst == srcl))):
            srcl_val = wb_fwd_value

        if can_fwd_mem & (mem_fwd_regdst == srcr):
            srcr_val = mem_fwd_value
        if can_fwd_wb & (wb_fwd_regdst == srcr) & (~(can_fwd_mem & (mem_fwd_regdst == srcr))):
            srcr_val = wb_fwd_value

        if can_fwd_mem & (mem_fwd_regdst == srcp):
            srcp_val = mem_fwd_value
        if can_fwd_wb & (wb_fwd_regdst == srcp) & (~(can_fwd_mem & (mem_fwd_regdst == srcp))):
            srcp_val = wb_fwd_value

        # T/U stack bypass (codes: T0..T3 = 24..27, U0..U3 = 28..31).
        if srcl == 24:
            srcl_val = t0_fwd
        if srcl == 25:
            srcl_val = t1_fwd
        if srcl == 26:
            srcl_val = t2_fwd
        if srcl == 27:
            srcl_val = t3_fwd
        if srcl == 28:
            srcl_val = u0_fwd
        if srcl == 29:
            srcl_val = u1_fwd
        if srcl == 30:
            srcl_val = u2_fwd
        if srcl == 31:
            srcl_val = u3_fwd

        if srcr == 24:
            srcr_val = t0_fwd
        if srcr == 25:
            srcr_val = t1_fwd
        if srcr == 26:
            srcr_val = t2_fwd
        if srcr == 27:
            srcr_val = t3_fwd
        if srcr == 28:
            srcr_val = u0_fwd
        if srcr == 29:
            srcr_val = u1_fwd
        if srcr == 30:
            srcr_val = u2_fwd
        if srcr == 31:
            srcr_val = u3_fwd

        if srcp == 24:
            srcp_val = t0_fwd
        if srcp == 25:
            srcp_val = t1_fwd
        if srcp == 26:
            srcp_val = t2_fwd
        if srcp == 27:
            srcp_val = t3_fwd
        if srcp == 28:
            srcp_val = u0_fwd
        if srcp == 29:
            srcp_val = u1_fwd
        if srcp == 30:
            srcp_val = u2_fwd
        if srcp == 31:
            srcp_val = u3_fwd

        op_c_bstart_std = op == OP_C_BSTART_STD
        op_c_bstart_cond = op == OP_C_BSTART_COND
        op_bstart_std_call = op == OP_BSTART_STD_CALL
        op_c_movr = op == OP_C_MOVR
        op_c_movi = op == OP_C_MOVI
        op_c_setret = op == OP_C_SETRET
        op_c_setc_eq = op == OP_C_SETC_EQ
        op_c_setc_tgt = op == OP_C_SETC_TGT
        op_addtpc = op == OP_ADDTPC
        op_addi = op == OP_ADDI
        op_subi = op == OP_SUBI
        op_addiw = op == OP_ADDIW
        op_addw = op == OP_ADDW
        op_orw = op == OP_ORW
        op_andw = op == OP_ANDW
        op_xorw = op == OP_XORW
        op_cmp_eq = op == OP_CMP_EQ
        op_csel = op == OP_CSEL
        op_hl_lui = op == OP_HL_LUI
        op_lwi = op == OP_LWI
        op_c_lwi = op == OP_C_LWI
        op_swi = op == OP_SWI
        op_c_swi = op == OP_C_SWI
        op_sdi = op == OP_SDI

        off = imm.shl(amount=2)

        alu = z64
        is_load = z1
        is_store = z1
        size = z4
        addr = z64
        wdata = z64

        # Block markers: forward imm through ALU (used for control-state updates in WB).
        if op_c_bstart_std | op_c_bstart_cond | op_bstart_std_call:
            alu = imm
            is_load = z1
            is_store = z1
            size = z4
            addr = z64
            wdata = z64

        # MOVR: pass-through.
        if op_c_movr:
            alu = srcl_val
            is_load = z1
            is_store = z1
            size = z4
            addr = z64
            wdata = z64

        # MOVI: immediate.
        if op_c_movi:
            alu = imm
            is_load = z1
            is_store = z1
            size = z4
            addr = z64
            wdata = z64

        # SETRET: ra = PC + off (off already shifted by 1 in decode).
        if op_c_setret:
            alu = pc + imm
            is_load = z1
            is_store = z1
            size = z4
            addr = z64
            wdata = z64

        # SETC.EQ / SETC.TGT: internal control state updates (committed in WB).
        setc_eq = z64
        if srcl_val == srcr_val:
            setc_eq = 1
        if op_c_setc_eq:
            alu = setc_eq
            is_load = z1
            is_store = z1
            size = z4
            addr = z64
            wdata = z64
        if op_c_setc_tgt:
            alu = srcl_val
            is_load = z1
            is_store = z1
            size = z4
            addr = z64
            wdata = z64

        # ADDTPC: PC + (imm<<12) (imm already shifted by 12 in decode).
        pc_page = pc & 0xFFFF_FFFF_FFFF_F000
        if op_addtpc:
            alu = pc_page + imm
            is_load = z1
            is_store = z1
            size = z4
            addr = z64
            wdata = z64

        # ADDI/SUBI/ADDIW: srcl +/- imm.
        if op_addi:
            alu = srcl_val + imm
            is_load = z1
            is_store = z1
            size = z4
            addr = z64
            wdata = z64
        subi = srcl_val + ((~imm) + 1)
        if op_subi:
            alu = subi
            is_load = z1
            is_store = z1
            size = z4
            addr = z64
            wdata = z64
        addiw = (srcl_val.trunc(width=32) + imm.trunc(width=32)).sext(width=64)
        if op_addiw:
            alu = addiw
            is_load = z1
            is_store = z1
            size = z4
            addr = z64
            wdata = z64

        # ADDW/ORW/ANDW/XORW: 32-bit ops with sign-extend.
        addw = (srcl_val.trunc(width=32) + srcr_val.trunc(width=32)).sext(width=64)
        orw = (srcl_val.trunc(width=32) | srcr_val.trunc(width=32)).sext(width=64)
        andw = (srcl_val.trunc(width=32) & srcr_val.trunc(width=32)).sext(width=64)
        xorw = (srcl_val.trunc(width=32) ^ srcr_val.trunc(width=32)).sext(width=64)
        if op_addw:
            alu = addw
            is_load = z1
            is_store = z1
            size = z4
            addr = z64
            wdata = z64
        if op_orw:
            alu = orw
            is_load = z1
            is_store = z1
            size = z4
            addr = z64
            wdata = z64
        if op_andw:
            alu = andw
            is_load = z1
            is_store = z1
            size = z4
            addr = z64
            wdata = z64
        if op_xorw:
            alu = xorw
            is_load = z1
            is_store = z1
            size = z4
            addr = z64
            wdata = z64

        # CMP_EQ: (srcl == srcr) ? 1 : 0
        cmp = z64
        if srcl_val == srcr_val:
            cmp = 1
        if op_cmp_eq:
            alu = cmp
            is_load = z1
            is_store = z1
            size = z4
            addr = z64
            wdata = z64

        # HL.LUI: imm.
        if op_hl_lui:
            alu = imm
            is_load = z1
            is_store = z1
            size = z4
            addr = z64
            wdata = z64

        # CSEL: (srcp != 0) ? srcr : srcl.
        csel_val = srcl_val
        if srcp_val != 0:
            csel_val = srcr_val
        if op_csel:
            alu = csel_val
            is_load = z1
            is_store = z1
            size = z4
            addr = z64
            wdata = z64

        # LWI / C.LWI: load word, address = srcl + (imm << 2)
        is_lwi = op_lwi | op_c_lwi
        lwi_addr = srcl_val + off
        if is_lwi:
            alu = z64
            is_load = 1
            is_store = z1
            size = 4
            addr = lwi_addr
            wdata = z64

        # SWI / C.SWI: store word (4 bytes)
        store_addr = srcl_val + off
        store_data = srcr_val
        if op_swi:
            store_addr = srcr_val + off
            store_data = srcl_val
        if op_swi | op_c_swi:
            alu = z64
            is_load = z1
            is_store = 1
            size = 4
            addr = store_addr
            wdata = store_data

        # SDI: store double word (8 bytes), addr = SrcR + (simm12<<3)
        sdi_off = imm.shl(amount=3)
        sdi_addr = srcr_val + sdi_off
        if op_sdi:
            alu = z64
            is_load = z1
            is_store = 1
            size = 8
            addr = sdi_addr
            wdata = srcl_val

        # Pipeline regs: EX/MEM.
        exmem.pc.set(pc, when=do_ex)
        exmem.window.set(window, when=do_ex)
        exmem.op.set(op, when=do_ex)
        exmem.len_bytes.set(len_bytes, when=do_ex)
        exmem.regdst.set(regdst, when=do_ex)
        exmem.alu.set(alu, when=do_ex)
        exmem.is_load.set(is_load, when=do_ex)
        exmem.is_store.set(is_store, when=do_ex)
        exmem.size.set(size, when=do_ex)
        exmem.addr.set(addr, when=do_ex)
        exmem.wdata.set(wdata, when=do_ex)
