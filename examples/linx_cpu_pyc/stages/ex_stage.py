from __future__ import annotations

from dataclasses import dataclass

from pycircuit import Circuit, Wire

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
)
from ..pipeline import ExMemRegs, IdExRegs
from ..util import Consts, latch_many


@dataclass(frozen=True)
class ExBundle:
    alu: Wire
    is_load: Wire
    is_store: Wire
    size: Wire
    addr: Wire
    wdata: Wire


def _ex_apply(m: Circuit, ex: ExBundle, cond: Wire, *, alu: Wire, is_load: Wire, is_store: Wire, size: Wire, addr: Wire, wdata: Wire) -> ExBundle:
    return ExBundle(
        alu=cond.select(alu, ex.alu),
        is_load=cond.select(is_load, ex.is_load),
        is_store=cond.select(is_store, ex.is_store),
        size=cond.select(size, ex.size),
        addr=cond.select(addr, ex.addr),
        wdata=cond.select(wdata, ex.wdata),
    )


def build_ex_stage(m: Circuit, *, do_ex: Wire, pc: Wire, idex: IdExRegs, exmem: ExMemRegs, consts: Consts) -> None:
    c = m.const_wire

    op_c_bstart_std = idex.op.eq(c(OP_C_BSTART_STD, width=6))
    op_c_bstart_cond = idex.op.eq(c(OP_C_BSTART_COND, width=6))
    op_bstart_std_call = idex.op.eq(c(OP_BSTART_STD_CALL, width=6))
    op_c_movr = idex.op.eq(c(OP_C_MOVR, width=6))
    op_c_movi = idex.op.eq(c(OP_C_MOVI, width=6))
    op_c_setret = idex.op.eq(c(OP_C_SETRET, width=6))
    op_c_setc_eq = idex.op.eq(c(OP_C_SETC_EQ, width=6))
    op_c_setc_tgt = idex.op.eq(c(OP_C_SETC_TGT, width=6))
    op_addtpc = idex.op.eq(c(OP_ADDTPC, width=6))
    op_addi = idex.op.eq(c(OP_ADDI, width=6))
    op_subi = idex.op.eq(c(OP_SUBI, width=6))
    op_addiw = idex.op.eq(c(OP_ADDIW, width=6))
    op_addw = idex.op.eq(c(OP_ADDW, width=6))
    op_orw = idex.op.eq(c(OP_ORW, width=6))
    op_andw = idex.op.eq(c(OP_ANDW, width=6))
    op_xorw = idex.op.eq(c(OP_XORW, width=6))
    op_cmp_eq = idex.op.eq(c(OP_CMP_EQ, width=6))
    op_csel = idex.op.eq(c(OP_CSEL, width=6))
    op_hl_lui = idex.op.eq(c(OP_HL_LUI, width=6))
    op_lwi = idex.op.eq(c(OP_LWI, width=6))
    op_c_lwi = idex.op.eq(c(OP_C_LWI, width=6))
    op_swi = idex.op.eq(c(OP_SWI, width=6))
    op_c_swi = idex.op.eq(c(OP_C_SWI, width=6))
    op_sdi = idex.op.eq(c(OP_SDI, width=6))

    off = idex.imm.shl(amount=2)

    ex = ExBundle(alu=consts.zero64, is_load=consts.zero1, is_store=consts.zero1, size=consts.zero3, addr=consts.zero64, wdata=consts.zero64)

    # Block markers: forward imm through ALU (used for control-state updates in WB).
    ex = _ex_apply(m, ex, op_c_bstart_std | op_c_bstart_cond | op_bstart_std_call, alu=idex.imm, is_load=consts.zero1, is_store=consts.zero1, size=consts.zero3, addr=consts.zero64, wdata=consts.zero64)

    # MOVR: pass-through.
    ex = _ex_apply(m, ex, op_c_movr, alu=idex.srcl_val, is_load=consts.zero1, is_store=consts.zero1, size=consts.zero3, addr=consts.zero64, wdata=consts.zero64)

    # MOVI: immediate.
    ex = _ex_apply(m, ex, op_c_movi, alu=idex.imm, is_load=consts.zero1, is_store=consts.zero1, size=consts.zero3, addr=consts.zero64, wdata=consts.zero64)

    # SETRET: ra = PC + off (off already shifted by 1 in decode).
    ex = _ex_apply(m, ex, op_c_setret, alu=pc + idex.imm, is_load=consts.zero1, is_store=consts.zero1, size=consts.zero3, addr=consts.zero64, wdata=consts.zero64)

    # SETC.EQ / SETC.TGT: internal control state updates (committed in WB).
    setc_eq = idex.srcl_val.eq(idex.srcr_val).select(consts.one64, consts.zero64)
    ex = _ex_apply(m, ex, op_c_setc_eq, alu=setc_eq, is_load=consts.zero1, is_store=consts.zero1, size=consts.zero3, addr=consts.zero64, wdata=consts.zero64)
    ex = _ex_apply(m, ex, op_c_setc_tgt, alu=idex.srcl_val, is_load=consts.zero1, is_store=consts.zero1, size=consts.zero3, addr=consts.zero64, wdata=consts.zero64)

    # ADDTPC: PC + (imm<<12) (imm already shifted by 12 in decode).
    pc_page = pc & c(0xFFFF_FFFF_FFFF_F000, width=64)
    ex = _ex_apply(m, ex, op_addtpc, alu=pc_page + idex.imm, is_load=consts.zero1, is_store=consts.zero1, size=consts.zero3, addr=consts.zero64, wdata=consts.zero64)

    # ADDI/SUBI/ADDIW: srcl +/- imm.
    ex = _ex_apply(m, ex, op_addi, alu=idex.srcl_val + idex.imm, is_load=consts.zero1, is_store=consts.zero1, size=consts.zero3, addr=consts.zero64, wdata=consts.zero64)
    subi = idex.srcl_val + ((~idex.imm) + consts.one64)
    ex = _ex_apply(m, ex, op_subi, alu=subi, is_load=consts.zero1, is_store=consts.zero1, size=consts.zero3, addr=consts.zero64, wdata=consts.zero64)
    addiw = (idex.srcl_val.trunc(width=32) + idex.imm.trunc(width=32)).sext(width=64)
    ex = _ex_apply(m, ex, op_addiw, alu=addiw, is_load=consts.zero1, is_store=consts.zero1, size=consts.zero3, addr=consts.zero64, wdata=consts.zero64)

    # ADDW/ORW/ANDW/XORW: 32-bit ops with sign-extend.
    addw = (idex.srcl_val.trunc(width=32) + idex.srcr_val.trunc(width=32)).sext(width=64)
    orw = (idex.srcl_val.trunc(width=32) | idex.srcr_val.trunc(width=32)).sext(width=64)
    andw = (idex.srcl_val.trunc(width=32) & idex.srcr_val.trunc(width=32)).sext(width=64)
    xorw = (idex.srcl_val.trunc(width=32) ^ idex.srcr_val.trunc(width=32)).sext(width=64)
    ex = _ex_apply(m, ex, op_addw, alu=addw, is_load=consts.zero1, is_store=consts.zero1, size=consts.zero3, addr=consts.zero64, wdata=consts.zero64)
    ex = _ex_apply(m, ex, op_orw, alu=orw, is_load=consts.zero1, is_store=consts.zero1, size=consts.zero3, addr=consts.zero64, wdata=consts.zero64)
    ex = _ex_apply(m, ex, op_andw, alu=andw, is_load=consts.zero1, is_store=consts.zero1, size=consts.zero3, addr=consts.zero64, wdata=consts.zero64)
    ex = _ex_apply(m, ex, op_xorw, alu=xorw, is_load=consts.zero1, is_store=consts.zero1, size=consts.zero3, addr=consts.zero64, wdata=consts.zero64)

    # CMP_EQ: (srcl == srcr) ? 1 : 0
    cmp = idex.srcl_val.eq(idex.srcr_val).select(consts.one64, consts.zero64)
    ex = _ex_apply(m, ex, op_cmp_eq, alu=cmp, is_load=consts.zero1, is_store=consts.zero1, size=consts.zero3, addr=consts.zero64, wdata=consts.zero64)

    # HL.LUI: imm.
    ex = _ex_apply(m, ex, op_hl_lui, alu=idex.imm, is_load=consts.zero1, is_store=consts.zero1, size=consts.zero3, addr=consts.zero64, wdata=consts.zero64)

    # CSEL: (srcp != 0) ? srcr : srcl.
    srcp_nz = ~idex.srcp_val.eq(consts.zero64)
    csel_val = srcp_nz.select(idex.srcr_val, idex.srcl_val)
    ex = _ex_apply(m, ex, op_csel, alu=csel_val, is_load=consts.zero1, is_store=consts.zero1, size=consts.zero3, addr=consts.zero64, wdata=consts.zero64)

    # LWI / C.LWI: load word, address = srcl + (imm << 2)
    is_lwi = op_lwi | op_c_lwi
    lwi_addr = idex.srcl_val + off
    ex = _ex_apply(m, ex, is_lwi, alu=consts.zero64, is_load=consts.one1, is_store=consts.zero1, size=c(4, width=3), addr=lwi_addr, wdata=consts.zero64)

    # SWI / C.SWI: store word (4 bytes)
    store_addr = op_swi.select(idex.srcr_val + off, idex.srcl_val + off)
    store_data = op_swi.select(idex.srcl_val, idex.srcr_val)
    ex = _ex_apply(m, ex, op_swi | op_c_swi, alu=consts.zero64, is_load=consts.zero1, is_store=consts.one1, size=c(4, width=3), addr=store_addr, wdata=store_data)

    # SDI: store double word (8 bytes), addr = SrcR + (simm12<<3)
    sdi_off = idex.imm.shl(amount=3)
    sdi_addr = idex.srcr_val + sdi_off
    ex = _ex_apply(m, ex, op_sdi, alu=consts.zero64, is_load=consts.zero1, is_store=consts.one1, size=c(8, width=3), addr=sdi_addr, wdata=idex.srcl_val)

    latch_many(
        m,
        do_ex,
        [
            (exmem.op, idex.op),
            (exmem.len_bytes, idex.len_bytes),
            (exmem.regdst, idex.regdst),
            (exmem.alu, ex.alu),
            (exmem.is_load, ex.is_load),
            (exmem.is_store, ex.is_store),
            (exmem.size, ex.size),
            (exmem.addr, ex.addr),
            (exmem.wdata, ex.wdata),
        ],
    )
