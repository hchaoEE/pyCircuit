from __future__ import annotations

from pycircuit import Circuit

# This design is written in "JIT mode": `pycircuit.cli emit` will compile
# `build(m: Circuit, ...)` via the AST/SCF frontend.

from examples.linx_cpu_pyc.isa import (
    BK_FALL,
    OP_BSTART_STD_CALL,
    OP_C_BSTART_COND,
    OP_C_BSTART_STD,
    OP_C_LWI,
    OP_EBREAK,
    OP_INVALID,
    REG_INVALID,
    ST_IF,
    ST_WB,
)
from examples.linx_cpu_pyc.memory import build_byte_mem
from examples.linx_cpu_pyc.pipeline import CoreState, ExMemRegs, IdExRegs, IfIdRegs, MemWbRegs, RegFiles
from examples.linx_cpu_pyc.regfile import make_gpr, make_regs
from examples.linx_cpu_pyc.stages.ex_stage import build_ex_stage
from examples.linx_cpu_pyc.stages.id_stage import build_id_stage
from examples.linx_cpu_pyc.stages.if_stage import build_if_stage
from examples.linx_cpu_pyc.stages.mem_stage import build_mem_stage
from examples.linx_cpu_pyc.stages.wb_stage import build_wb_stage
from examples.linx_cpu_pyc.util import make_consts


def build(m: Circuit, *, mem_bytes: int = (1 << 20)) -> None:
    # --- ports ---
    clk = m.clock("clk")
    rst = m.reset("rst")

    boot_pc = m.input("boot_pc", width=64)
    boot_sp = m.input("boot_sp", width=64)

    consts = make_consts(m)

    # --- core state regs (named) ---
    with m.scope("state"):
        state = CoreState(
            pc=m.out("pc_fetch", clk=clk, rst=rst, width=64, init=boot_pc, en=consts.one1),
            br_kind=m.out("br_kind", clk=clk, rst=rst, width=2, init=BK_FALL, en=consts.one1),
            br_base_pc=m.out("br_base_pc", clk=clk, rst=rst, width=64, init=boot_pc, en=consts.one1),
            br_off=m.out("br_off", clk=clk, rst=rst, width=64, init=0, en=consts.one1),
            commit_cond=m.out("commit_cond", clk=clk, rst=rst, width=1, init=0, en=consts.one1),
            commit_tgt=m.out("commit_tgt", clk=clk, rst=rst, width=64, init=0, en=consts.one1),
            cycles=m.out("cycles", clk=clk, rst=rst, width=64, init=0, en=consts.one1),
            halted=m.out("halted", clk=clk, rst=rst, width=1, init=0, en=consts.one1),
        )

    with m.scope("ifid"):
        pipe_ifid = IfIdRegs(
            valid=m.out("valid", clk=clk, rst=rst, width=1, init=0, en=consts.one1),
            pc=m.out("pc", clk=clk, rst=rst, width=64, init=0, en=consts.one1),
            window=m.out("window", clk=clk, rst=rst, width=64, init=0, en=consts.one1),
        )

    with m.scope("idex"):
        pipe_idex = IdExRegs(
            valid=m.out("valid", clk=clk, rst=rst, width=1, init=0, en=consts.one1),
            pc=m.out("pc", clk=clk, rst=rst, width=64, init=0, en=consts.one1),
            window=m.out("window", clk=clk, rst=rst, width=64, init=0, en=consts.one1),
            op=m.out("op", clk=clk, rst=rst, width=6, init=0, en=consts.one1),
            len_bytes=m.out("len_bytes", clk=clk, rst=rst, width=3, init=0, en=consts.one1),
            regdst=m.out("regdst", clk=clk, rst=rst, width=6, init=REG_INVALID, en=consts.one1),
            srcl=m.out("srcl", clk=clk, rst=rst, width=6, init=REG_INVALID, en=consts.one1),
            srcr=m.out("srcr", clk=clk, rst=rst, width=6, init=REG_INVALID, en=consts.one1),
            srcp=m.out("srcp", clk=clk, rst=rst, width=6, init=REG_INVALID, en=consts.one1),
            imm=m.out("imm", clk=clk, rst=rst, width=64, init=0, en=consts.one1),
            srcl_val=m.out("srcl_val", clk=clk, rst=rst, width=64, init=0, en=consts.one1),
            srcr_val=m.out("srcr_val", clk=clk, rst=rst, width=64, init=0, en=consts.one1),
            srcp_val=m.out("srcp_val", clk=clk, rst=rst, width=64, init=0, en=consts.one1),
        )

    with m.scope("exmem"):
        pipe_exmem = ExMemRegs(
            valid=m.out("valid", clk=clk, rst=rst, width=1, init=0, en=consts.one1),
            pc=m.out("pc", clk=clk, rst=rst, width=64, init=0, en=consts.one1),
            window=m.out("window", clk=clk, rst=rst, width=64, init=0, en=consts.one1),
            op=m.out("op", clk=clk, rst=rst, width=6, init=0, en=consts.one1),
            len_bytes=m.out("len_bytes", clk=clk, rst=rst, width=3, init=0, en=consts.one1),
            regdst=m.out("regdst", clk=clk, rst=rst, width=6, init=REG_INVALID, en=consts.one1),
            alu=m.out("alu", clk=clk, rst=rst, width=64, init=0, en=consts.one1),
            is_load=m.out("is_load", clk=clk, rst=rst, width=1, init=0, en=consts.one1),
            is_store=m.out("is_store", clk=clk, rst=rst, width=1, init=0, en=consts.one1),
            size=m.out("size", clk=clk, rst=rst, width=4, init=0, en=consts.one1),
            addr=m.out("addr", clk=clk, rst=rst, width=64, init=0, en=consts.one1),
            wdata=m.out("wdata", clk=clk, rst=rst, width=64, init=0, en=consts.one1),
        )

    with m.scope("memwb"):
        pipe_memwb = MemWbRegs(
            valid=m.out("valid", clk=clk, rst=rst, width=1, init=0, en=consts.one1),
            pc=m.out("pc", clk=clk, rst=rst, width=64, init=0, en=consts.one1),
            window=m.out("window", clk=clk, rst=rst, width=64, init=0, en=consts.one1),
            op=m.out("op", clk=clk, rst=rst, width=6, init=0, en=consts.one1),
            len_bytes=m.out("len_bytes", clk=clk, rst=rst, width=3, init=0, en=consts.one1),
            regdst=m.out("regdst", clk=clk, rst=rst, width=6, init=REG_INVALID, en=consts.one1),
            value=m.out("value", clk=clk, rst=rst, width=64, init=0, en=consts.one1),
            is_store=m.out("is_store", clk=clk, rst=rst, width=1, init=0, en=consts.one1),
            size=m.out("size", clk=clk, rst=rst, width=4, init=0, en=consts.one1),
            addr=m.out("addr", clk=clk, rst=rst, width=64, init=0, en=consts.one1),
            wdata=m.out("wdata", clk=clk, rst=rst, width=64, init=0, en=consts.one1),
        )

    # --- register files ---
    with m.scope("gpr"):
        gpr = make_gpr(m, clk, rst, boot_sp=boot_sp, en=consts.one1)
    with m.scope("t"):
        t = make_regs(m, clk, rst, count=4, width=64, init=consts.zero64, en=consts.one1)
    with m.scope("u"):
        u = make_regs(m, clk, rst, count=4, width=64, init=consts.zero64, en=consts.one1)

    rf = RegFiles(gpr=gpr, t=t, u=u)

    # --- pipeline control ---
    halt_set = pipe_memwb.valid.out() & (~state.halted) & ((pipe_memwb.op == OP_EBREAK) | (pipe_memwb.op == OP_INVALID))
    stop = state.halted | halt_set
    active = ~stop

    do_wb = pipe_memwb.valid.out() & active

    wb = build_wb_stage(m, do_wb=do_wb, state=state, memwb=pipe_memwb, rf=rf)
    flush = wb.valid

    # --- unified byte memory (instruction + data) ---
    mem_load = active & (~flush) & pipe_exmem.valid.out() & pipe_exmem.is_load.out()

    mem_raddr = state.pc.out()
    if mem_load:
        mem_raddr = pipe_exmem.addr.out()

    mem_wvalid = do_wb & pipe_memwb.is_store.out()
    mem_waddr = pipe_memwb.addr
    mem_wdata = pipe_memwb.wdata
    mem_wstrb = consts.zero8
    if pipe_memwb.size.out() == 8:
        mem_wstrb = 0xFF
    if pipe_memwb.size.out() == 4:
        mem_wstrb = 0x0F

    mem_rdata = build_byte_mem(
        m,
        clk,
        rst,
        raddr=mem_raddr,
        wvalid=mem_wvalid,
        waddr=mem_waddr,
        wdata=mem_wdata,
        wstrb=mem_wstrb,
        depth_bytes=mem_bytes,
        name="mem",
    )

    # --- stages ---
    do_if = active & (~flush) & (~mem_load)
    do_id = active & (~flush) & pipe_ifid.valid.out()
    do_ex = active & (~flush) & pipe_idex.valid.out()
    do_mem = active & (~flush) & pipe_exmem.valid.out()

    build_if_stage(m, do_if=do_if, ifid=pipe_ifid, fetch_pc=state.pc.out(), mem_rdata=mem_rdata)
    build_id_stage(
        m,
        do_id=do_id,
        ifid=pipe_ifid,
        idex=pipe_idex,
        rf=rf,
        consts=consts,
        wb_fwd_valid=do_wb & (~pipe_memwb.is_store.out()),
        wb_fwd_regdst=pipe_memwb.regdst.out(),
        wb_fwd_value=pipe_memwb.value.out(),
    )

    wb_store_fwd = do_wb & pipe_memwb.is_store.out()
    mem_fwd_value = build_mem_stage(
        m,
        do_mem=do_mem,
        exmem=pipe_exmem,
        memwb=pipe_memwb,
        mem_rdata=mem_rdata,
        wb_store_valid=wb_store_fwd,
        wb_store_addr=pipe_memwb.addr.out(),
        wb_store_size=pipe_memwb.size.out(),
        wb_store_wdata=pipe_memwb.wdata.out(),
    )

    # --- T/U stack bypass for EX stage ---
    # The ISA models T/U as small stacks (shift registers). Updates commit in WB,
    # but younger instructions must see the effects of older in-flight pushes.
    #
    # This computes a forwarded view of the stacks after applying pending
    # clear/push effects from the WB and MEM stages (oldest -> youngest).
    t0 = rf.t[0].out()
    t1 = rf.t[1].out()
    t2 = rf.t[2].out()
    t3 = rf.t[3].out()
    u0 = rf.u[0].out()
    u1 = rf.u[1].out()
    u2 = rf.u[2].out()
    u3 = rf.u[3].out()

    wb_op = pipe_memwb.op.out()
    wb_regdst = pipe_memwb.regdst.out()
    wb_value = pipe_memwb.value.out()
    wb_is_store = pipe_memwb.is_store.out()
    wb_do_reg_write = do_wb & (~wb_is_store) & (wb_regdst != REG_INVALID)
    wb_is_start = (wb_op == OP_C_BSTART_STD) | (wb_op == OP_C_BSTART_COND) | (wb_op == OP_BSTART_STD_CALL)
    wb_clear = do_wb & wb_is_start
    wb_push_t = do_wb & ((wb_op == OP_C_LWI) | (wb_do_reg_write & (wb_regdst == 31)))
    wb_push_u = do_wb & (wb_do_reg_write & (wb_regdst == 30))

    t0_fwd = t0
    t1_fwd = t1
    t2_fwd = t2
    t3_fwd = t3
    u0_fwd = u0
    u1_fwd = u1
    u2_fwd = u2
    u3_fwd = u3

    if wb_clear:
        t0_fwd = 0
        t1_fwd = 0
        t2_fwd = 0
        t3_fwd = 0
        u0_fwd = 0
        u1_fwd = 0
        u2_fwd = 0
        u3_fwd = 0

    if wb_push_t:
        t3_fwd = t2_fwd
        t2_fwd = t1_fwd
        t1_fwd = t0_fwd
        t0_fwd = wb_value

    if wb_push_u:
        u3_fwd = u2_fwd
        u2_fwd = u1_fwd
        u1_fwd = u0_fwd
        u0_fwd = wb_value

    mem_op = pipe_exmem.op.out()
    mem_regdst = pipe_exmem.regdst.out()
    mem_is_store = pipe_exmem.is_store.out()
    mem_pending = active & pipe_exmem.valid.out()
    mem_do_reg_write = mem_pending & (~mem_is_store) & (mem_regdst != REG_INVALID)
    mem_is_start = (mem_op == OP_C_BSTART_STD) | (mem_op == OP_C_BSTART_COND) | (mem_op == OP_BSTART_STD_CALL)
    mem_clear = mem_pending & mem_is_start
    mem_push_t = mem_pending & ((mem_op == OP_C_LWI) | (mem_do_reg_write & (mem_regdst == 31)))
    mem_push_u = mem_pending & (mem_do_reg_write & (mem_regdst == 30))
    mem_value = mem_fwd_value

    if mem_clear:
        t0_fwd = 0
        t1_fwd = 0
        t2_fwd = 0
        t3_fwd = 0
        u0_fwd = 0
        u1_fwd = 0
        u2_fwd = 0
        u3_fwd = 0

    if mem_push_t:
        t3_fwd = t2_fwd
        t2_fwd = t1_fwd
        t1_fwd = t0_fwd
        t0_fwd = mem_value

    if mem_push_u:
        u3_fwd = u2_fwd
        u2_fwd = u1_fwd
        u1_fwd = u0_fwd
        u0_fwd = mem_value

    build_ex_stage(
        m,
        do_ex=do_ex,
        idex=pipe_idex,
        exmem=pipe_exmem,
        consts=consts,
        mem_fwd_valid=pipe_exmem.valid.out() & (~pipe_exmem.is_store.out()),
        mem_fwd_regdst=pipe_exmem.regdst.out(),
        mem_fwd_value=mem_fwd_value,
        wb_fwd_valid=pipe_memwb.valid.out() & (~pipe_memwb.is_store.out()),
        wb_fwd_regdst=pipe_memwb.regdst.out(),
        wb_fwd_value=pipe_memwb.value.out(),
        t0_fwd=t0_fwd,
        t1_fwd=t1_fwd,
        t2_fwd=t2_fwd,
        t3_fwd=t3_fwd,
        u0_fwd=u0_fwd,
        u1_fwd=u1_fwd,
        u2_fwd=u2_fwd,
        u3_fwd=u3_fwd,
    )

    # --- pipeline valid shift + fetch PC ---
    ifid_v_next = do_if
    idex_v_next = pipe_ifid.valid.out()
    exmem_v_next = pipe_idex.valid.out()
    memwb_v_next = pipe_exmem.valid.out()
    if flush:
        ifid_v_next = 0
        idex_v_next = 0
        exmem_v_next = 0
        memwb_v_next = 0

    pipe_ifid.valid.set(ifid_v_next, when=~stop)
    pipe_idex.valid.set(idex_v_next, when=~stop)
    pipe_exmem.valid.set(exmem_v_next, when=~stop)
    pipe_memwb.valid.set(memwb_v_next, when=~stop)

    # Fetch PC update (variable-length fetch via fast predecode).
    fetch_pc = state.pc.out()
    insn16 = mem_rdata.trunc(width=16)
    low4 = insn16[0:4]
    is_hl = low4 == 0xE
    is32 = insn16[0]
    fetch_len = 2
    if (~is_hl) & is32:
        fetch_len = 4
    if is_hl:
        fetch_len = 6

    pc_next = fetch_pc
    if wb.valid:
        pc_next = wb.pc
    if do_if:
        pc_next = fetch_pc + fetch_len
    state.pc.set(pc_next, when=~stop)

    # Halt latch + cycle counter (always increments; TB stops on halt).
    state.halted.set(1, when=halt_set)
    state.cycles.set(state.cycles.out() + 1)

    # --- outputs ---
    stage = m.const(ST_IF, width=3)
    if pipe_memwb.valid.out():
        stage = ST_WB
    m.output("halted", state.halted)

    pc_out = state.pc.out()
    if stage == ST_WB:
        pc_out = pipe_memwb.pc.out()
    m.output("pc", pc_out)
    m.output("stage", stage)
    m.output("cycles", state.cycles)
    m.output("a0", rf.gpr[2])
    m.output("a1", rf.gpr[3])
    m.output("ra", rf.gpr[10])
    m.output("sp", rf.gpr[1])
    m.output("br_kind", state.br_kind)
    # Debug/trace hooks (stable, optional consumers).
    m.output("if_window", pipe_memwb.window)
    m.output("wb_op", pipe_memwb.op)
    m.output("wb_regdst", pipe_memwb.regdst)
    m.output("wb_value", pipe_memwb.value)
    m.output("commit_cond", state.commit_cond)
    m.output("commit_tgt", state.commit_tgt)


# Preserve the historical top/module name expected by existing testbenches.
build.__pycircuit_name__ = "linx_cpu_pyc"
