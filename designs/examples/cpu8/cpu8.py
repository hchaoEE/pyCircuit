"""
cpu8 -- 5-stage pipelined 8-bit CPU in pyCircuit.

Stages:  IF -> ID -> EX -> MEM -> WB

ISA (8-bit instruction word, 4 registers r0-r3):
  Bits [7:6] = opcode
  Bits [5:4] = rd  (destination register)
  Bits [3:2] = rs1 (source register 1)
  Bits [1:0] = rs2 (source register 2 / immediate)

  Opcodes:
    00  ADD   rd = rs1 + rs2
    01  SUB   rd = rs1 - rs2
    10  AND   rd = rs1 & rs2
    11  LDI   rd = {rs1_field, rs2_field} (load 4-bit immediate)

Features:
  - 4 x 8-bit general-purpose registers (r0..r3, r0 hardwired to 0)
  - Data forwarding from EX and MEM/WB to ID
  - Simple sequential PC (no branches, wraps at memory depth)
  - Instruction memory read in IF, data written back in WB
"""
from __future__ import annotations

from pycircuit import Circuit, compile, module, u

NUM_REGS = 4
REG_W = 8
INST_W = 8
PC_W = 8

OP_ADD = 0
OP_SUB = 1
OP_AND = 2
OP_LDI = 3


@module
def build(m: Circuit, mem_depth: int = 256) -> None:
    clk = m.clock("clk")
    rst = m.reset("rst")

    inst_in = m.input("inst_in", width=INST_W)

    # ------------------------------------------------------------------ #
    #  Register file (4 x 8-bit, r0 hardwired to 0)                      #
    # ------------------------------------------------------------------ #
    regs = [
        m.out(f"rf{i}", clk=clk, rst=rst, width=REG_W, init=u(REG_W, 0))
        for i in range(NUM_REGS)
    ]

    # ------------------------------------------------------------------ #
    #  Program counter                                                    #
    # ------------------------------------------------------------------ #
    pc = m.out("pc_q", clk=clk, rst=rst, width=PC_W, init=u(PC_W, 0))

    # ------------------------------------------------------------------ #
    #  Pipeline registers: IF/ID                                          #
    # ------------------------------------------------------------------ #
    ifid_valid = m.out("ifid_valid", clk=clk, rst=rst, width=1, init=u(1, 0))
    ifid_inst = m.out("ifid_inst", clk=clk, rst=rst, width=INST_W, init=u(INST_W, 0))
    ifid_pc = m.out("ifid_pc", clk=clk, rst=rst, width=PC_W, init=u(PC_W, 0))

    # ------------------------------------------------------------------ #
    #  Pipeline registers: ID/EX                                          #
    # ------------------------------------------------------------------ #
    idex_valid = m.out("idex_valid", clk=clk, rst=rst, width=1, init=u(1, 0))
    idex_op = m.out("idex_op", clk=clk, rst=rst, width=2, init=u(2, 0))
    idex_rd = m.out("idex_rd", clk=clk, rst=rst, width=2, init=u(2, 0))
    idex_src1 = m.out("idex_src1", clk=clk, rst=rst, width=REG_W, init=u(REG_W, 0))
    idex_src2 = m.out("idex_src2", clk=clk, rst=rst, width=REG_W, init=u(REG_W, 0))
    idex_imm = m.out("idex_imm", clk=clk, rst=rst, width=4, init=u(4, 0))

    # ------------------------------------------------------------------ #
    #  Pipeline registers: EX/MEM                                         #
    # ------------------------------------------------------------------ #
    exmem_valid = m.out("exmem_valid", clk=clk, rst=rst, width=1, init=u(1, 0))
    exmem_rd = m.out("exmem_rd", clk=clk, rst=rst, width=2, init=u(2, 0))
    exmem_result = m.out(
        "exmem_result", clk=clk, rst=rst, width=REG_W, init=u(REG_W, 0)
    )

    # ------------------------------------------------------------------ #
    #  Pipeline registers: MEM/WB                                         #
    # ------------------------------------------------------------------ #
    memwb_valid = m.out("memwb_valid", clk=clk, rst=rst, width=1, init=u(1, 0))
    memwb_rd = m.out("memwb_rd", clk=clk, rst=rst, width=2, init=u(2, 0))
    memwb_result = m.out(
        "memwb_result", clk=clk, rst=rst, width=REG_W, init=u(REG_W, 0)
    )

    # ================================================================== #
    #  STAGE 1 -- IF (Instruction Fetch)                                  #
    # ================================================================== #
    pc_val = pc.out()
    pc.set((pc_val + 1)[0:PC_W])

    ifid_valid.set(u(1, 1))
    ifid_inst.set(inst_in)
    ifid_pc.set(pc_val)

    # ================================================================== #
    #  STAGE 2 -- ID (Instruction Decode + Register Read + Forwarding)    #
    # ================================================================== #
    id_inst = ifid_inst.out()
    id_valid = ifid_valid.out()

    id_opcode = id_inst[6:8]
    id_rd = id_inst[4:6]
    id_rs1 = id_inst[2:4]
    id_rs2 = id_inst[0:2]
    id_imm = id_inst[0:4]

    # Register read: mux chain for rs1
    rs1_val = u(REG_W, 0)
    rs1_val = regs[1].out() if id_rs1 == u(2, 1) else rs1_val
    rs1_val = regs[2].out() if id_rs1 == u(2, 2) else rs1_val
    rs1_val = regs[3].out() if id_rs1 == u(2, 3) else rs1_val

    # Register read: mux chain for rs2
    rs2_val = u(REG_W, 0)
    rs2_val = regs[1].out() if id_rs2 == u(2, 1) else rs2_val
    rs2_val = regs[2].out() if id_rs2 == u(2, 2) else rs2_val
    rs2_val = regs[3].out() if id_rs2 == u(2, 3) else rs2_val

    # EX->ID forwarding
    ex_fwd_valid = exmem_valid.out()
    ex_fwd_rd = exmem_rd.out()
    ex_fwd_data = exmem_result.out()
    ex_fwd_rs1_hit = ex_fwd_valid & (ex_fwd_rd == id_rs1) & (id_rs1 != u(2, 0))
    ex_fwd_rs2_hit = ex_fwd_valid & (ex_fwd_rd == id_rs2) & (id_rs2 != u(2, 0))
    rs1_val = ex_fwd_data if ex_fwd_rs1_hit else rs1_val
    rs2_val = ex_fwd_data if ex_fwd_rs2_hit else rs2_val

    # WB->ID forwarding
    wb_fwd_valid = memwb_valid.out()
    wb_fwd_rd = memwb_rd.out()
    wb_fwd_data = memwb_result.out()
    wb_fwd_rs1_hit = wb_fwd_valid & (wb_fwd_rd == id_rs1) & (id_rs1 != u(2, 0))
    wb_fwd_rs2_hit = wb_fwd_valid & (wb_fwd_rd == id_rs2) & (id_rs2 != u(2, 0))
    rs1_val = wb_fwd_data if wb_fwd_rs1_hit else rs1_val
    rs2_val = wb_fwd_data if wb_fwd_rs2_hit else rs2_val

    idex_valid.set(id_valid)
    idex_op.set(id_opcode)
    idex_rd.set(id_rd)
    idex_src1.set(rs1_val)
    idex_src2.set(rs2_val)
    idex_imm.set(id_imm)

    # ================================================================== #
    #  STAGE 3 -- EX (Execute / ALU)                                      #
    # ================================================================== #
    ex_valid = idex_valid.out()
    ex_op = idex_op.out()
    ex_src1 = idex_src1.out()
    ex_src2 = idex_src2.out()
    ex_rd = idex_rd.out()
    ex_imm = idex_imm.out()

    alu_result = (ex_src1 + ex_src2)[0:REG_W]
    alu_result = (ex_src1 - ex_src2)[0:REG_W] if ex_op == u(2, OP_SUB) else alu_result
    alu_result = (ex_src1 & ex_src2) if ex_op == u(2, OP_AND) else alu_result
    alu_result = (u(REG_W - 4, 0) + ex_imm + u(REG_W, 0))[0:REG_W] if ex_op == u(2, OP_LDI) else alu_result

    exmem_valid.set(ex_valid)
    exmem_rd.set(ex_rd)
    exmem_result.set(alu_result)

    # ================================================================== #
    #  STAGE 4 -- MEM (Memory -- pass-through for this design)            #
    # ================================================================== #
    mem_valid = exmem_valid.out()
    mem_rd = exmem_rd.out()
    mem_result = exmem_result.out()

    memwb_valid.set(mem_valid)
    memwb_rd.set(mem_rd)
    memwb_result.set(mem_result)

    # ================================================================== #
    #  STAGE 5 -- WB (Write-Back to register file)                        #
    # ================================================================== #
    wb_valid = memwb_valid.out()
    wb_rd = memwb_rd.out()
    wb_result = memwb_result.out()

    wb_wen = wb_valid & (wb_rd != u(2, 0))

    regs[0].set(u(REG_W, 0))
    regs[1].set(wb_result, when=wb_wen & (wb_rd == u(2, 1)))
    regs[2].set(wb_result, when=wb_wen & (wb_rd == u(2, 2)))
    regs[3].set(wb_result, when=wb_wen & (wb_rd == u(2, 3)))

    # ------------------------------------------------------------------ #
    #  Outputs                                                            #
    # ------------------------------------------------------------------ #
    m.output("pc", pc)
    m.output("r0", regs[0])
    m.output("r1", regs[1])
    m.output("r2", regs[2])
    m.output("r3", regs[3])
    m.output("wb_valid", memwb_valid)
    m.output("wb_rd", memwb_rd)
    m.output("wb_result", memwb_result)


build.__pycircuit_name__ = "cpu8"


if __name__ == "__main__":
    print(compile(build, name="cpu8").emit_mlir())
