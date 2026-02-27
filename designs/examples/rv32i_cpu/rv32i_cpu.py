"""
rv32i_cpu -- 32-bit 5-stage pipelined RV32I CPU in pyCircuit.

Pipeline:  IF -> ID -> EX -> MEM -> WB

Supported RV32I instructions:
  R-type : ADD SUB SLL SLT SLTU XOR SRL SRA OR AND
  I-type : ADDI SLTI SLTIU XORI ORI ANDI SLLI SRLI SRAI
  Load   : LW
  Store  : SW
  Branch : BEQ BNE BLT BGE BLTU BGEU
  Upper  : LUI AUIPC
  Jump   : JAL JALR

Micro-architecture:
  - 32 x 32-bit register file (x0 hardwired to 0)
  - Data forwarding from EX/MEM and MEM/WB to ID stage
  - Branch / jump resolved in EX; pipeline flush on taken
  - Barrel shifter for SLL / SRL / SRA
  - External instruction memory (Harvard architecture)
  - External data memory (word-aligned LW / SW)
"""
from __future__ import annotations

from pycircuit import Circuit, compile, module, u, unsigned

XLEN = 32


@module
def build(m: Circuit) -> None:
    clk = m.clock("clk")
    rst = m.reset("rst")

    inst_data = m.input("inst_data", width=32)
    dmem_rdata = m.input("dmem_rdata", width=32)

    # ================================================================== #
    #  Register file  (x0 = 0)                                           #
    # ================================================================== #
    regs = [
        m.out(f"x{i}", clk=clk, rst=rst, width=XLEN, init=u(XLEN, 0))
        for i in range(32)
    ]

    pc = m.out("pc_q", clk=clk, rst=rst, width=32, init=u(32, 0))

    # ================================================================== #
    #  Pipeline registers                                                 #
    # ================================================================== #

    # IF/ID
    ifid_v = m.out("ifid_v", clk=clk, rst=rst, width=1, init=u(1, 0))
    ifid_pc = m.out("ifid_pc", clk=clk, rst=rst, width=32, init=u(32, 0))
    ifid_inst = m.out("ifid_inst", clk=clk, rst=rst, width=32, init=u(32, 0))

    # ID/EX
    idex_v = m.out("idex_v", clk=clk, rst=rst, width=1, init=u(1, 0))
    idex_pc = m.out("idex_pc", clk=clk, rst=rst, width=32, init=u(32, 0))
    idex_rs1 = m.out("idex_rs1", clk=clk, rst=rst, width=32, init=u(32, 0))
    idex_rs2 = m.out("idex_rs2", clk=clk, rst=rst, width=32, init=u(32, 0))
    idex_imm = m.out("idex_imm", clk=clk, rst=rst, width=32, init=u(32, 0))
    idex_rd = m.out("idex_rd", clk=clk, rst=rst, width=5, init=u(5, 0))
    idex_f3 = m.out("idex_f3", clk=clk, rst=rst, width=3, init=u(3, 0))
    idex_f7b5 = m.out("idex_f7b5", clk=clk, rst=rst, width=1, init=u(1, 0))
    idex_rtype = m.out("idex_rtype", clk=clk, rst=rst, width=1, init=u(1, 0))
    idex_ld = m.out("idex_ld", clk=clk, rst=rst, width=1, init=u(1, 0))
    idex_st = m.out("idex_st", clk=clk, rst=rst, width=1, init=u(1, 0))
    idex_br = m.out("idex_br", clk=clk, rst=rst, width=1, init=u(1, 0))
    idex_jal = m.out("idex_jal", clk=clk, rst=rst, width=1, init=u(1, 0))
    idex_jalr = m.out("idex_jalr", clk=clk, rst=rst, width=1, init=u(1, 0))
    idex_lui = m.out("idex_lui", clk=clk, rst=rst, width=1, init=u(1, 0))
    idex_auipc = m.out("idex_auipc", clk=clk, rst=rst, width=1, init=u(1, 0))

    # EX/MEM
    exmem_v = m.out("exmem_v", clk=clk, rst=rst, width=1, init=u(1, 0))
    exmem_rd = m.out("exmem_rd", clk=clk, rst=rst, width=5, init=u(5, 0))
    exmem_res = m.out("exmem_res", clk=clk, rst=rst, width=32, init=u(32, 0))
    exmem_rs2 = m.out("exmem_rs2v", clk=clk, rst=rst, width=32, init=u(32, 0))
    exmem_ld = m.out("exmem_ld", clk=clk, rst=rst, width=1, init=u(1, 0))
    exmem_st = m.out("exmem_st", clk=clk, rst=rst, width=1, init=u(1, 0))

    # MEM/WB
    memwb_v = m.out("memwb_v", clk=clk, rst=rst, width=1, init=u(1, 0))
    memwb_rd = m.out("memwb_rd", clk=clk, rst=rst, width=5, init=u(5, 0))
    memwb_res = m.out("memwb_res", clk=clk, rst=rst, width=32, init=u(32, 0))

    # ================================================================== #
    #  STAGE 3 — EX  (compute first; produces flush signal)              #
    # ================================================================== #
    ev = idex_v.out()
    epc = idex_pc.out()
    ea = idex_rs1.out()
    eb = idex_rs2.out()
    eimm = idex_imm.out()
    erd = idex_rd.out()
    ef3 = idex_f3.out()
    ef7 = idex_f7b5.out()
    e_rt = idex_rtype.out()
    e_ld = idex_ld.out()
    e_st = idex_st.out()
    e_br = idex_br.out()
    e_jal = idex_jal.out()
    e_jalr = idex_jalr.out()
    e_lui = idex_lui.out()
    e_auipc = idex_auipc.out()

    alu_b = eb if e_rt else eimm

    # --- arithmetic ---
    add_r = (ea + alu_b)[0:32]
    sub_r = (ea - alu_b)[0:32]
    and_r = ea & alu_b
    or_r = ea | alu_b
    xor_r = ea ^ alu_b

    # --- signed less-than ---
    sa = ea[31]
    sb = alu_b[31]
    sd = sa ^ sb
    u_lt = ea < alu_b
    s_lt = sa if sd else u_lt
    slt_r = (unsigned(s_lt) + u(32, 0))[0:32]
    sltu_r = (unsigned(u_lt) + u(32, 0))[0:32]

    # --- barrel SLL ---
    sh = alu_b[0:5]
    sll = ea
    for i in range(5):
        a = 1 << i
        t = (sll << a)[0:32]
        sll = t if sh[i] else sll

    # --- barrel SRL ---
    srl = ea
    for i in range(5):
        a = 1 << i
        lo = srl[a:32]
        t = (unsigned(lo) + u(32, 0))[0:32]
        srl = t if sh[i] else srl

    # --- barrel SRA ---
    sra = ea
    for i in range(5):
        a = 1 << i
        sign = sra[31]
        lo = sra[a:32]
        t = (unsigned(lo) + u(32, 0))[0:32]
        mv = ((1 << a) - 1) << (32 - a)
        filled = (t | u(32, mv)) if sign else t
        sra = filled if sh[i] else sra

    sr_r = sra if ef7 else srl

    # --- ALU select (funct3) ---
    is_sub = ef7 & e_rt
    alu = sub_r if is_sub else add_r
    alu = sll if ef3 == u(3, 1) else alu
    alu = slt_r if ef3 == u(3, 2) else alu
    alu = sltu_r if ef3 == u(3, 3) else alu
    alu = xor_r if ef3 == u(3, 4) else alu
    alu = sr_r if ef3 == u(3, 5) else alu
    alu = or_r if ef3 == u(3, 6) else alu
    alu = and_r if ef3 == u(3, 7) else alu

    # --- branch / jump ---
    br_tgt = (epc + eimm)[0:32]
    jalr_tgt = ((ea + eimm) & u(32, 0xFFFFFFFE))[0:32]
    ret_addr = (epc + u(32, 4))[0:32]

    eq_f = ea == eb
    ne_f = ~eq_f
    sa2 = ea[31]
    sb2 = eb[31]
    sd2 = sa2 ^ sb2
    ult2 = ea < eb
    lt_f = sa2 if sd2 else ult2
    ge_f = ~lt_f

    br_cond = u(1, 0)
    br_cond = eq_f if ef3 == u(3, 0) else br_cond
    br_cond = ne_f if ef3 == u(3, 1) else br_cond
    br_cond = lt_f if ef3 == u(3, 4) else br_cond
    br_cond = ge_f if ef3 == u(3, 5) else br_cond
    br_cond = ult2 if ef3 == u(3, 6) else br_cond
    br_cond = ~ult2 if ef3 == u(3, 7) else br_cond

    br_taken = ev & e_br & br_cond
    j_taken = ev & e_jal
    jr_taken = ev & e_jalr
    flush = br_taken | j_taken | jr_taken

    redir = br_tgt
    redir = jalr_tgt if jr_taken else redir

    # --- EX result ---
    ex_res = alu
    ex_res = eimm if e_lui else ex_res
    ex_res = (epc + eimm)[0:32] if e_auipc else ex_res
    ex_res = ret_addr if (e_jal | e_jalr) else ex_res
    ex_res = (ea + eimm)[0:32] if (e_ld | e_st) else ex_res

    # ================================================================== #
    #  STAGE 4 — MEM                                                      #
    # ================================================================== #
    m_v = exmem_v.out()
    m_rd = exmem_rd.out()
    m_res = exmem_res.out()
    m_rs2 = exmem_rs2.out()
    m_ld = exmem_ld.out()
    m_st = exmem_st.out()

    mem_result = dmem_rdata if m_ld else m_res

    # ================================================================== #
    #  STAGE 5 — WB                                                       #
    # ================================================================== #
    wb_v = memwb_v.out()
    wb_rd = memwb_rd.out()
    wb_data = memwb_res.out()
    wb_wen = wb_v & (wb_rd != u(5, 0))

    # ================================================================== #
    #  STAGE 1 — IF                                                       #
    # ================================================================== #
    pc_val = pc.out()
    pc_plus4 = (pc_val + u(32, 4))[0:32]
    pc_next = redir if flush else pc_plus4

    # ================================================================== #
    #  STAGE 2 — ID  (decode + imm-gen + reg-read + forwarding)          #
    # ================================================================== #
    id_v = ifid_v.out()
    id_pc = ifid_pc.out()
    id_inst = ifid_inst.out()

    opcode = id_inst[0:7]
    rd_f = id_inst[7:12]
    f3 = id_inst[12:15]
    rs1_f = id_inst[15:20]
    rs2_f = id_inst[20:25]
    f7b5 = id_inst[30:31]

    is_lui = opcode == u(7, 0b0110111)
    is_auipc = opcode == u(7, 0b0010111)
    is_jal = opcode == u(7, 0b1101111)
    is_jalr = opcode == u(7, 0b1100111)
    is_br = opcode == u(7, 0b1100011)
    is_ld = opcode == u(7, 0b0000011)
    is_st = opcode == u(7, 0b0100011)
    is_alui = opcode == u(7, 0b0010011)
    is_alur = opcode == u(7, 0b0110011)

    # --- immediate generation ---

    # I-type (sign-extend 12-bit)
    i_raw = id_inst[20:32]
    i_z = (unsigned(i_raw) + u(32, 0))[0:32]
    imm_i = (i_z | u(32, 0xFFFFF000)) if i_raw[11] else i_z

    # S-type (sign-extend 12-bit from scattered fields)
    s_hi = (unsigned(id_inst[25:32]) + u(32, 0))[0:32]
    s_lo = (unsigned(id_inst[7:12]) + u(32, 0))[0:32]
    s_raw = ((s_hi << 5) | s_lo)[0:12]
    s_z = (unsigned(s_raw) + u(32, 0))[0:32]
    imm_s = (s_z | u(32, 0xFFFFF000)) if s_raw[11] else s_z

    # B-type
    b12 = (unsigned(id_inst[31:32]) + u(32, 0))[0:32]
    b11 = (unsigned(id_inst[7:8]) + u(32, 0))[0:32]
    b10_5 = (unsigned(id_inst[25:31]) + u(32, 0))[0:32]
    b4_1 = (unsigned(id_inst[8:12]) + u(32, 0))[0:32]
    imm_b_raw = ((b12 << 12) | (b11 << 11) | (b10_5 << 5) | (b4_1 << 1))[0:32]
    imm_b_sign = id_inst[31]
    imm_b = (imm_b_raw | u(32, 0xFFFFE000)) if imm_b_sign else imm_b_raw

    # U-type
    imm_u = ((unsigned(id_inst[12:32]) + u(32, 0)) << 12)[0:32]

    # J-type
    j20 = (unsigned(id_inst[31:32]) + u(32, 0))[0:32]
    j19_12 = (unsigned(id_inst[12:20]) + u(32, 0))[0:32]
    j11 = (unsigned(id_inst[20:21]) + u(32, 0))[0:32]
    j10_1 = (unsigned(id_inst[21:31]) + u(32, 0))[0:32]
    imm_j_raw = ((j20 << 20) | (j19_12 << 12) | (j11 << 11) | (j10_1 << 1))[0:32]
    imm_j_sign = id_inst[31]
    imm_j = (imm_j_raw | u(32, 0xFFE00000)) if imm_j_sign else imm_j_raw

    # select immediate
    imm = imm_i
    imm = imm_s if is_st else imm
    imm = imm_b if is_br else imm
    imm = imm_u if (is_lui | is_auipc) else imm
    imm = imm_j if is_jal else imm

    # --- register read ---
    rs1_val = u(32, 0)
    for i in range(1, 32):
        rs1_val = regs[i].out() if rs1_f == u(5, i) else rs1_val

    rs2_val = u(32, 0)
    for i in range(1, 32):
        rs2_val = regs[i].out() if rs2_f == u(5, i) else rs2_val

    # --- forwarding  (EX/MEM → ID,  MEM/WB → ID) ---
    ex_fwd_v = exmem_v.out() & (exmem_rd.out() != u(5, 0)) & ~exmem_ld.out()
    ex_fwd_rd = exmem_rd.out()
    ex_fwd_d = exmem_res.out()

    wb_fwd_v = memwb_v.out() & (memwb_rd.out() != u(5, 0))
    wb_fwd_rd = memwb_rd.out()
    wb_fwd_d = memwb_res.out()

    rs1_val = ex_fwd_d if (ex_fwd_v & (ex_fwd_rd == rs1_f)) else rs1_val
    rs2_val = ex_fwd_d if (ex_fwd_v & (ex_fwd_rd == rs2_f)) else rs2_val
    rs1_val = wb_fwd_d if (wb_fwd_v & (wb_fwd_rd == rs1_f)) else rs1_val
    rs2_val = wb_fwd_d if (wb_fwd_v & (wb_fwd_rd == rs2_f)) else rs2_val

    # ================================================================== #
    #  Pipeline register updates                                          #
    # ================================================================== #

    # -- PC --
    pc.set(pc_next)

    # -- IF/ID --
    ifid_v.set(u(1, 0) if flush else u(1, 1))
    ifid_pc.set(pc_val)
    ifid_inst.set(inst_data)

    # -- ID/EX --
    id_valid_next = u(1, 0) if flush else id_v
    idex_v.set(id_valid_next)
    idex_pc.set(id_pc)
    idex_rs1.set(rs1_val)
    idex_rs2.set(rs2_val)
    idex_imm.set(imm)
    idex_rd.set(rd_f)
    idex_f3.set(f3)
    idex_f7b5.set(f7b5)
    idex_rtype.set(is_alur)
    idex_ld.set(is_ld)
    idex_st.set(is_st)
    idex_br.set(is_br)
    idex_jal.set(is_jal)
    idex_jalr.set(is_jalr)
    idex_lui.set(is_lui)
    idex_auipc.set(is_auipc)

    # -- EX/MEM --
    exmem_v.set(ev)
    exmem_rd.set(erd)
    exmem_res.set(ex_res)
    exmem_rs2.set(eb)
    exmem_ld.set(e_ld)
    exmem_st.set(e_st)

    # -- MEM/WB --
    memwb_v.set(m_v)
    memwb_rd.set(m_rd)
    memwb_res.set(mem_result)

    # -- Register file WB --
    regs[0].set(u(32, 0))
    for i in range(1, 32):
        regs[i].set(wb_data, when=wb_wen & (wb_rd == u(5, i)))

    # ================================================================== #
    #  Outputs                                                            #
    # ================================================================== #
    m.output("inst_addr", pc)
    m.output("dmem_addr", exmem_res)
    m.output("dmem_wdata", exmem_rs2)
    m.output("dmem_wen", exmem_st.out() & exmem_v.out())
    m.output("dmem_ren", exmem_ld.out() & exmem_v.out())

    m.output("wb_valid", memwb_v)
    m.output("wb_rd", memwb_rd)
    m.output("wb_data", memwb_res)

    m.output("x1_ra", regs[1])
    m.output("x2_sp", regs[2])
    m.output("x10_a0", regs[10])
    m.output("x11_a1", regs[11])


build.__pycircuit_name__ = "rv32i_cpu"


if __name__ == "__main__":
    print(compile(build, name="rv32i_cpu").emit_mlir())
