from __future__ import annotations

import sys
from pathlib import Path

from pycircuit import Tb, compile, testbench

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from rv32i_cpu import build  # noqa: E402
from rv32i_cpu_config import DEFAULT_PARAMS, TB_PRESETS  # noqa: E402


def r_type(funct7: int, rs2: int, rs1: int, funct3: int, rd: int, opcode: int = 0x33) -> int:
    return (funct7 << 25) | (rs2 << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def i_type(imm12: int, rs1: int, funct3: int, rd: int, opcode: int = 0x13) -> int:
    return ((imm12 & 0xFFF) << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode


def u_type(imm20: int, rd: int, opcode: int = 0x37) -> int:
    return ((imm20 & 0xFFFFF) << 12) | (rd << 7) | opcode


NOP = i_type(0, 0, 0, 0)
ADD = lambda rd, rs1, rs2: r_type(0x00, rs2, rs1, 0, rd)
SUB = lambda rd, rs1, rs2: r_type(0x20, rs2, rs1, 0, rd)
AND = lambda rd, rs1, rs2: r_type(0x00, rs2, rs1, 7, rd)
OR = lambda rd, rs1, rs2: r_type(0x00, rs2, rs1, 6, rd)
XOR = lambda rd, rs1, rs2: r_type(0x00, rs2, rs1, 4, rd)
ADDI = lambda rd, rs1, imm: i_type(imm, rs1, 0, rd)
LUI = lambda rd, imm20: u_type(imm20, rd, 0x37)


@testbench
def tb(t: Tb) -> None:
    p = TB_PRESETS["smoke"]
    t.clock("clk")
    t.reset("rst", cycles_asserted=2, cycles_deasserted=1)
    t.timeout(int(p["timeout"]))

    t.drive("inst_data", NOP, at=0)
    t.drive("dmem_rdata", 0, at=0)

    t.drive("inst_data", ADDI(1, 0, 5), at=0)
    t.drive("inst_data", ADDI(2, 0, 10), at=1)
    t.drive("inst_data", ADD(3, 1, 2), at=2)
    t.drive("inst_data", SUB(4, 2, 1), at=3)
    t.drive("inst_data", NOP, at=4)
    t.drive("inst_data", NOP, at=5)
    t.drive("inst_data", NOP, at=6)

    t.finish(at=int(p["finish"]))


if __name__ == "__main__":
    print(compile(build, name="tb_rv32i_cpu_top", **DEFAULT_PARAMS).emit_mlir())
