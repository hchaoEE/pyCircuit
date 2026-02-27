from __future__ import annotations

import sys
from pathlib import Path

from pycircuit import Tb, compile, testbench

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from cpu8 import build  # noqa: E402
from cpu8_config import DEFAULT_PARAMS, TB_PRESETS  # noqa: E402

OP_ADD = 0
OP_SUB = 1
OP_AND = 2
OP_LDI = 3


def encode(opcode: int, rd: int, rs1: int, rs2: int) -> int:
    return ((opcode & 0x3) << 6) | ((rd & 0x3) << 4) | ((rs1 & 0x3) << 2) | (rs2 & 0x3)


def encode_ldi(rd: int, imm4: int) -> int:
    return encode(OP_LDI, rd, (imm4 >> 2) & 0x3, imm4 & 0x3)


@testbench
def tb(t: Tb) -> None:
    p = TB_PRESETS["smoke"]
    t.clock("clk")
    t.reset("rst", cycles_asserted=2, cycles_deasserted=1)
    t.timeout(int(p["timeout"]))

    t.drive("inst_in", 0, at=0)

    t.drive("inst_in", encode_ldi(1, 5), at=0)
    t.drive("inst_in", encode_ldi(2, 3), at=1)
    t.drive("inst_in", encode(OP_ADD, 3, 1, 2), at=2)
    t.drive("inst_in", 0, at=3)
    t.drive("inst_in", 0, at=4)
    t.drive("inst_in", 0, at=5)
    t.drive("inst_in", 0, at=6)

    t.finish(at=int(p["finish"]))


if __name__ == "__main__":
    print(compile(build, name="tb_cpu8_top", **DEFAULT_PARAMS).emit_mlir())
