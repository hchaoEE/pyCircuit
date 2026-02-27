"""Case 9: mem_subsys — Memory subsystem with 4 banks + parity.

Tests: SRAM inference, address decoding, ECC/parity logic.
"""
from __future__ import annotations

from pycircuit import Circuit, compile, module, u, unsigned

BANKS = 4
BANK_DEPTH = 16
BANK_ADDR_W = 4
DATA_W = 8
FULL_ADDR_W = 6


@module
def build(m: Circuit) -> None:
    clk = m.clock("clk")
    rst = m.reset("rst")

    addr = m.input("addr", width=FULL_ADDR_W)
    wdata = m.input("wdata", width=DATA_W)
    wen = m.input("wen", width=1)
    ren = m.input("ren", width=1)

    bank_sel = addr[BANK_ADDR_W:FULL_ADDR_W]
    word_addr = addr[0:BANK_ADDR_W]

    # 4 banks x 16 x 8-bit storage + 16 x 1-bit parity
    banks = []
    parity_banks = []
    for b in range(BANKS):
        bank = [
            m.out(f"bk{b}_w{w}", clk=clk, rst=rst, width=DATA_W, init=u(DATA_W, 0))
            for w in range(BANK_DEPTH)
        ]
        pbank = [
            m.out(f"pk{b}_w{w}", clk=clk, rst=rst, width=1, init=u(1, 0))
            for w in range(BANK_DEPTH)
        ]
        banks.append(bank)
        parity_banks.append(pbank)

    # Compute parity of write data (XOR reduction)
    wr_parity = wdata[0]
    for i in range(1, DATA_W):
        wr_parity = wr_parity ^ wdata[i]

    # Write logic
    for b in range(BANKS):
        b_sel = wen & (bank_sel == u(FULL_ADDR_W - BANK_ADDR_W, b))
        for w in range(BANK_DEPTH):
            w_sel = b_sel & (word_addr == u(BANK_ADDR_W, w))
            banks[b][w].set(wdata, when=w_sel)
            parity_banks[b][w].set(wr_parity, when=w_sel)

    # Read logic: bank select → word select
    rd_data = u(DATA_W, 0)
    rd_parity = u(1, 0)

    for b in range(BANKS):
        b_hit = bank_sel == u(FULL_ADDR_W - BANK_ADDR_W, b)
        bank_rd = u(DATA_W, 0)
        bank_par = u(1, 0)
        for w in range(BANK_DEPTH):
            bank_rd = banks[b][w].out() if word_addr == u(BANK_ADDR_W, w) else bank_rd
            bank_par = parity_banks[b][w].out() if word_addr == u(BANK_ADDR_W, w) else bank_par
        rd_data = bank_rd if b_hit else rd_data
        rd_parity = bank_par if b_hit else rd_parity

    # Parity check
    check_parity = rd_data[0]
    for i in range(1, DATA_W):
        check_parity = check_parity ^ rd_data[i]
    parity_err = (check_parity ^ rd_parity) & ren

    m.output("rd_data", rd_data)
    m.output("parity_err", parity_err)
    m.output("ready", u(1, 1))


build.__pycircuit_name__ = "mem_subsys"

if __name__ == "__main__":
    print(compile(build, name="mem_subsys").emit_mlir())
