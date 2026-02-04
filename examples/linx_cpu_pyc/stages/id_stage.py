from __future__ import annotations

from pycircuit import Circuit, Wire

from ..decode import decode_window
from ..pipeline import IdExRegs, IfIdRegs, RegFiles
from ..regfile import read_reg
from ..util import Consts, latch_many


def build_id_stage(m: Circuit, *, do_id: Wire, ifid: IfIdRegs, idex: IdExRegs, rf: RegFiles, consts: Consts) -> None:
    dec = decode_window(m, ifid.window)

    latch_many(
        m,
        do_id,
        [
            (idex.op, dec.op),
            (idex.len_bytes, dec.len_bytes),
            (idex.regdst, dec.regdst),
            (idex.srcl, dec.srcl),
            (idex.srcr, dec.srcr),
            (idex.srcp, dec.srcp),
            (idex.imm, dec.imm),
        ],
    )

    srcl_val_new = read_reg(m, dec.srcl, gpr=rf.gpr, t=rf.t, u=rf.u, default=consts.zero64)
    srcr_val_new = read_reg(m, dec.srcr, gpr=rf.gpr, t=rf.t, u=rf.u, default=consts.zero64)
    srcp_val_new = read_reg(m, dec.srcp, gpr=rf.gpr, t=rf.t, u=rf.u, default=consts.zero64)
    latch_many(m, do_id, [(idex.srcl_val, srcl_val_new), (idex.srcr_val, srcr_val_new), (idex.srcp_val, srcp_val_new)])

