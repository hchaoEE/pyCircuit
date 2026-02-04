from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Union, overload

from .dsl import Module, Signal


def _int_width(ty: str) -> int:
    if not ty.startswith("i"):
        raise TypeError(f"expected integer type iN, got {ty!r}")
    w = int(ty[1:])
    if w <= 0:
        raise ValueError(f"invalid integer width: {ty!r}")
    return w


@dataclass(frozen=True)
class Wire:
    m: Module
    sig: Signal

    def __post_init__(self) -> None:
        _int_width(self.sig.ty)

    @property
    def ref(self) -> str:
        return self.sig.ref

    @property
    def ty(self) -> str:
        return self.sig.ty

    @property
    def width(self) -> int:
        return _int_width(self.sig.ty)

    def __str__(self) -> str:
        return self.sig.ref

    def _coerce(self, other: Union["Wire", "Reg", Signal, int]) -> Signal:
        if isinstance(other, Reg):
            if other.q.m is not self.m:
                raise ValueError("cannot combine wires from different modules")
            return other.q.sig
        if isinstance(other, Wire):
            if other.m is not self.m:
                raise ValueError("cannot combine wires from different modules")
            return other.sig
        if isinstance(other, Signal):
            return other
        if isinstance(other, int):
            return self.m.const(other, width=self.width)
        raise TypeError(f"unsupported operand type: {type(other).__name__}")

    def __add__(self, other: Union["Wire", "Reg", Signal, int]) -> "Wire":
        b = self._coerce(other)
        return Wire(self.m, self.m.add(self.sig, b))

    def __and__(self, other: Union["Wire", "Reg", Signal, int]) -> "Wire":
        b = self._coerce(other)
        return Wire(self.m, self.m.and_(self.sig, b))

    def __or__(self, other: Union["Wire", "Reg", Signal, int]) -> "Wire":
        b = self._coerce(other)
        return Wire(self.m, self.m.or_(self.sig, b))

    def __xor__(self, other: Union["Wire", "Reg", Signal, int]) -> "Wire":
        b = self._coerce(other)
        return Wire(self.m, self.m.xor(self.sig, b))

    def __invert__(self) -> "Wire":
        return Wire(self.m, self.m.not_(self.sig))

    def eq(self, other: Union["Wire", "Reg", Signal, int]) -> "Wire":
        b = self._coerce(other)
        if b.ty != self.ty:
            raise TypeError(f"eq requires same types, got {self.ty} and {b.ty}")
        return Wire(self.m, self.m.eq(self.sig, b))

    def select(self, a: Union["Wire", "Reg", Signal, int], b: Union["Wire", "Reg", Signal, int]) -> "Wire":
        if self.ty != "i1":
            raise TypeError("select() requires a 1-bit selector wire (i1)")
        aa = self._coerce(a)
        bb = self._coerce(b)
        return Wire(self.m, self.m.mux(self.sig, aa, bb))

    def trunc(self, *, width: int) -> "Wire":
        return Wire(self.m, self.m.trunc(self.sig, width=width))

    def zext(self, *, width: int) -> "Wire":
        return Wire(self.m, self.m.zext(self.sig, width=width))

    def sext(self, *, width: int) -> "Wire":
        return Wire(self.m, self.m.sext(self.sig, width=width))

    def slice(self, *, lsb: int, width: int) -> "Wire":
        return Wire(self.m, self.m.extract(self.sig, lsb=lsb, width=width))

    def shl(self, *, amount: int) -> "Wire":
        return Wire(self.m, self.m.shli(self.sig, amount=amount))


@dataclass(frozen=True)
class ClockDomain:
    clk: Signal
    rst: Signal


@dataclass(frozen=True)
class Reg:
    q: Wire
    clk: Signal
    rst: Signal
    en: Wire
    next: Wire
    init: Wire

    @property
    def ref(self) -> str:
        return self.q.ref

    @property
    def ty(self) -> str:
        return self.q.ty

    @property
    def width(self) -> int:
        return self.q.width

    def __str__(self) -> str:
        return self.q.ref

    def __add__(self, other: Union[Wire, Signal, int]) -> Wire:
        return self.q + other

    def __and__(self, other: Union[Wire, Signal, int]) -> Wire:
        return self.q & other

    def __or__(self, other: Union[Wire, Signal, int]) -> Wire:
        return self.q | other

    def __xor__(self, other: Union[Wire, Signal, int]) -> Wire:
        return self.q ^ other

    def __invert__(self) -> Wire:
        return ~self.q

    def eq(self, other: Union[Wire, Signal, int]) -> Wire:
        return self.q.eq(other)

    def select(self, a: Union[Wire, "Reg", Signal, int], b: Union[Wire, "Reg", Signal, int]) -> Wire:
        return self.q.select(a, b)

    def trunc(self, *, width: int) -> Wire:
        return self.q.trunc(width=width)

    def zext(self, *, width: int) -> Wire:
        return self.q.zext(width=width)

    def sext(self, *, width: int) -> Wire:
        return self.q.sext(width=width)

    def slice(self, *, lsb: int, width: int) -> Wire:
        return self.q.slice(lsb=lsb, width=width)

    def shl(self, *, amount: int) -> Wire:
        return self.q.shl(amount=amount)


class Circuit(Module):
    """High-level wrapper over `Module` that returns `Wire`/`Reg` objects."""

    def domain(self, name: str) -> ClockDomain:
        return ClockDomain(clk=self.clock(f"{name}_clk"), rst=self.reset(f"{name}_rst"))

    def in_wire(self, name: str, *, width: int) -> Wire:
        return Wire(self, self.input(name, width=width))

    def const_wire(self, value: int, *, width: int) -> Wire:
        return Wire(self, self.const(value, width=width))

    def new_wire(self, *, width: int) -> Wire:
        return Wire(self, super().new_wire(width=width))

    def wire(self, sig: Signal) -> Wire:
        return Wire(self, sig)

    def reg_wire(
        self, clk: Signal, rst: Signal, en: Union[Wire, Signal], next_: Union[Wire, Signal], init: Union[Wire, Signal, int]
    ) -> Reg:
        en_w = en if isinstance(en, Wire) else Wire(self, en)
        next_w = next_ if isinstance(next_, Wire) else Wire(self, next_)
        if isinstance(init, int):
            init_w = self.const_wire(init, width=next_w.width)
        else:
            init_w = init if isinstance(init, Wire) else Wire(self, init)

        q_sig = self.reg(clk, rst, en_w.sig, next_w.sig, init_w.sig)
        q_w = Wire(self, q_sig)
        return Reg(q=q_w, clk=clk, rst=rst, en=en_w, next=next_w, init=init_w)

    def reg_domain(self, domain: ClockDomain, en: Union[Wire, Signal], next_: Union[Wire, Signal], init: Union[Wire, Signal, int]) -> Reg:
        return self.reg_wire(domain.clk, domain.rst, en, next_, init)

    def backedge_reg(
        self,
        clk: Signal,
        rst: Signal,
        *,
        width: int,
        init: Union[Wire, Signal, int],
        en: Union[Wire, Signal, int] = 1,
    ) -> Reg:
        """Create a register whose `next` is a placeholder `pyc.wire` meant to be driven via `pyc.assign`.

        This pattern enables feedback loops (state machines) in a netlist-like style:

        - `r = m.backedge_reg(...)` creates `r.next` as a `pyc.wire`
        - Later: `m.assign(r.next.sig, some_next_value.sig)`
        """
        next_w = self.new_wire(width=width)
        if isinstance(en, int):
            en_w: Union[Wire, Signal] = self.const_wire(en, width=1)
        else:
            en_w = en
        return self.reg_wire(clk, rst, en_w, next_w, init)

    def vec(self, *elems: Union["Wire", "Reg"]) -> "Vec":
        return Vec(elems)

    def byte_mem(
        self,
        clk: Signal,
        rst: Signal,
        *,
        raddr: Union[Wire, Reg, Signal],
        wvalid: Union[Wire, Reg, Signal],
        waddr: Union[Wire, Reg, Signal],
        wdata: Union[Wire, Reg, Signal],
        wstrb: Union[Wire, Reg, Signal],
        depth: int,
        name: str | None = None,
    ) -> Wire:
        def as_sig(v: Union[Wire, Reg, Signal]) -> Signal:
            if isinstance(v, Reg):
                return v.q.sig
            if isinstance(v, Wire):
                return v.sig
            return v

        rdata = super().byte_mem(
            clk,
            rst,
            as_sig(raddr),
            as_sig(wvalid),
            as_sig(waddr),
            as_sig(wdata),
            as_sig(wstrb),
            depth=depth,
            name=name,
        )
        return Wire(self, rdata)


@dataclass(frozen=True)
class Vec:
    """A small fixed-length container of wires/regs for building pipelines."""

    elems: tuple[Union[Wire, Reg], ...]

    def __post_init__(self) -> None:
        if not self.elems:
            raise ValueError("Vec cannot be empty")

        m0 = self._module_of(self.elems[0])
        for e in self.elems[1:]:
            if self._module_of(e) is not m0:
                raise ValueError("Vec elements must belong to the same Circuit/Module")

    @staticmethod
    def _module_of(e: Union[Wire, Reg]) -> Module:
        if isinstance(e, Wire):
            return e.m
        return e.q.m

    @property
    def m(self) -> Module:
        return self._module_of(self.elems[0])

    def __len__(self) -> int:
        return len(self.elems)

    def __iter__(self) -> Iterator[Union[Wire, Reg]]:
        return iter(self.elems)

    @overload
    def __getitem__(self, idx: int) -> Union[Wire, Reg]: ...

    @overload
    def __getitem__(self, idx: slice) -> "Vec": ...

    def __getitem__(self, idx: int | slice) -> Union[Wire, Reg, "Vec"]:
        if isinstance(idx, slice):
            return Vec(self.elems[idx])
        return self.elems[int(idx)]

    def wires(self) -> tuple[Wire, ...]:
        out: list[Wire] = []
        for e in self.elems:
            out.append(e if isinstance(e, Wire) else e.q)
        return tuple(out)

    @property
    def total_width(self) -> int:
        return sum(w.width for w in self.wires())

    def pack(self) -> Wire:
        """Concatenate elements into a single bus wire (MSB-first).

        `Vec([a, b, c]).pack()` yields `{a, b, c}` in Verilog terms.
        """
        ws = self.wires()
        out_w = self.total_width
        if out_w <= 0:
            raise ValueError("cannot pack a zero-width Vec")

        m = ws[0].m
        acc = m.const_wire(0, width=out_w)
        lsb = 0
        for w in reversed(ws):
            part = w.zext(width=out_w)
            if lsb:
                part = part.shl(amount=lsb)
            acc = acc | part
            lsb += w.width
        return acc

    def unpack(self, packed: Wire) -> "Vec":
        """Extract elements from a packed bus (inverse of pack())."""
        ws = self.wires()
        if packed.width != self.total_width:
            raise ValueError(f"unpack width mismatch: got i{packed.width}, expected i{self.total_width}")

        parts_rev: list[Wire] = []
        lsb = 0
        for w in reversed(ws):
            parts_rev.append(packed.slice(lsb=lsb, width=w.width))
            lsb += w.width
        return Vec(tuple(reversed(parts_rev)))

    def regs_domain(self, domain: ClockDomain, en: Union[Wire, Signal, int], init: Union[Wire, Signal, int] = 0) -> "Vec":
        """Create a register per element and return a Vec of Regs."""
        ws = self.wires()
        m = ws[0].m
        if not isinstance(m, Circuit):
            raise TypeError("regs_domain requires elements to belong to a Circuit")
        regs: list[Reg] = []
        for w in ws:
            regs.append(m.reg_domain(domain, en, w, init))
        return Vec(tuple(regs))
