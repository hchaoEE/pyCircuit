from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .spec import LinxISASpec, load_spec


def _pattern_to_mask_match(pattern: str) -> tuple[int, int]:
    width_bits = len(pattern)
    mask = 0
    match = 0
    for i, ch in enumerate(pattern):
        bit = width_bits - 1 - i
        if ch == ".":
            continue
        if ch not in ("0", "1"):
            raise ValueError(f"invalid pattern char {ch!r}")
        mask |= 1 << bit
        if ch == "1":
            match |= 1 << bit
    return mask, match


@dataclass(frozen=True)
class FieldPiece:
    insn_lsb: int
    width: int
    value_lsb: int


@dataclass
class Field:
    name: str
    signed: Optional[bool]
    pieces: list[FieldPiece]

    @property
    def bit_width(self) -> int:
        if not self.pieces:
            return 0
        return max(p.value_lsb + p.width for p in self.pieces)


@dataclass(frozen=True)
class Form:
    id: str
    mnemonic: str
    asm_fmt: str
    length_bits: int
    mask: int
    match: int
    fixed_bits: int
    fields: dict[str, Field]


@dataclass(frozen=True)
class DecodedInst:
    form: Form
    bits: int
    fields: dict[str, int]

    @property
    def mnemonic(self) -> str:
        return self.form.mnemonic

    @property
    def length_bits(self) -> int:
        return self.form.length_bits

    @property
    def length_bytes(self) -> int:
        return self.form.length_bits // 8


def _build_combined(inst: dict[str, Any]) -> tuple[int, str, dict[str, Field]]:
    enc = inst.get("encoding", {})
    parts: list[dict[str, Any]] = list(enc.get("parts", []))
    length_bits = int(enc.get("length_bits", inst.get("length_bits", 0)))
    if not parts:
        return length_bits, "." * length_bits, {}

    offsets: list[int] = []
    off = 0
    for p in parts:
        offsets.append(off)
        off += int(p.get("width_bits", 0))

    pattern = "".join(str(parts[i].get("pattern", "")).replace(" ", "") for i in reversed(range(len(parts))))
    if len(pattern) != length_bits:
        pattern = (("." * length_bits) + pattern)[-length_bits:]

    fields: dict[str, Field] = {}
    for part_index, part in enumerate(parts):
        part_off = offsets[part_index]
        for f in part.get("fields", []):
            name = str(f.get("name", "")).strip()
            if not name:
                continue
            existing = fields.get(name)
            if existing is None:
                existing = Field(name=name, signed=f.get("signed", None), pieces=[])
                fields[name] = existing
            if existing.signed is None and f.get("signed") is not None:
                existing.signed = f.get("signed")

            for p in f.get("pieces", []):
                insn_lsb = int(p.get("insn_lsb", 0)) + part_off
                width = int(p.get("width", 0))
                value_lsb = int(p.get("value_lsb", 0) if p.get("value_lsb") is not None else 0)
                existing.pieces.append(FieldPiece(insn_lsb=insn_lsb, width=width, value_lsb=value_lsb))

    for fld in fields.values():
        fld.pieces.sort(key=lambda p: (p.value_lsb, p.insn_lsb))
    return length_bits, pattern, fields


def _extract_fields(bits: int, form: Form) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, field in form.fields.items():
        v = 0
        for p in field.pieces:
            part = (bits >> p.insn_lsb) & ((1 << p.width) - 1)
            v |= part << p.value_lsb

        if field.signed is True and field.bit_width > 0:
            sign_bit = 1 << (field.bit_width - 1)
            if v & sign_bit:
                v -= 1 << field.bit_width

        out[name] = v
    return out


class Codec:
    def __init__(self, spec: LinxISASpec | None = None) -> None:
        if spec is None:
            spec = load_spec()
        self.spec = spec

        self._forms_by_len: dict[int, list[Form]] = {}
        for inst in self.spec.data.get("instructions", []):
            length_bits, pattern, fields = _build_combined(inst)
            mask, match = _pattern_to_mask_match(pattern)
            fixed_bits = mask.bit_count()
            f = Form(
                id=str(inst.get("id", "")),
                mnemonic=str(inst.get("mnemonic", "")),
                asm_fmt=str(inst.get("asm", "") or ""),
                length_bits=length_bits,
                mask=mask,
                match=match,
                fixed_bits=fixed_bits,
                fields=fields,
            )
            self._forms_by_len.setdefault(length_bits, []).append(f)

        # Prefer more-specific encodings first.
        for l in list(self._forms_by_len.keys()):
            self._forms_by_len[l].sort(key=lambda f: (-f.fixed_bits, f.id))

    def decode_at(self, blob: bytes, offset: int = 0) -> DecodedInst:
        window = int.from_bytes(blob[offset : offset + 8].ljust(8, b"\x00"), "little", signed=False)
        best: Form | None = None
        best_bits = 0
        best_fixed_bits = -1
        best_len_bits = -1

        for length_bits in (16, 32, 48, 64):
            if offset + (length_bits // 8) > len(blob):
                continue
            if length_bits == 64:
                bits = window
            else:
                bits = window & ((1 << length_bits) - 1)

            for form in self._forms_by_len.get(length_bits, []):
                if (bits & form.mask) != form.match:
                    continue
                # Forms are already sorted by fixed-bit count (descending), so the first match is
                # the best for this length.
                if form.fixed_bits > best_fixed_bits or (
                    form.fixed_bits == best_fixed_bits and length_bits > best_len_bits
                ):
                    best = form
                    best_bits = bits
                    best_fixed_bits = form.fixed_bits
                    best_len_bits = length_bits
                break

        if best is None:
            raise ValueError(f"no matching instruction form at offset {offset}")

        fields = _extract_fields(best_bits, best)
        return DecodedInst(form=best, bits=best_bits, fields=fields)

    def encode(self, mnemonic: str, fields: dict[str, int], *, length_bits: int | None = None) -> bytes:
        candidates = []
        for l, forms in self._forms_by_len.items():
            if length_bits is not None and l != length_bits:
                continue
            for f in forms:
                if f.mnemonic == mnemonic:
                    candidates.append(f)

        if not candidates:
            raise KeyError(f"unknown mnemonic {mnemonic!r} (length_bits={length_bits})")

        # Heuristic: require that all provided field keys exist on the form.
        forms = [f for f in candidates if all(k in f.fields for k in fields.keys())]
        if len(forms) != 1:
            raise ValueError(f"ambiguous encode form for {mnemonic!r}: {len(forms)} candidates")
        form = forms[0]

        bits = form.match
        for name, value in fields.items():
            fld = form.fields[name]
            # Pack the logical field bits into instruction pieces.
            for p in fld.pieces:
                part = (int(value) >> p.value_lsb) & ((1 << p.width) - 1)
                bits &= ~(((1 << p.width) - 1) << p.insn_lsb)
                bits |= part << p.insn_lsb

        return int(bits).to_bytes(form.length_bits // 8, "little", signed=False)
