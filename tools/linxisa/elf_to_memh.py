#!/usr/bin/env python3
from __future__ import annotations

import argparse
import struct
from dataclasses import dataclass
from pathlib import Path


SHT_SYMTAB = 2
SHT_STRTAB = 3
SHT_RELA = 4
SHT_NOBITS = 8
SHT_REL = 9

ET_REL = 1
ET_EXEC = 2


@dataclass(frozen=True)
class Section:
    index: int
    name: str
    sh_type: int
    sh_flags: int
    sh_addr: int
    sh_offset: int
    sh_size: int
    sh_link: int
    sh_info: int
    sh_addralign: int
    sh_entsize: int


@dataclass(frozen=True)
class Symbol:
    name: str
    st_value: int
    st_size: int
    st_info: int
    st_other: int
    st_shndx: int


def _align_up(x: int, align: int) -> int:
    if align <= 0:
        return x
    return (x + align - 1) & ~(align - 1)


def _read_cstr(blob: bytes, off: int) -> str:
    end = blob.find(b"\x00", off)
    if end < 0:
        end = len(blob)
    return blob[off:end].decode("utf-8", errors="replace")


def _parse_sections(data: bytes) -> tuple[int, int, list[Section], dict[str, Section]]:
    if data[:4] != b"\x7fELF":
        raise ValueError("not an ELF file")
    if data[4] != 2 or data[5] != 1:
        raise ValueError("unsupported ELF class/endian (need 64-bit LSB)")

    (e_ident, e_type, _e_machine, _e_version, e_entry, _e_phoff, e_shoff, _e_flags, _e_ehsize, _e_phentsize, _e_phnum, e_shentsize, e_shnum, e_shstrndx,) = struct.unpack_from(
        "<16sHHIQQQIHHHHHH", data, 0
    )
    _ = (e_ident, _e_version, e_entry, _e_phoff, _e_flags, _e_ehsize, _e_phentsize, _e_phnum)

    shdrs: list[tuple[int, int, int, int, int, int, int, int, int, int]] = []
    for i in range(e_shnum):
        sh = struct.unpack_from("<IIQQQQIIQQ", data, e_shoff + i * e_shentsize)
        shdrs.append(tuple(int(x) for x in sh))

    shstr = b""
    if 0 <= e_shstrndx < len(shdrs):
        sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize = shdrs[
            e_shstrndx
        ]
        _ = (sh_name, sh_type, sh_flags, sh_addr, sh_link, sh_info, sh_addralign, sh_entsize)
        shstr = data[sh_offset : sh_offset + sh_size]

    sections: list[Section] = []
    by_name: dict[str, Section] = {}
    for i, raw in enumerate(shdrs):
        sh_name, sh_type, sh_flags, sh_addr, sh_offset, sh_size, sh_link, sh_info, sh_addralign, sh_entsize = raw
        name = _read_cstr(shstr, sh_name) if shstr else ""
        sec = Section(
            index=i,
            name=name,
            sh_type=sh_type,
            sh_flags=sh_flags,
            sh_addr=sh_addr,
            sh_offset=sh_offset,
            sh_size=sh_size,
            sh_link=sh_link,
            sh_info=sh_info,
            sh_addralign=sh_addralign,
            sh_entsize=sh_entsize,
        )
        sections.append(sec)
        if name:
            by_name[name] = sec

    return int(e_type), int(e_entry), sections, by_name


def _parse_symtab(data: bytes, sections: list[Section], symtab: Section) -> list[Symbol]:
    if symtab.sh_type != SHT_SYMTAB:
        return []
    if symtab.sh_entsize == 0:
        raise ValueError("symtab has entsize=0")
    if not (0 <= symtab.sh_link < len(sections)):
        raise ValueError("symtab sh_link out of range")
    strtab = sections[symtab.sh_link]
    if strtab.sh_type != SHT_STRTAB:
        raise ValueError("symtab sh_link does not point to strtab")
    strblob = data[strtab.sh_offset : strtab.sh_offset + strtab.sh_size]

    out: list[Symbol] = []
    for off in range(0, symtab.sh_size, symtab.sh_entsize):
        st_name, st_info, st_other, st_shndx, st_value, st_size = struct.unpack_from(
            "<IBBHQQ", data, symtab.sh_offset + off
        )
        name = _read_cstr(strblob, int(st_name)) if st_name else ""
        out.append(
            Symbol(
                name=name,
                st_value=int(st_value),
                st_size=int(st_size),
                st_info=int(st_info),
                st_other=int(st_other),
                st_shndx=int(st_shndx),
            )
        )
    return out


def _apply_text_relocs(
    text: bytearray,
    *,
    text_addr: int,
    data: bytes,
    sections: list[Section],
    symtab: list[Symbol],
    section_addrs: dict[int, int],
) -> None:
    # Handle relocations that target .text. In linx-test, only ADDTPC and BSTART.STD CALL need patching.
    for sec in sections:
        if sec.sh_type not in (SHT_RELA, SHT_REL):
            continue
        if sec.sh_info < 0 or sec.sh_info >= len(sections):
            continue
        target = sections[sec.sh_info]
        if target.name != ".text":
            continue

        if sec.sh_entsize == 0:
            raise ValueError(f"{sec.name}: relocation section has entsize=0")

        for off in range(0, sec.sh_size, sec.sh_entsize):
            if sec.sh_type == SHT_RELA:
                r_offset, r_info, r_addend = struct.unpack_from("<QQq", data, sec.sh_offset + off)
                addend = int(r_addend)
            else:
                r_offset, r_info = struct.unpack_from("<QQ", data, sec.sh_offset + off)
                addend = 0
            r_offset = int(r_offset)
            r_info = int(r_info)
            sym_index = (r_info >> 32) & 0xFFFFFFFF

            if sym_index < 0 or sym_index >= len(symtab):
                raise ValueError(f"reloc @{r_offset:#x}: sym index out of range: {sym_index}")
            sym = symtab[sym_index]
            if sym.st_shndx == 0:
                raise ValueError(f"reloc @{r_offset:#x}: undefined symbol {sym.name!r}")
            if sym.st_shndx not in section_addrs:
                raise ValueError(f"reloc @{r_offset:#x}: no address for section {sym.st_shndx}")

            S = section_addrs[sym.st_shndx] + sym.st_value
            P = text_addr + r_offset

            if r_offset + 4 > len(text):
                raise ValueError(f"reloc @{r_offset:#x}: out of bounds for .text size {len(text)}")

            insn = int.from_bytes(text[r_offset : r_offset + 4], "little", signed=False)

            # ADDTPC: mask=0x7f match=0x07, imm20 in bits[31:12]. Base is the current 4K page.
            if (insn & 0x7F) == 0x07:
                p_page = P & ~0xFFF
                delta = (S + addend) - p_page
                if (delta & 0xFFF) != 0:
                    raise ValueError(f"ADDTPC reloc @{r_offset:#x}: delta {delta:#x} not 4K-aligned (S={S:#x} Ppage={p_page:#x})")
                imm20 = delta >> 12
                imm20_bits = imm20 & ((1 << 20) - 1)
                patched = (insn & ~(0xFFFFF << 12)) | (imm20_bits << 12)
                text[r_offset : r_offset + 4] = int(patched).to_bytes(4, "little", signed=False)
                continue

            # BSTART.STD CALL: mask=0x7fff match=0x4001, simm17 in bits[31:15] (offset in halfwords).
            if (insn & 0x7FFF) == 0x4001:
                delta = (S + addend) - P
                if (delta & 0x1) != 0:
                    raise ValueError(
                        f"BSTART.STD reloc @{r_offset:#x}: delta {delta:#x} not 2-byte aligned (S={S:#x} P={P:#x})"
                    )
                simm17 = delta >> 1
                simm17_bits = simm17 & ((1 << 17) - 1)
                patched = (insn & ~(((1 << 17) - 1) << 15)) | (simm17_bits << 15)
                text[r_offset : r_offset + 4] = int(patched).to_bytes(4, "little", signed=False)
                continue

            raise ValueError(f"unsupported relocation at .text+0x{r_offset:x}: insn=0x{insn:08x} sym={sym.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a memory init (.memh) from an ELF (relocatable or executable).")
    ap.add_argument("elf", help="Input ELF (.o or .elf)")
    ap.add_argument("-o", "--out", required=True, help="Output memh path")
    ap.add_argument("--text-base", default="0x10000", help="Base address for .text when input is ET_REL (hex)")
    ap.add_argument("--data-base", default="0x20000", help="Base address for .data when input is ET_REL (hex)")
    ap.add_argument("--page-align", default="0x1000", help="Alignment for section placement when ET_REL (hex)")
    ap.add_argument("--start-symbol", default="_start", help="Symbol to use as boot PC when emitting metadata (default: _start)")
    ap.add_argument("--print-start", action="store_true", help="Print resolved start PC (hex) to stdout")
    ns = ap.parse_args()

    path = Path(ns.elf)
    data = path.read_bytes()
    e_type, e_entry, sections, by_name = _parse_sections(data)

    text_sec = by_name.get(".text")
    if text_sec is None:
        raise SystemExit("error: missing .text section")
    text_bytes = bytearray(data[text_sec.sh_offset : text_sec.sh_offset + text_sec.sh_size])

    data_sec = by_name.get(".data")
    bss_sec = by_name.get(".bss")

    symtab_sec = by_name.get(".symtab")
    if symtab_sec is None:
        raise SystemExit("error: missing .symtab section (needed for relocations)")
    symtab = _parse_symtab(data, sections, symtab_sec)

    # Determine section runtime addresses.
    section_addrs: dict[int, int] = {}
    if e_type == ET_REL:
        page_align = int(str(ns.page_align), 0)
        text_base = int(str(ns.text_base), 0)
        data_base = int(str(ns.data_base), 0)

        text_addr = _align_up(text_base, page_align)
        section_addrs[text_sec.index] = text_addr

        data_addr = _align_up(data_base, page_align)
        if data_sec is not None:
            section_addrs[data_sec.index] = data_addr
            data_addr_end = data_addr + data_sec.sh_size
        else:
            data_addr_end = data_addr

        bss_addr = _align_up(data_addr_end, page_align)
        if bss_sec is not None:
            section_addrs[bss_sec.index] = bss_addr
    elif e_type == ET_EXEC:
        for sec in sections:
            if sec.sh_addr != 0 and sec.sh_size != 0 and sec.name:
                section_addrs[sec.index] = sec.sh_addr
        text_addr = section_addrs.get(text_sec.index, text_sec.sh_addr)
    else:
        raise SystemExit(f"error: unsupported ELF type {e_type} (expected ET_REL={ET_REL} or ET_EXEC={ET_EXEC})")

    text_addr = section_addrs[text_sec.index]

    _apply_text_relocs(
        text_bytes,
        text_addr=text_addr,
        data=data,
        sections=sections,
        symtab=symtab,
        section_addrs=section_addrs,
    )

    segments: list[tuple[int, bytes]] = [(text_addr, bytes(text_bytes))]

    if data_sec is not None:
        segments.append((section_addrs[data_sec.index], data[data_sec.sh_offset : data_sec.sh_offset + data_sec.sh_size]))
    if bss_sec is not None:
        segments.append((section_addrs[bss_sec.index], bytes([0] * bss_sec.sh_size)))

    segments.sort(key=lambda x: x[0])

    out_lines: list[str] = []
    for addr, blob in segments:
        if not blob:
            continue
        out_lines.append(f"@{addr:08x}")
        for b in blob:
            out_lines.append(f"{b:02x}")

    Path(ns.out).write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    if ns.print_start:
        start_sym = str(ns.start_symbol)
        start_pc: int | None = None
        for sym in symtab:
            if sym.name != start_sym:
                continue
            if sym.st_shndx == 0:
                continue
            if e_type == ET_REL:
                if sym.st_shndx not in section_addrs:
                    raise SystemExit(f"error: start symbol {start_sym!r} has unknown section index {sym.st_shndx}")
                start_pc = section_addrs[sym.st_shndx] + sym.st_value
            else:
                start_pc = sym.st_value
            break

        if start_pc is None and e_type == ET_EXEC and e_entry:
            start_pc = int(e_entry)

        if start_pc is None:
            raise SystemExit(f"error: start symbol {start_sym!r} not found")

        print(f"0x{start_pc:x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
