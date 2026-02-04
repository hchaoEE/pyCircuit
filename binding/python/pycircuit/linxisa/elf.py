from __future__ import annotations

import struct
from pathlib import Path


def read_elf64_section(path: str | Path, section_name: str) -> bytes:
    p = Path(path)
    data = p.read_bytes()
    if data[:4] != b"\x7fELF":
        raise ValueError(f"not an ELF file: {p}")
    if data[4] != 2 or data[5] != 1:
        raise ValueError(f"unsupported ELF class/endian (need 64-bit LSB): {p}")

    # Elf64_Ehdr: 16s H H I Q Q Q I H H H H H H
    (_, _, _, _, _, _, e_shoff, _, _, _, _, e_shentsize, e_shnum, e_shstrndx) = struct.unpack_from(
        "<16sHHIQQQIHHHHHH", data, 0
    )

    def read_sh(i: int) -> tuple[int, int, int]:
        sh = struct.unpack_from("<IIQQQQIIQQ", data, e_shoff + i * e_shentsize)
        sh_name = int(sh[0])
        sh_off = int(sh[4])
        sh_size = int(sh[5])
        return sh_name, sh_off, sh_size

    shstr_name_off, shstr_off, shstr_size = read_sh(e_shstrndx)
    _ = shstr_name_off
    shstr = data[shstr_off : shstr_off + shstr_size]

    def get_name(off: int) -> str:
        end = shstr.find(b"\x00", off)
        if end < 0:
            end = len(shstr)
        return shstr[off:end].decode("utf-8")

    for i in range(e_shnum):
        name_off, off, size = read_sh(i)
        if get_name(name_off) == section_name:
            return data[off : off + size]

    raise KeyError(section_name)

