#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=../../scripts/lib.sh
source "${ROOT_DIR}/scripts/lib.sh"
pyc_find_pyc_compile

GEN_DIR="${ROOT_DIR}/janus/generated/janus_bcc_ooo_pyc"
HDR="${GEN_DIR}/janus_bcc_ooo_pyc_gen.hpp"
if [[ ! -f "${HDR}" ]]; then
  bash "${ROOT_DIR}/janus/update_generated.sh"
fi

TB_SRC="${ROOT_DIR}/janus/tb/tb_janus_bcc_ooo_pyc.cpp"
TB_EXE="${GEN_DIR}/tb_janus_bcc_ooo_pyc_cpp"

need_build=0
if [[ ! -x "${TB_EXE}" ]]; then
  need_build=1
elif [[ "${TB_SRC}" -nt "${TB_EXE}" || "${HDR}" -nt "${TB_EXE}" ]]; then
  need_build=1
fi

if [[ "${need_build}" -ne 0 ]]; then
  mkdir -p "${GEN_DIR}"
  tmp_exe="${TB_EXE}.tmp.$$"
  "${CXX:-clang++}" -std=c++17 -O3 -DNDEBUG \
    -I "${ROOT_DIR}/include" \
    -I "${GEN_DIR}" \
    -o "${tmp_exe}" \
    "${TB_SRC}"
  mv -f "${tmp_exe}" "${TB_EXE}"
fi

if [[ $# -gt 0 ]]; then
  "${TB_EXE}" "$@"
else
  "${TB_EXE}"
fi
