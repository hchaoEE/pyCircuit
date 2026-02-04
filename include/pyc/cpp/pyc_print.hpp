#pragma once

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <ostream>

#include "pyc_bits.hpp"
#include "pyc_fifo.hpp"
#include "pyc_interfaces.hpp"
#include "pyc_picker.hpp"
#include "pyc_primitives.hpp"
#include "pyc_queue.hpp"
#include "pyc_reg.hpp"
#include "pyc_rr_arb.hpp"
#include "pyc_sram.hpp"
#include "pyc_vec.hpp"

namespace pyc::cpp {
namespace detail {

struct StreamStateGuard {
  explicit StreamStateGuard(std::ostream &os) : os_(os), flags_(os.flags()), fill_(os.fill()) {}
  ~StreamStateGuard() {
    os_.flags(flags_);
    os_.fill(fill_);
  }

private:
  std::ostream &os_;
  std::ios::fmtflags flags_;
  char fill_;
};

template <unsigned Width>
inline void printBits(std::ostream &os, Bits<Width> v) {
  constexpr unsigned digits = (Width + 3) / 4;
  StreamStateGuard guard(os);
  os << Width << "'h" << std::hex << std::setw(digits) << std::setfill('0') << v.value();
}

} // namespace detail

template <unsigned Width>
inline std::ostream &operator<<(std::ostream &os, const Bits<Width> &v) {
  detail::printBits<Width>(os, v);
  return os;
}

template <typename T, std::size_t N>
inline std::ostream &operator<<(std::ostream &os, const Vec<T, N> &v) {
  os << "{";
  for (std::size_t i = 0; i < N; i++) {
    if (i)
      os << ", ";
    os << v[i];
  }
  os << "}";
  return os;
}

template <unsigned Width>
inline std::ostream &operator<<(std::ostream &os, const pyc_stream_if<Width> &s) {
  os << "pyc_stream_if<" << Width << ">{valid=" << s.valid << " ready=" << s.ready << " data=" << s.data << "}";
  return os;
}

template <unsigned AddrWidth, unsigned DataWidth>
inline std::ostream &operator<<(std::ostream &os, const pyc_mem_if<AddrWidth, DataWidth> &m) {
  os << "pyc_mem_if<" << AddrWidth << "," << DataWidth << ">{";
  os << "req_valid=" << m.req_valid << " req_ready=" << m.req_ready << " req_addr=" << m.req_addr;
  os << " req_write=" << m.req_write << " req_wdata=" << m.req_wdata << " req_wstrb=" << m.req_wstrb;
  os << " resp_valid=" << m.resp_valid << " resp_ready=" << m.resp_ready << " resp_rdata=" << m.resp_rdata;
  os << "}";
  return os;
}

template <unsigned Width>
inline std::ostream &operator<<(std::ostream &os, const pyc_add<Width> &m) {
  os << "pyc_add<" << Width << ">{a=" << m.a << " b=" << m.b << " y=" << m.y << "}";
  return os;
}

template <unsigned Width>
inline std::ostream &operator<<(std::ostream &os, const pyc_mux<Width> &m) {
  os << "pyc_mux<" << Width << ">{sel=" << m.sel << " a=" << m.a << " b=" << m.b << " y=" << m.y << "}";
  return os;
}

template <unsigned Width>
inline std::ostream &operator<<(std::ostream &os, const pyc_and<Width> &m) {
  os << "pyc_and<" << Width << ">{a=" << m.a << " b=" << m.b << " y=" << m.y << "}";
  return os;
}

template <unsigned Width>
inline std::ostream &operator<<(std::ostream &os, const pyc_or<Width> &m) {
  os << "pyc_or<" << Width << ">{a=" << m.a << " b=" << m.b << " y=" << m.y << "}";
  return os;
}

template <unsigned Width>
inline std::ostream &operator<<(std::ostream &os, const pyc_xor<Width> &m) {
  os << "pyc_xor<" << Width << ">{a=" << m.a << " b=" << m.b << " y=" << m.y << "}";
  return os;
}

template <unsigned Width>
inline std::ostream &operator<<(std::ostream &os, const pyc_not<Width> &m) {
  os << "pyc_not<" << Width << ">{a=" << m.a << " y=" << m.y << "}";
  return os;
}

template <unsigned Width>
inline std::ostream &operator<<(std::ostream &os, const pyc_reg<Width> &m) {
  os << "pyc_reg<" << Width << ">{en=" << m.en << " d=" << m.d << " init=" << m.init << " q=" << m.q << "}";
  return os;
}

template <unsigned Width, unsigned Depth>
inline std::ostream &operator<<(std::ostream &os, const pyc_fifo<Width, Depth> &m) {
  os << "pyc_fifo<" << Width << "," << Depth << ">{";
  os << "in_valid=" << m.in_valid << " in_ready=" << m.in_ready << " in_data=" << m.in_data;
  os << " out_valid=" << m.out_valid << " out_ready=" << m.out_ready << " out_data=" << m.out_data;
  os << " count=" << m.debug_count() << " rd=" << m.debug_rd_ptr() << " wr=" << m.debug_wr_ptr();
  os << "}";
  return os;
}

template <unsigned Width, unsigned Depth>
inline std::ostream &operator<<(std::ostream &os, const pyc_queue<Width, Depth> &m) {
  if constexpr (Depth == 1) {
    os << "pyc_queue<" << Width << "," << Depth << ">{";
    os << "in_valid=" << m.in_valid << " in_ready=" << m.in_ready << " in_data=" << m.in_data;
    os << " out_valid=" << m.out_valid << " out_ready=" << m.out_ready << " out_data=" << m.out_data;
    os << " full=" << (m.full ? 1 : 0) << " storage=" << m.storage;
    os << "}";
  } else {
    os << "pyc_queue<" << Width << "," << Depth << ">{" << m.fifo << "}";
  }
  return os;
}

template <unsigned Width, std::size_t N>
inline std::ostream &operator<<(std::ostream &os, const pyc_picker_onehot<Width, N> &m) {
  os << "pyc_picker_onehot<" << Width << "," << N << ">{sel=" << m.sel << " in_data=" << m.in_data << " y=" << m.y
     << "}";
  return os;
}

template <unsigned Width, std::size_t N>
inline std::ostream &operator<<(std::ostream &os, const pyc_rr_arb<Width, N> &m) {
  os << "pyc_rr_arb<" << Width << "," << N << ">{";
  os << "in_valid=" << m.in_valid << " in_ready=" << m.in_ready << " out_valid=" << m.out_valid
     << " out_ready=" << m.out_ready << " out_data=" << m.out_data << " out_sel=" << m.out_sel;
  os << " rr_ptr=" << m.rr_ptr << " selected={valid=" << (m.selected.valid ? 1 : 0) << " idx=" << m.selected.idx
     << "}";
  os << "}";
  return os;
}

template <unsigned AddrWidth, unsigned DataWidth, std::size_t Depth>
inline std::ostream &operator<<(std::ostream &os, const pyc_sram<AddrWidth, DataWidth, Depth> &m) {
  os << "pyc_sram<" << AddrWidth << "," << DataWidth << "," << Depth << ">{";
  os << "req_valid=" << m.req_valid << " req_ready=" << m.req_ready << " req_addr=" << m.req_addr
     << " req_write=" << m.req_write << " req_wdata=" << m.req_wdata << " req_wstrb=" << m.req_wstrb;
  os << " resp_valid=" << m.resp_valid << " resp_ready=" << m.resp_ready << " resp_rdata=" << m.resp_rdata;
  os << " pending=" << (m.pending ? 1 : 0);
  os << "}";
  return os;
}

template <unsigned Width>
inline std::ostream &operator<<(std::ostream &os, const Reg<Width> &r) {
  os << "Reg<" << Width << ">{q=" << r.q << " init=" << r.init << "}";
  return os;
}

template <unsigned Width, unsigned Depth>
inline std::ostream &operator<<(std::ostream &os, const Fifo<Width, Depth> &f) {
  os << "Fifo<" << Width << "," << Depth << ">{out_valid=" << (f.out_valid ? 1 : 0) << " in_ready=" << (f.in_ready ? 1 : 0)
     << " out_data=" << f.out_data << "}";
  return os;
}

} // namespace pyc::cpp
