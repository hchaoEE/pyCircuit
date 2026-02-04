#pragma once

#include <array>
#include <cstddef>

#include "pyc_bits.hpp"
#include "pyc_clock.hpp"

namespace pyc::cpp {

// Strict ready/valid FIFO (single-clock, prototype).
//
// - Push on (in_valid && in_ready) at tick()
// - Pop  on (out_valid && out_ready) at tick()
// - out_data is combinational and only meaningful when out_valid is true.
template <unsigned Width, unsigned Depth>
class Fifo {
public:
  static_assert(Width > 0 && Width <= 64, "pyc::cpp::Fifo supports widths 1..64 in the prototype");
  static_assert(Depth > 0, "pyc::cpp::Fifo Depth must be > 0");

  Fifo() = default;
  explicit Fifo(ClockDomain &domain) { attach(domain); }

  void attach(ClockDomain &domain) {
    domain_ = &domain;
    resetState();
  }

  // Combinational interface evaluation: update in_ready/out_valid/out_data.
  void eval(bool in_valid, Wire<Width> in_data, bool out_ready) {
    in_valid_ = in_valid;
    in_data_ = in_data;
    out_ready_ = out_ready;

    in_ready = (count_ < Depth) || (out_ready_ && (count_ != 0));
    out_valid = (count_ != 0);
    out_data = storage_[rd_];
  }

  // Sequential update on clock edge (uses domain_->rst).
  void tick() {
    if (!domain_)
      return;
    if (domain_->rst) {
      resetState();
      return;
    }

    bool do_pop = out_valid && out_ready_;
    bool do_push = in_valid_ && in_ready;

    // Pop then push (preserves ordering).
    if (do_pop) {
      rd_ = bump(rd_);
      count_--;
    }
    if (do_push) {
      storage_[wr_] = in_data_;
      wr_ = bump(wr_);
      count_++;
    }
  }

  // Output pins.
  bool in_ready = false;
  bool out_valid = false;
  Wire<Width> out_data{};

private:
  static constexpr std::size_t bump(std::size_t p) { return (p + 1 >= Depth) ? 0 : (p + 1); }

  void resetState() {
    rd_ = 0;
    wr_ = 0;
    count_ = 0;
  }

  ClockDomain *domain_ = nullptr;
  std::array<Wire<Width>, Depth> storage_{};
  std::size_t rd_ = 0;
  std::size_t wr_ = 0;
  std::size_t count_ = 0;

  // Latched inputs for tick().
  bool in_valid_ = false;
  bool out_ready_ = false;
  Wire<Width> in_data_{};
};

} // namespace pyc::cpp
