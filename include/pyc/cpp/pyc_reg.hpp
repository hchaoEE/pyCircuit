#pragma once

#include "pyc_bits.hpp"
#include "pyc_clock.hpp"

namespace pyc::cpp {

template <unsigned Width>
class Reg {
public:
  Reg() = default;
  explicit Reg(ClockDomain &domain, Wire<Width> initValue = Wire<Width>{}) { attach(domain, initValue); }

  void attach(ClockDomain &domain, Wire<Width> initValue = Wire<Width>{}) {
    domain_ = &domain;
    init = initValue;
    q = initValue;
  }

  void setNext(Wire<Width> nextValue, bool enable = true) {
    next_ = nextValue;
    en_ = enable;
  }

  void tick() {
    if (!domain_)
      return;
    if (domain_->rst)
      q = init;
    else if (en_)
      q = next_;
  }

  Wire<Width> q{};
  Wire<Width> init{};

private:
  ClockDomain *domain_ = nullptr;
  Wire<Width> next_{};
  bool en_ = false;
};

} // namespace pyc::cpp

