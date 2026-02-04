#pragma once

#include <cstdint>

namespace pyc::cpp {

template <unsigned Width>
class Bits {
public:
  static_assert(Width > 0 && Width <= 64, "pyc::cpp::Bits supports widths 1..64 in the prototype");

  using storage_type = std::uint64_t;

  constexpr Bits() = default;
  constexpr explicit Bits(storage_type v) : v_(v & mask()) {}

  constexpr storage_type value() const { return v_; }
  constexpr bool toBool() const { return (v_ & 1u) != 0; }

  static constexpr storage_type mask() {
    if constexpr (Width == 64)
      return ~storage_type{0};
    return (storage_type{1} << Width) - 1;
  }

  friend constexpr Bits operator+(Bits a, Bits b) { return Bits(a.v_ + b.v_); }
  friend constexpr Bits operator&(Bits a, Bits b) { return Bits(a.v_ & b.v_); }
  friend constexpr Bits operator|(Bits a, Bits b) { return Bits(a.v_ | b.v_); }
  friend constexpr Bits operator^(Bits a, Bits b) { return Bits(a.v_ ^ b.v_); }
  friend constexpr Bits operator~(Bits a) { return Bits(~a.v_); }

  friend constexpr bool operator==(Bits a, Bits b) { return a.value() == b.value(); }
  friend constexpr bool operator!=(Bits a, Bits b) { return !(a == b); }

private:
  storage_type v_ = 0;
};

template <unsigned Width>
using Wire = Bits<Width>;

template <unsigned OutWidth, unsigned InWidth>
constexpr Wire<OutWidth> trunc(Wire<InWidth> v) {
  static_assert(OutWidth > 0 && OutWidth <= 64, "trunc supports widths 1..64");
  static_assert(InWidth > 0 && InWidth <= 64, "trunc supports widths 1..64");
  static_assert(OutWidth <= InWidth, "trunc requires OutWidth <= InWidth");
  return Wire<OutWidth>(v.value());
}

template <unsigned OutWidth, unsigned InWidth>
constexpr Wire<OutWidth> zext(Wire<InWidth> v) {
  static_assert(OutWidth > 0 && OutWidth <= 64, "zext supports widths 1..64");
  static_assert(InWidth > 0 && InWidth <= 64, "zext supports widths 1..64");
  static_assert(OutWidth >= InWidth, "zext requires OutWidth >= InWidth");
  return Wire<OutWidth>(v.value());
}

template <unsigned OutWidth, unsigned InWidth>
constexpr Wire<OutWidth> sext(Wire<InWidth> v) {
  static_assert(OutWidth > 0 && OutWidth <= 64, "sext supports widths 1..64");
  static_assert(InWidth > 0 && InWidth <= 64, "sext supports widths 1..64");
  static_assert(OutWidth >= InWidth, "sext requires OutWidth >= InWidth");

  if constexpr (InWidth == 64) {
    return Wire<OutWidth>(v.value());
  } else {
    std::uint64_t x = v.value();
    std::uint64_t signBit = std::uint64_t{1} << (InWidth - 1);
    if (x & signBit)
      x |= (~std::uint64_t{0}) << InWidth;
    return Wire<OutWidth>(x);
  }
}

template <unsigned OutWidth, unsigned InWidth>
constexpr Wire<OutWidth> extract(Wire<InWidth> v, unsigned lsb) {
  static_assert(OutWidth > 0 && OutWidth <= 64, "extract supports widths 1..64");
  static_assert(InWidth > 0 && InWidth <= 64, "extract supports widths 1..64");
  return Wire<OutWidth>(v.value() >> lsb);
}

} // namespace pyc::cpp
