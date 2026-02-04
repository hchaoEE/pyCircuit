#pragma once

#include "pyc_bits.hpp"

namespace pyc::cpp {

inline bool fire(const Wire<1> &valid, const Wire<1> &ready) { return valid.toBool() && ready.toBool(); }

} // namespace pyc::cpp

