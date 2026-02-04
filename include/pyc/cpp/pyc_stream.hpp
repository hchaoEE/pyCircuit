#pragma once

#include "pyc_interfaces.hpp"

namespace pyc::cpp {

template <unsigned Width>
using Stream = pyc_stream_if<Width>;

} // namespace pyc::cpp
