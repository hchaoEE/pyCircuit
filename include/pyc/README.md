# `include/pyc/`

Backend “template libraries” used by code generators:

- `include/pyc/cpp/`: cycle-accurate C++ models (header-only, template-heavy)
- `include/pyc/verilog/`: Verilog primitives used by emitted RTL

Generated code instantiates these templates for **stateful** primitives (regs,
FIFOs, memories). Most **combinational** logic is emitted inline as expressions
(Verilog operators / C++ `Wire<>` operator overloads) to keep output netlist-like
and easy to read.

## Primitive API (prototype)

The intent is for each primitive to exist in both backends with the same name and
port names (e.g. `pyc_add` / `pyc_reg` / `pyc_fifo`), so MLIR lowering and codegen
can stay backend-agnostic.

Examples:

- Verilog: `include/pyc/verilog/pyc_reg.v` defines `module pyc_reg #(WIDTH) (...)`
- C++: `include/pyc/cpp/pyc_primitives.hpp` defines `template<unsigned Width> class pyc::cpp::pyc_reg { ... }`

Current checked-in primitives (prototype):

- Combinational: (usually emitted inline; wrappers exist but are not required)
- Sequential: `pyc_reg`
- Ready/valid: `pyc_fifo` (strict handshake, single-clock)
- Memory: `pyc_byte_mem` (byte-addressed, async read + sync write, prototype)

Debug/testbench helpers (C++ only):

- `include/pyc/cpp/pyc_print.hpp`: `operator<<` for wires and primitives
- `include/pyc/cpp/pyc_tb.hpp`: small multi-clock-capable testbench harness
- `include/pyc/cpp/pyc_vcd.hpp`: tiny VCD writer (waveform dumping)
- Convenience include: `include/pyc/cpp/pyc_debug.hpp`
