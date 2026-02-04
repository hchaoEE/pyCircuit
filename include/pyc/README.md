# `include/pyc/`

Backend “template libraries” used by code generators:

- `include/pyc/cpp/`: cycle-accurate C++ models (header-only, template-heavy)
- `include/pyc/verilog/`: Verilog/SystemVerilog primitives used by emitted RTL

Generated code should only need to include/instantiate these templates.

## Primitive API (prototype)

The intent is for each primitive to exist in both backends with the same name and
port names (e.g. `pyc_add` / `pyc_reg` / `pyc_fifo`), so MLIR lowering and codegen
can stay backend-agnostic.

Examples:

- Verilog: `include/pyc/verilog/pyc_add.sv` defines `module pyc_add #(WIDTH) (a, b, y)`
- C++: `include/pyc/cpp/pyc_primitives.hpp` defines `template<unsigned Width> struct pyc::cpp::pyc_add { a, b, y; eval(); }`

Additional conventional RTL building blocks (prototype):

- Ready/valid: `pyc_queue`, `pyc_rr_arb`, `pyc_picker_onehot`
- Memory: `pyc_mem_if` + `pyc_sram`

Debug/testbench helpers (C++ only):

- `include/pyc/cpp/pyc_print.hpp`: `operator<<` for wires, interfaces, and primitives
- `include/pyc/cpp/pyc_tb.hpp`: small multi-clock-capable testbench harness
- Convenience include: `include/pyc/cpp/pyc_debug.hpp`
