# PYC Primitive API (prototype)

This file documents the *single-source* “contract” we want to keep stable as
`pyCircuit` grows: every primitive has a **matching C++ template** and a
**matching SystemVerilog module** with the same name and port names.

The MLIR dialect (`pyc.*` ops) should lower to this primitive layer.

## 1) Common data structures

### 1.1 `Wire<W>`

Represents a combinational value of width `W` bits.

- C++: `pyc::cpp::Wire<W>` (see `include/pyc/cpp/pyc_bits.hpp`)
- Verilog: `logic [W-1:0]`

### 1.2 `Reg<W>` (module-like)

Represents a clocked storage element with synchronous reset and clock enable.

- Verilog module: `pyc_reg` (`include/pyc/verilog/pyc_reg.sv`)
- C++ class: `pyc::cpp::pyc_reg<W>` (`include/pyc/cpp/pyc_primitives.hpp`)

Ports:

- `clk` (i1 / logic)
- `rst` (i1 / logic)
- `en` (i1 / logic)
- `d` (W-bit)
- `init` (W-bit)
- `q` (W-bit)

Semantics (posedge `clk`):

- if `rst`: `q <= init`
- else if `en`: `q <= d`

### 1.3 `Stream<W>` (ready/valid)

Bundle for strict ready/valid handshake.

- C++: `pyc::cpp::Stream<W>` (`include/pyc/cpp/pyc_stream.hpp`)
- Verilog: `pyc_stream_if #(WIDTH=W)` interface (`include/pyc/verilog/pyc_stream_if.sv`)

Signals:

- `valid` (producer → consumer)
- `ready` (consumer → producer)
- `data` (producer → consumer)

Convenience:

- Verilog: `pyc_handshake_pkg::fire(valid, ready)` (`include/pyc/verilog/pyc_handshake_pkg.sv`)
- C++: `pyc::cpp::fire(valid, ready)` (`include/pyc/cpp/pyc_handshake.hpp`)

### 1.4 `Vec<T, N>`

Fixed-size container (useful for regfiles, bundles of lanes, etc.).

- C++: `pyc::cpp::Vec<T, N>` (`include/pyc/cpp/pyc_vec.hpp`)
- Verilog: use unpacked arrays (`T v [0:N-1]`) or packed arrays, depending on style.

## 2) Primitive operations (combinational)

All combinational primitives have an `eval()` method in C++ and continuous
assign semantics in Verilog.

### 2.1 `pyc_add` (W-bit)

- Verilog: `module pyc_add #(WIDTH) (a, b, y)`
- C++: `pyc::cpp::pyc_add<W> { a, b, y; eval(); }`

### 2.2 `pyc_mux` (W-bit)

- Verilog: `module pyc_mux #(WIDTH) (sel, a, b, y)`
- C++: `pyc::cpp::pyc_mux<W> { sel, a, b, y; eval(); }`

### 2.3 Bitwise ops (W-bit)

- `pyc_and`: `(a, b) -> y`
- `pyc_or`: `(a, b) -> y`
- `pyc_xor`: `(a, b) -> y`
- `pyc_not`: `(a) -> y`

## 3) Primitive operations (ready/valid)

### 3.1 `pyc_fifo` (single-clock, strict ready/valid)

- Verilog: `module pyc_fifo #(WIDTH, DEPTH) (...)`
- C++: `pyc::cpp::pyc_fifo<Width, Depth>` (`include/pyc/cpp/pyc_primitives.hpp`)

Ports (explicit, for compatibility with simple codegen):

- `clk`, `rst`
- input: `in_valid`, `in_ready`, `in_data`
- output: `out_valid`, `out_ready`, `out_data`

Handshake:

- push when `in_valid && in_ready`
- pop when `out_valid && out_ready`

Note: this is currently **single-clock**; async FIFO should be a separate
primitive.

### 3.2 `pyc_queue` (fall-through queue)

- Verilog: `module pyc_queue #(WIDTH, DEPTH) (...)`
- C++: `pyc::cpp::pyc_queue<Width, Depth>` (`include/pyc/cpp/pyc_queue.hpp`)

Ports match `pyc_fifo`. When `DEPTH==1`, it behaves like a standard fall-through
skid buffer (0-cycle when downstream is ready, buffers 1 element when stalled).

## 4) Arbitration / Picking

### 4.1 `pyc_rr_arb` (round-robin arbiter)

- Verilog: `module pyc_rr_arb #(WIDTH, N) (...)`
- C++: `pyc::cpp::pyc_rr_arb<Width, N>` (`include/pyc/cpp/pyc_rr_arb.hpp`)

Ports:

- `in_valid[N]`, `in_ready[N]`, `in_data[N]`
- `out_valid`, `out_ready`, `out_data`, `out_sel`

### 4.2 `pyc_picker_onehot`

- Verilog: `module pyc_picker_onehot #(WIDTH, N) (sel, in_data, y)`
- C++: `pyc::cpp::pyc_picker_onehot<Width, N>` (`include/pyc/cpp/pyc_picker.hpp`)

## 5) Memory / SRAM

### 5.1 `pyc_mem_if` (interface / bundle)

- Verilog: `interface pyc_mem_if #(ADDR_WIDTH, DATA_WIDTH)` (`include/pyc/verilog/pyc_mem_if.sv`)
- C++: `pyc::cpp::pyc_mem_if<AddrWidth, DataWidth>` (`include/pyc/cpp/pyc_interfaces.hpp`)

Signals:

- request: `req_valid`, `req_ready`, `req_addr`, `req_write`, `req_wdata`, `req_wstrb`
- response: `resp_valid`, `resp_ready`, `resp_rdata`

### 5.2 `pyc_sram` (single outstanding)

- Verilog: `module pyc_sram #(ADDR_WIDTH, DATA_WIDTH, DEPTH) (...)` (`include/pyc/verilog/pyc_sram.sv`)
- C++: `pyc::cpp::pyc_sram<AddrWidth, DataWidth, Depth>` (`include/pyc/cpp/pyc_sram.hpp`)

## 6) Debugging / Testbench (C++)

Prototype-only utilities to help with bring-up and debugging:

- Printing: `include/pyc/cpp/pyc_print.hpp` defines `operator<<` for `Wire`, `Vec`, interfaces, and primitives.
- Testbench: `include/pyc/cpp/pyc_tb.hpp` provides `pyc::cpp::Testbench<Dut>` (multi-clock ready).
- Convenience include: `include/pyc/cpp/pyc_debug.hpp`.

Example C++ testbench:

- `examples/cpp/tb_fifo.cpp`
