# BF16 Fused Multiply-Accumulate (FMAC)

A BF16 floating-point fused multiply-accumulate unit with 4-stage pipeline,
built from primitive standard cells (half adders, full adders, MUXes).

## Operation

```
acc_out (FP32) = acc_in (FP32) + a (BF16) × b (BF16)
```

## Formats

| Format | Bits | Layout | Bias |
|--------|------|--------|------|
| BF16 | 16 | sign(1) \| exp(8) \| mantissa(7) | 127 |
| FP32 | 32 | sign(1) \| exp(8) \| mantissa(23) | 127 |

## 4-Stage Pipeline

| Stage | Function | Critical Path Depth |
|-------|----------|-------------------|
| 1 | Unpack BF16, exponent addition | 8 |
| 2 | 8×8 mantissa multiply (Wallace tree) | 46 |
| 3 | Align exponents, add mantissas | 21 |
| 4 | Normalize (LZC + barrel shift), pack FP32 | 27 |

## Design Hierarchy

```
bf16_fmac.py (top level)
└── primitive_standard_cells.py
    ├── half_adder, full_adder        (1-bit)
    ├── ripple_carry_adder            (N-bit)
    ├── partial_product_array         (AND gate array)
    ├── compress_3to2 (CSA)           (carry-save adder)
    ├── reduce_partial_products       (Wallace tree)
    ├── unsigned_multiplier           (N×M multiply)
    ├── barrel_shift_right/left       (MUX layers)
    └── leading_zero_count            (priority encoder)
```

## Files

| File | Description |
|------|-------------|
| `primitive_standard_cells.py` | HA, FA, RCA, CSA, multiplier, shifters, LZC |
| `bf16_fmac.py` | 4-stage pipelined FMAC |
| `fmac_capi.cpp` | C API wrapper |
| `test_bf16_fmac.py` | 100 test cases (true RTL simulation) |

## Build & Run

```bash
# 1. Compile RTL
PYTHONPATH=python:. python -m pycircuit.cli emit \
    examples/fmac/bf16_fmac.py \
    -o examples/generated/fmac/bf16_fmac.pyc
build/bin/pyc-compile examples/generated/fmac/bf16_fmac.pyc \
    --emit=cpp -o examples/generated/fmac/bf16_fmac_gen.hpp

# 2. Build shared library
c++ -std=c++17 -O2 -shared -fPIC -I include -I . \
    -o examples/fmac/libfmac_sim.dylib examples/fmac/fmac_capi.cpp

# 3. Run 100 test cases
python examples/fmac/test_bf16_fmac.py
```
