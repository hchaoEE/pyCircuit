# Examples

Emit `.pyc` (MLIR) from Python:

```bash
PYTHONPATH=../binding/python python3 -m pycircuit.cli emit counter.py -o /tmp/counter.pyc
PYTHONPATH=../binding/python python3 -m pycircuit.cli emit fifo_loopback.py -o /tmp/fifo_loopback.pyc
PYTHONPATH=../binding/python python3 -m pycircuit.cli emit multiclock_regs.py -o /tmp/multiclock_regs.pyc
PYTHONPATH=../binding/python python3 -m pycircuit.cli emit wire_ops.py -o /tmp/wire_ops.pyc
PYTHONPATH=../binding/python python3 -m pycircuit.cli emit jit_control_flow.py -o /tmp/jit_control_flow.pyc
PYTHONPATH=../binding/python python3 -m pycircuit.cli emit jit_pipeline_vec.py -o /tmp/jit_pipeline_vec.pyc
```

Then compile to Verilog:

```bash
../pyc/mlir/build/bin/pyc-compile /tmp/counter.pyc --emit=verilog -o /tmp/counter.sv
```

## Checked-in generated outputs

This repo checks in generated outputs under `examples/generated/`:

```bash
bash examples/update_generated.sh
```

## Generated outputs (checked in)

This repo checks in generated `*.sv` and `*.hpp` outputs under `examples/generated/`.

Regenerate (all examples + Linx CPU):

```bash
bash examples/update_generated.sh
```
