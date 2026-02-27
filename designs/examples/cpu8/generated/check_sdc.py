#!/usr/bin/env python3
"""
check_sdc.py — Timing constraint completeness & consistency checker for cpu8.

Parses the Verilog netlist and the SDC file, then checks:

  Completeness
  ~~~~~~~~~~~~
  C1  Every clock port has a create_clock.
  C2  Every non-clock input port has set_input_delay or is covered by a false path.
  C3  Every output port has set_output_delay or is covered by a false path.
  C4  Clock uncertainty is specified (setup & hold).
  C5  All sequential elements share a defined clock domain.
  C6  set_max_transition and set_max_fanout are present.

  Consistency
  ~~~~~~~~~~~
  K1  No input port appears in both set_input_delay AND an unconditional
      set_false_path -from.
  K2  No output port appears in both set_output_delay AND an unconditional
      set_false_path -to (unless the false-path is intentional, e.g. constant r0).
  K3  input_delay_max >= input_delay_min for every constrained input.
  K4  output_delay_max >= output_delay_min for every constrained output.
  K5  input_delay_max + output_delay_max < clock_period (basic feasibility).
  K6  Clock period > 0 and waveform rise < fall < period.
  K7  Clock uncertainty < clock_period / 2 (sanity).
  K8  No duplicate create_clock on the same port.

Usage:
    python check_sdc.py <verilog_file> <sdc_file>
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ─────────────────────── Verilog port extraction ────────────────────────── #

@dataclass
class VerilogDesign:
    module_name: str = ""
    clock_ports: list[str] = field(default_factory=list)
    reset_ports: list[str] = field(default_factory=list)
    input_ports: list[str] = field(default_factory=list)
    output_ports: list[str] = field(default_factory=list)
    reg_instances: list[str] = field(default_factory=list)
    reg_clk_connections: dict[str, str] = field(default_factory=dict)


def parse_verilog(path: Path) -> VerilogDesign:
    text = path.read_text()
    d = VerilogDesign()

    m = re.search(r"module\s+(\w+)\s*\(", text)
    if m:
        d.module_name = m.group(1)

    port_block = re.search(r"module\s+\w+\s*\((.*?)\);", text, re.DOTALL)
    if not port_block:
        return d

    for line in port_block.group(1).splitlines():
        line = line.strip().rstrip(",")
        if not line or line.startswith("//"):
            continue

        im = re.match(r"input\s+(?:\[[\d:]+\]\s*)?(\w+)", line)
        om = re.match(r"output\s+(?:\[[\d:]+\]\s*)?(\w+)", line)
        if im:
            port_name = im.group(1)
            if port_name == "clk" or port_name.startswith("clk_"):
                d.clock_ports.append(port_name)
            elif port_name == "rst" or port_name.startswith("rst_"):
                d.reset_ports.append(port_name)
            else:
                d.input_ports.append(port_name)
        elif om:
            d.output_ports.append(om.group(1))

    for m in re.finditer(r"pyc_reg\s+#\(.*?\)\s+(\w+)\s*\(", text, re.DOTALL):
        inst_name = m.group(1)
        d.reg_instances.append(inst_name)
        block_start = m.end()
        block_end = text.find(");", block_start)
        block = text[block_start:block_end] if block_end > 0 else ""
        clk_m = re.search(r"\.clk\((\w+)\)", block)
        if clk_m:
            d.reg_clk_connections[inst_name] = clk_m.group(1)

    return d


# ─────────────────────── SDC parsing ────────────────────────────────────── #

@dataclass
class SdcInfo:
    clocks: dict[str, dict] = field(default_factory=dict)
    input_delays: dict[str, dict] = field(default_factory=dict)
    output_delays: dict[str, dict] = field(default_factory=dict)
    false_path_from: list[str] = field(default_factory=list)
    false_path_to: list[str] = field(default_factory=list)
    clock_uncertainty_setup: float | None = None
    clock_uncertainty_hold: float | None = None
    has_max_transition: bool = False
    has_max_fanout: bool = False
    raw_lines: list[str] = field(default_factory=list)


def _expand_port_pattern(pattern: str) -> str:
    """Extract base port name from SDC get_ports expressions."""
    m = re.search(r"get_ports\s+(\w+)", pattern)
    if m:
        return m.group(1)
    m = re.search(r"get_ports\s+(\w+)\[", pattern)
    if m:
        return m.group(1)
    m = re.search(r"get_ports\s+\{?(\w+)", pattern)
    if m:
        return m.group(1)
    return pattern


def parse_sdc(path: Path) -> SdcInfo:
    text = path.read_text()
    resolved: dict[str, str] = {}
    info = SdcInfo()

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        info.raw_lines.append(line)

        sm = re.match(r"set\s+(\w+)\s+([\d.]+)", line)
        if sm:
            resolved[sm.group(1)] = sm.group(2)
            continue

    def resolve(token: str) -> str:
        t = token.strip().lstrip("$")
        return resolved.get(t, token.lstrip("$"))

    for line in info.raw_lines:
        # create_clock
        cc = re.match(
            r"create_clock\s+.*-name\s+(\w+)\s+.*-period\s+([\d.\w$]+)\s+"
            r".*-waveform\s+\{([\d.\s]+)\}.*\[get_ports\s+(\w+)\]",
            line,
        )
        if cc:
            name = cc.group(1)
            period = float(resolve(cc.group(2)))
            wave = [float(x) for x in cc.group(3).split()]
            port = cc.group(4)
            info.clocks[name] = {
                "period": period, "rise": wave[0], "fall": wave[1], "port": port,
            }
            continue

        # set_clock_uncertainty
        cu = re.match(
            r"set_clock_uncertainty\s+-(setup|hold)\s+([\d.\w$]+)", line,
        )
        if cu:
            val = float(resolve(cu.group(2)))
            if cu.group(1) == "setup":
                info.clock_uncertainty_setup = val
            else:
                info.clock_uncertainty_hold = val
            continue

        # set_input_delay
        sid = re.match(
            r"set_input_delay\s+.*-clock\s+(\w+)\s+-(max|min)\s+([\d.\w$]+)\s+"
            r".*(\[get_ports\s+\S+\])",
            line,
        )
        if sid:
            port = _expand_port_pattern(sid.group(4))
            kind = sid.group(2)
            val = float(resolve(sid.group(3)))
            info.input_delays.setdefault(port, {})[kind] = val
            info.input_delays[port]["clock"] = sid.group(1)
            continue

        # set_output_delay
        sod = re.match(
            r"set_output_delay\s+.*-clock\s+(\w+)\s+-(max|min)\s+([\d.\w$]+)\s+"
            r".*(\[get_ports\s+\S+\])",
            line,
        )
        if sod:
            port = _expand_port_pattern(sod.group(4))
            kind = sod.group(2)
            val = float(resolve(sod.group(3)))
            info.output_delays.setdefault(port, {})[kind] = val
            info.output_delays[port]["clock"] = sod.group(1)
            continue

        # set_false_path
        fp_from = re.search(r"set_false_path\s+.*-from\s+\[get_ports\s+(\w+)", line)
        if fp_from:
            info.false_path_from.append(fp_from.group(1))
        fp_to = re.search(r"set_false_path\s+.*-to\s+\[get_ports\s+(\w+)", line)
        if fp_to:
            info.false_path_to.append(fp_to.group(1))

        # set_max_transition / set_max_fanout
        if re.match(r"set_max_transition\b", line):
            info.has_max_transition = True
        if re.match(r"set_max_fanout\b", line):
            info.has_max_fanout = True

    return info


# ─────────────────────── Checks ─────────────────────────────────────────── #

@dataclass
class CheckResult:
    tag: str
    ok: bool
    message: str


def run_checks(design: VerilogDesign, sdc: SdcInfo) -> list[CheckResult]:
    results: list[CheckResult] = []

    def ok(tag: str, msg: str) -> None:
        results.append(CheckResult(tag, True, msg))

    def fail(tag: str, msg: str) -> None:
        results.append(CheckResult(tag, False, msg))

    # ── C1: clock definition ──
    for cp in design.clock_ports:
        found = any(c["port"] == cp for c in sdc.clocks.values())
        if found:
            ok("C1", f"Clock port '{cp}' has create_clock")
        else:
            fail("C1", f"Clock port '{cp}' missing create_clock")

    if not sdc.clocks:
        fail("C1", "No create_clock found in SDC")

    # ── C2: input delay or false-path ──
    all_data_inputs = design.input_ports + design.reset_ports
    for p in all_data_inputs:
        has_delay = p in sdc.input_delays
        has_fp = p in sdc.false_path_from
        if has_delay or has_fp:
            ok("C2", f"Input '{p}' constrained (delay={has_delay}, false_path={has_fp})")
        else:
            fail("C2", f"Input '{p}' has no set_input_delay and no false_path")

    # ── C3: output delay or false-path ──
    for p in design.output_ports:
        has_delay = p in sdc.output_delays
        has_fp = p in sdc.false_path_to
        if has_delay or has_fp:
            ok("C3", f"Output '{p}' constrained (delay={has_delay}, false_path={has_fp})")
        else:
            fail("C3", f"Output '{p}' has no set_output_delay and no false_path")

    # ── C4: clock uncertainty ──
    if sdc.clock_uncertainty_setup is not None:
        ok("C4", f"Clock uncertainty (setup) = {sdc.clock_uncertainty_setup} ns")
    else:
        fail("C4", "Missing set_clock_uncertainty -setup")
    if sdc.clock_uncertainty_hold is not None:
        ok("C4", f"Clock uncertainty (hold) = {sdc.clock_uncertainty_hold} ns")
    else:
        fail("C4", "Missing set_clock_uncertainty -hold")

    # ── C5: all regs on a defined clock ──
    defined_clk_ports = {c["port"] for c in sdc.clocks.values()}
    for inst, clk_net in design.reg_clk_connections.items():
        if clk_net in defined_clk_ports:
            ok("C5", f"Reg '{inst}' clocked by defined clock '{clk_net}'")
        else:
            fail("C5", f"Reg '{inst}' clocked by '{clk_net}' which has no create_clock")
    if not design.reg_instances:
        fail("C5", "No sequential elements found in the design")

    # ── C6: max_transition / max_fanout ──
    if sdc.has_max_transition:
        ok("C6", "set_max_transition present")
    else:
        fail("C6", "Missing set_max_transition")
    if sdc.has_max_fanout:
        ok("C6", "set_max_fanout present")
    else:
        fail("C6", "Missing set_max_fanout")

    # ── K1: no conflict input_delay + false_path_from ──
    for p in sdc.false_path_from:
        if p in sdc.input_delays:
            fail("K1", f"Input '{p}' has BOTH set_input_delay AND set_false_path -from (conflict)")
        else:
            ok("K1", f"Input '{p}' false_path has no conflicting input_delay")

    # ── K2: output_delay + false_path_to ──
    for p in sdc.false_path_to:
        if p in sdc.output_delays:
            fail("K2", f"Output '{p}' has BOTH set_output_delay AND set_false_path -to (conflict)")
        else:
            ok("K2", f"Output '{p}' false_path has no conflicting output_delay (intentional: constant)")

    # ── K3: input_delay max >= min ──
    for p, d in sdc.input_delays.items():
        mx = d.get("max", 0)
        mn = d.get("min", 0)
        if mx >= mn:
            ok("K3", f"Input '{p}' delay max({mx}) >= min({mn})")
        else:
            fail("K3", f"Input '{p}' delay max({mx}) < min({mn}) — inverted!")

    # ── K4: output_delay max >= min ──
    for p, d in sdc.output_delays.items():
        mx = d.get("max", 0)
        mn = d.get("min", 0)
        if mx >= mn:
            ok("K4", f"Output '{p}' delay max({mx}) >= min({mn})")
        else:
            fail("K4", f"Output '{p}' delay max({mx}) < min({mn}) — inverted!")

    # ── K5: feasibility ──
    for clk_name, clk_info in sdc.clocks.items():
        period = clk_info["period"]
        for inp, id_info in sdc.input_delays.items():
            for outp, od_info in sdc.output_delays.items():
                id_max = id_info.get("max", 0)
                od_max = od_info.get("max", 0)
                uncert = sdc.clock_uncertainty_setup or 0
                available = period - id_max - od_max - uncert
                if available > 0:
                    ok("K5", f"Path feasibility: {inp}→{outp} slack = {available:.1f} ns (period={period})")
                else:
                    fail("K5", f"Path infeasible: {inp}→{outp} slack = {available:.1f} ns "
                         f"(period={period}, in_max={id_max}, out_max={od_max}, uncert={uncert})")

    # ── K6: waveform sanity ──
    for clk_name, clk_info in sdc.clocks.items():
        period = clk_info["period"]
        rise = clk_info["rise"]
        fall = clk_info["fall"]
        if period <= 0:
            fail("K6", f"Clock '{clk_name}' period={period} <= 0")
        else:
            ok("K6", f"Clock '{clk_name}' period={period} > 0")
        if rise < fall <= period:
            ok("K6", f"Clock '{clk_name}' waveform rise({rise}) < fall({fall}) <= period({period})")
        else:
            fail("K6", f"Clock '{clk_name}' waveform invalid: rise={rise}, fall={fall}, period={period}")

    # ── K7: uncertainty < period/2 ──
    for clk_name, clk_info in sdc.clocks.items():
        period = clk_info["period"]
        for u_kind, u_val in [("setup", sdc.clock_uncertainty_setup),
                               ("hold", sdc.clock_uncertainty_hold)]:
            if u_val is not None:
                if u_val < period / 2:
                    ok("K7", f"Uncertainty ({u_kind}) {u_val} < period/2 ({period/2})")
                else:
                    fail("K7", f"Uncertainty ({u_kind}) {u_val} >= period/2 ({period/2}) — too large!")

    # ── K8: no duplicate create_clock ──
    seen_ports: dict[str, str] = {}
    for clk_name, clk_info in sdc.clocks.items():
        port = clk_info["port"]
        if port in seen_ports:
            fail("K8", f"Duplicate create_clock on port '{port}': "
                 f"'{seen_ports[port]}' and '{clk_name}'")
        else:
            seen_ports[port] = clk_name
            ok("K8", f"Clock '{clk_name}' on port '{port}' — unique")

    return results


# ─────────────────────── Report ─────────────────────────────────────────── #

def print_report(design: VerilogDesign, sdc: SdcInfo, results: list[CheckResult]) -> int:
    sep = "=" * 72

    print(sep)
    print("  cpu8 SDC Timing Constraint — Completeness & Consistency Report")
    print(sep)

    print(f"\n  Design:          {design.module_name}")
    print(f"  Clock ports:     {', '.join(design.clock_ports) or '(none)'}")
    print(f"  Reset ports:     {', '.join(design.reset_ports) or '(none)'}")
    print(f"  Data inputs:     {', '.join(design.input_ports) or '(none)'}")
    print(f"  Outputs:         {', '.join(design.output_ports) or '(none)'}")
    print(f"  Registers:       {len(design.reg_instances)} pyc_reg instances")

    print(f"\n  SDC clocks:      {len(sdc.clocks)}")
    for name, info in sdc.clocks.items():
        freq_mhz = 1000.0 / info['period'] if info['period'] > 0 else 0
        print(f"    {name}: period={info['period']} ns ({freq_mhz:.0f} MHz), "
              f"rise={info['rise']}, fall={info['fall']}, port={info['port']}")
    print(f"  Input delays:    {len(sdc.input_delays)} ports")
    print(f"  Output delays:   {len(sdc.output_delays)} ports")
    print(f"  False paths:     from={sdc.false_path_from}, to={sdc.false_path_to}")

    passed = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    print(f"\n{sep}")
    print(f"  Check Results: {len(passed)} PASS, {len(failed)} FAIL (total {len(results)})")
    print(sep)

    tags_seen: set[str] = set()
    for r in results:
        icon = "✓" if r.ok else "✗"
        status = "PASS" if r.ok else "FAIL"
        first = r.tag not in tags_seen
        tags_seen.add(r.tag)
        prefix = f"  [{r.tag}]" if first else f"       "
        print(f"  {icon} {status}  {prefix} {r.message}")

    print(f"\n{sep}")
    if failed:
        print(f"  RESULT:  {len(failed)} FAILURE(S) detected")
        for r in failed:
            print(f"    ✗ [{r.tag}] {r.message}")
    else:
        print("  RESULT:  ALL CHECKS PASSED — constraints are complete and consistent")
    print(sep)

    return 1 if failed else 0


# ─────────────────────── Main ───────────────────────────────────────────── #

def main() -> int:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <verilog_file> <sdc_file>", file=sys.stderr)
        return 2

    verilog_path = Path(sys.argv[1])
    sdc_path = Path(sys.argv[2])

    if not verilog_path.exists():
        print(f"error: Verilog file not found: {verilog_path}", file=sys.stderr)
        return 2
    if not sdc_path.exists():
        print(f"error: SDC file not found: {sdc_path}", file=sys.stderr)
        return 2

    design = parse_verilog(verilog_path)
    sdc = parse_sdc(sdc_path)
    results = run_checks(design, sdc)
    return print_report(design, sdc, results)


if __name__ == "__main__":
    sys.exit(main())
