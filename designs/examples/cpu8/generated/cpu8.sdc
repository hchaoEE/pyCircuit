###############################################################################
# cpu8 — SDC Timing Constraints
# 5-stage pipelined 8-bit CPU (IF → ID → EX → MEM → WB)
#
# Target:  100 MHz (10 ns period)
# Tool:    Synopsys / Vivado / Generic SDC-compatible
###############################################################################

# =========================================================================== #
#  1. Clock Definition                                                        #
# =========================================================================== #

create_clock -name sys_clk -period 10.0 -waveform {0.0 5.0} [get_ports clk]

set CLK_PERIOD   10.0
set CLK_HALF      5.0
set CLK_UNCERT    0.3
set IN_DELAY      2.0
set OUT_DELAY     2.0
set IN_DELAY_MAX  3.0
set OUT_DELAY_MAX 3.0
set IN_DELAY_MIN  0.5
set OUT_DELAY_MIN 0.5

# =========================================================================== #
#  2. Clock Uncertainty (jitter + skew budget)                                #
# =========================================================================== #

set_clock_uncertainty -setup $CLK_UNCERT [get_clocks sys_clk]
set_clock_uncertainty -hold  0.1         [get_clocks sys_clk]

# =========================================================================== #
#  3. Clock Transition (slew)                                                 #
# =========================================================================== #

set_clock_transition 0.2 [get_clocks sys_clk]

# =========================================================================== #
#  4. Input Delay Constraints                                                 #
# =========================================================================== #
# All input ports are sampled on the rising edge of sys_clk.
# inst_in[7:0] — instruction from external instruction memory (IF stage)
# rst         — asynchronous-style reset treated as synchronous in pyc_reg

set_input_delay -clock sys_clk -max $IN_DELAY_MAX [get_ports inst_in[*]]
set_input_delay -clock sys_clk -min $IN_DELAY_MIN [get_ports inst_in[*]]

set_input_delay -clock sys_clk -max $IN_DELAY_MAX [get_ports rst]
set_input_delay -clock sys_clk -min $IN_DELAY_MIN [get_ports rst]

# =========================================================================== #
#  5. Output Delay Constraints                                                #
# =========================================================================== #
# All outputs are driven from register Q or 1-level combinational logic.

# Program counter (combinational from pc_q register)
set_output_delay -clock sys_clk -max $OUT_DELAY_MAX [get_ports pc[*]]
set_output_delay -clock sys_clk -min $OUT_DELAY_MIN [get_ports pc[*]]

# Register file outputs (r0-r3)
# r0 is covered by set_false_path; no output_delay needed.

set_output_delay -clock sys_clk -max $OUT_DELAY_MAX [get_ports r1[*]]
set_output_delay -clock sys_clk -min $OUT_DELAY_MIN [get_ports r1[*]]

set_output_delay -clock sys_clk -max $OUT_DELAY_MAX [get_ports r2[*]]
set_output_delay -clock sys_clk -min $OUT_DELAY_MIN [get_ports r2[*]]

set_output_delay -clock sys_clk -max $OUT_DELAY_MAX [get_ports r3[*]]
set_output_delay -clock sys_clk -min $OUT_DELAY_MIN [get_ports r3[*]]

# Write-back stage diagnostic outputs
set_output_delay -clock sys_clk -max $OUT_DELAY_MAX [get_ports wb_valid]
set_output_delay -clock sys_clk -min $OUT_DELAY_MIN [get_ports wb_valid]

set_output_delay -clock sys_clk -max $OUT_DELAY_MAX [get_ports wb_rd[*]]
set_output_delay -clock sys_clk -min $OUT_DELAY_MIN [get_ports wb_rd[*]]

set_output_delay -clock sys_clk -max $OUT_DELAY_MAX [get_ports wb_result[*]]
set_output_delay -clock sys_clk -min $OUT_DELAY_MIN [get_ports wb_result[*]]

# =========================================================================== #
#  6. False Paths                                                             #
# =========================================================================== #
# r0 is hardwired to 0 (rf0__next = constant 0); the register exists but
# its data input is a constant.  No real timing path to optimize.
set_false_path -to [get_ports r0[*]]

# Note: rst is a synchronous reset (pyc_reg samples at posedge clk), so it
# MUST meet setup/hold timing.  Do NOT set_false_path on rst.

# =========================================================================== #
#  7. Multicycle Paths                                                        #
# =========================================================================== #
# The MEM stage is a pure pass-through (no memory access or extra logic).
# EX/MEM → MEM/WB is a register-to-register copy that could be relaxed to
# 2-cycle if needed, but we keep it at 1 for safety at 100 MHz.
# (Uncomment the following if timing is tight on EX/MEM → MEM/WB path)
#
# set_multicycle_path 2 -setup \
#   -from [get_pins pyc_reg_44_inst/q] -to [get_pins pyc_reg_50_inst/d]
# set_multicycle_path 1 -hold \
#   -from [get_pins pyc_reg_44_inst/q] -to [get_pins pyc_reg_50_inst/d]

# =========================================================================== #
#  8. Max Transition / Max Fanout / Max Capacitance                           #
# =========================================================================== #

set_max_transition 0.5 [current_design]
set_max_fanout     20  [current_design]

# =========================================================================== #
#  9. Operating Conditions (optional, tool-dependent)                         #
# =========================================================================== #
# set_operating_conditions -max slow -min fast

# =========================================================================== #
# 10. Design Rule Constraints                                                 #
# =========================================================================== #
# Prevent any combinational loops (should already be guaranteed by pycc).
# set_max_delay 0 -combinational_from_to  (tool-specific)

# =========================================================================== #
# Summary
# =========================================================================== #
# Clocks defined:      1 (sys_clk, 100 MHz, 10 ns)
# Input ports:         2 (inst_in[7:0], rst)
# Output ports:        8 (pc[7:0], r0-r3[7:0], wb_valid, wb_rd[1:0], wb_result[7:0])
# Sequential elements: 19 pyc_reg instances, all clocked by sys_clk
# False paths:         *→r0 (constant output, hardwired to 0)
# Clock domains:       1 (single-clock, no CDC)
###############################################################################
