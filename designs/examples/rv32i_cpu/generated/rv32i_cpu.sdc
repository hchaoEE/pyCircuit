###############################################################################
# rv32i_cpu — SDC Timing Constraints
# 32-bit 5-stage pipelined RV32I CPU (IF → ID → EX → MEM → WB)
#
# Target:  100 MHz (10 ns period)
###############################################################################

# =========================================================================== #
#  1. Clock Definition                                                        #
# =========================================================================== #

create_clock -name sys_clk -period 10.0 -waveform {0.0 5.0} [get_ports clk]

set CLK_PERIOD    10.0
set CLK_UNCERT    0.3
set IN_DELAY_MAX  3.0
set IN_DELAY_MIN  0.5
set OUT_DELAY_MAX 3.0
set OUT_DELAY_MIN 0.5

# =========================================================================== #
#  2. Clock Uncertainty                                                       #
# =========================================================================== #

set_clock_uncertainty -setup $CLK_UNCERT [get_clocks sys_clk]
set_clock_uncertainty -hold  0.1         [get_clocks sys_clk]

# =========================================================================== #
#  3. Clock Transition                                                        #
# =========================================================================== #

set_clock_transition 0.2 [get_clocks sys_clk]

# =========================================================================== #
#  4. Input Delay Constraints                                                 #
# =========================================================================== #

# Instruction data from external I-memory (IF stage)
set_input_delay -clock sys_clk -max $IN_DELAY_MAX [get_ports inst_data[*]]
set_input_delay -clock sys_clk -min $IN_DELAY_MIN [get_ports inst_data[*]]

# Data memory read data (MEM stage)
set_input_delay -clock sys_clk -max $IN_DELAY_MAX [get_ports dmem_rdata[*]]
set_input_delay -clock sys_clk -min $IN_DELAY_MIN [get_ports dmem_rdata[*]]

# Synchronous reset
set_input_delay -clock sys_clk -max $IN_DELAY_MAX [get_ports rst]
set_input_delay -clock sys_clk -min $IN_DELAY_MIN [get_ports rst]

# =========================================================================== #
#  5. Output Delay Constraints                                                #
# =========================================================================== #

# Instruction address (PC → I-memory)
set_output_delay -clock sys_clk -max $OUT_DELAY_MAX [get_ports inst_addr[*]]
set_output_delay -clock sys_clk -min $OUT_DELAY_MIN [get_ports inst_addr[*]]

# Data memory interface
set_output_delay -clock sys_clk -max $OUT_DELAY_MAX [get_ports dmem_addr[*]]
set_output_delay -clock sys_clk -min $OUT_DELAY_MIN [get_ports dmem_addr[*]]

set_output_delay -clock sys_clk -max $OUT_DELAY_MAX [get_ports dmem_wdata[*]]
set_output_delay -clock sys_clk -min $OUT_DELAY_MIN [get_ports dmem_wdata[*]]

set_output_delay -clock sys_clk -max $OUT_DELAY_MAX [get_ports dmem_wen]
set_output_delay -clock sys_clk -min $OUT_DELAY_MIN [get_ports dmem_wen]

set_output_delay -clock sys_clk -max $OUT_DELAY_MAX [get_ports dmem_ren]
set_output_delay -clock sys_clk -min $OUT_DELAY_MIN [get_ports dmem_ren]

# Write-back diagnostic outputs
set_output_delay -clock sys_clk -max $OUT_DELAY_MAX [get_ports wb_valid]
set_output_delay -clock sys_clk -min $OUT_DELAY_MIN [get_ports wb_valid]

set_output_delay -clock sys_clk -max $OUT_DELAY_MAX [get_ports wb_rd[*]]
set_output_delay -clock sys_clk -min $OUT_DELAY_MIN [get_ports wb_rd[*]]

set_output_delay -clock sys_clk -max $OUT_DELAY_MAX [get_ports wb_data[*]]
set_output_delay -clock sys_clk -min $OUT_DELAY_MIN [get_ports wb_data[*]]

# Register observation outputs
set_output_delay -clock sys_clk -max $OUT_DELAY_MAX [get_ports x1_ra[*]]
set_output_delay -clock sys_clk -min $OUT_DELAY_MIN [get_ports x1_ra[*]]

set_output_delay -clock sys_clk -max $OUT_DELAY_MAX [get_ports x2_sp[*]]
set_output_delay -clock sys_clk -min $OUT_DELAY_MIN [get_ports x2_sp[*]]

set_output_delay -clock sys_clk -max $OUT_DELAY_MAX [get_ports x10_a0[*]]
set_output_delay -clock sys_clk -min $OUT_DELAY_MIN [get_ports x10_a0[*]]

set_output_delay -clock sys_clk -max $OUT_DELAY_MAX [get_ports x11_a1[*]]
set_output_delay -clock sys_clk -min $OUT_DELAY_MIN [get_ports x11_a1[*]]

# =========================================================================== #
#  6. False Paths                                                             #
# =========================================================================== #

# None — all paths are real timing paths in this single-clock design.
# rst is synchronous (pyc_reg uses posedge clk), so it must meet timing.

# =========================================================================== #
#  7. Design Rule Constraints                                                 #
# =========================================================================== #

set_max_transition 0.5 [current_design]
set_max_fanout     20  [current_design]

###############################################################################
# Summary
# =========================================================================== #
# Clocks:            1 (sys_clk, 100 MHz, 10 ns)
# Input ports:       3 (inst_data[31:0], dmem_rdata[31:0], rst)
# Output ports:      12 (inst_addr, dmem_addr, dmem_wdata, dmem_wen, dmem_ren,
#                        wb_valid, wb_rd, wb_data, x1_ra, x2_sp, x10_a0, x11_a1)
# Sequential:        60 pyc_reg instances (1345 bits), single clock domain
# False paths:       none
###############################################################################
