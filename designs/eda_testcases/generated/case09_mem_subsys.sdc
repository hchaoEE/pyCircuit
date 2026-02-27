###############################################################################
# Case 9: mem_subsys — Memory subsystem with 4 banks + parity
# Single clock, 100 MHz target
###############################################################################

create_clock -name sys_clk -period 10.0 -waveform {0.0 5.0} [get_ports clk]

set_clock_uncertainty -setup 0.3 [get_clocks sys_clk]
set_clock_uncertainty -hold  0.1 [get_clocks sys_clk]
set_clock_transition  0.2        [get_clocks sys_clk]

set_input_delay -clock sys_clk -max 3.0 [get_ports addr[*]]
set_input_delay -clock sys_clk -min 0.5 [get_ports addr[*]]
set_input_delay -clock sys_clk -max 3.0 [get_ports wdata[*]]
set_input_delay -clock sys_clk -min 0.5 [get_ports wdata[*]]
set_input_delay -clock sys_clk -max 3.0 [get_ports wen]
set_input_delay -clock sys_clk -min 0.5 [get_ports wen]
set_input_delay -clock sys_clk -max 3.0 [get_ports ren]
set_input_delay -clock sys_clk -min 0.5 [get_ports ren]
set_input_delay -clock sys_clk -max 3.0 [get_ports rst]
set_input_delay -clock sys_clk -min 0.5 [get_ports rst]

set_output_delay -clock sys_clk -max 3.0 [get_ports rd_data[*]]
set_output_delay -clock sys_clk -min 0.5 [get_ports rd_data[*]]
set_output_delay -clock sys_clk -max 3.0 [get_ports parity_err]
set_output_delay -clock sys_clk -min 0.5 [get_ports parity_err]

# ready is a constant 1 — no real timing path
set_false_path -to [get_ports ready]

set_max_transition 0.5 [current_design]
set_max_fanout     20  [current_design]
