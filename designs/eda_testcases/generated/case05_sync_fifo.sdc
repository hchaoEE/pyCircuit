###############################################################################
# Case 5: sync_fifo — Synchronous FIFO (depth 16, width 32)
# Single clock, 150 MHz target
###############################################################################

create_clock -name sys_clk -period 6.67 -waveform {0.0 3.335} [get_ports clk]

set_clock_uncertainty -setup 0.25 [get_clocks sys_clk]
set_clock_uncertainty -hold  0.08 [get_clocks sys_clk]
set_clock_transition  0.2         [get_clocks sys_clk]

set_input_delay -clock sys_clk -max 2.0 [get_ports wr_en]
set_input_delay -clock sys_clk -min 0.4 [get_ports wr_en]
set_input_delay -clock sys_clk -max 2.0 [get_ports wr_data[*]]
set_input_delay -clock sys_clk -min 0.4 [get_ports wr_data[*]]
set_input_delay -clock sys_clk -max 2.0 [get_ports rd_en]
set_input_delay -clock sys_clk -min 0.4 [get_ports rd_en]
set_input_delay -clock sys_clk -max 2.0 [get_ports rst]
set_input_delay -clock sys_clk -min 0.4 [get_ports rst]

set_output_delay -clock sys_clk -max 2.0 [get_ports rd_data[*]]
set_output_delay -clock sys_clk -min 0.4 [get_ports rd_data[*]]
set_output_delay -clock sys_clk -max 2.0 [get_ports full]
set_output_delay -clock sys_clk -min 0.4 [get_ports full]
set_output_delay -clock sys_clk -max 2.0 [get_ports empty]
set_output_delay -clock sys_clk -min 0.4 [get_ports empty]
set_output_delay -clock sys_clk -max 2.0 [get_ports half_full]
set_output_delay -clock sys_clk -min 0.4 [get_ports half_full]
set_output_delay -clock sys_clk -max 2.0 [get_ports count[*]]
set_output_delay -clock sys_clk -min 0.4 [get_ports count[*]]
set_output_delay -clock sys_clk -max 2.0 [get_ports overflow]
set_output_delay -clock sys_clk -min 0.4 [get_ports overflow]
set_output_delay -clock sys_clk -max 2.0 [get_ports underflow]
set_output_delay -clock sys_clk -min 0.4 [get_ports underflow]

set_max_transition 0.4 [current_design]
set_max_fanout     20  [current_design]
