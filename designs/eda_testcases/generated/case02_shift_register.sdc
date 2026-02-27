###############################################################################
# Case 2: shift_register — 32-bit configurable shift register
# Single clock, 100 MHz target
###############################################################################

create_clock -name sys_clk -period 10.0 -waveform {0.0 5.0} [get_ports clk]

set_clock_uncertainty -setup 0.3 [get_clocks sys_clk]
set_clock_uncertainty -hold  0.1 [get_clocks sys_clk]
set_clock_transition  0.2        [get_clocks sys_clk]

set_input_delay -clock sys_clk -max 3.0 [get_ports din]
set_input_delay -clock sys_clk -min 0.5 [get_ports din]
set_input_delay -clock sys_clk -max 3.0 [get_ports pdin[*]]
set_input_delay -clock sys_clk -min 0.5 [get_ports pdin[*]]
set_input_delay -clock sys_clk -max 3.0 [get_ports mode[*]]
set_input_delay -clock sys_clk -min 0.5 [get_ports mode[*]]
set_input_delay -clock sys_clk -max 3.0 [get_ports shift_en]
set_input_delay -clock sys_clk -min 0.5 [get_ports shift_en]
set_input_delay -clock sys_clk -max 3.0 [get_ports rst]
set_input_delay -clock sys_clk -min 0.5 [get_ports rst]

set_output_delay -clock sys_clk -max 3.0 [get_ports dout]
set_output_delay -clock sys_clk -min 0.5 [get_ports dout]
set_output_delay -clock sys_clk -max 3.0 [get_ports pout[*]]
set_output_delay -clock sys_clk -min 0.5 [get_ports pout[*]]

set_max_transition 0.5 [current_design]
set_max_fanout     20  [current_design]
