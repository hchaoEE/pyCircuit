###############################################################################
# Case 3: fsm_traffic — Traffic light FSM controller
# Single clock, 50 MHz target (slower for IoT/embedded)
###############################################################################

create_clock -name sys_clk -period 20.0 -waveform {0.0 10.0} [get_ports clk]

set_clock_uncertainty -setup 0.5 [get_clocks sys_clk]
set_clock_uncertainty -hold  0.1 [get_clocks sys_clk]
set_clock_transition  0.3        [get_clocks sys_clk]

set_input_delay -clock sys_clk -max 5.0 [get_ports emergency]
set_input_delay -clock sys_clk -min 1.0 [get_ports emergency]
set_input_delay -clock sys_clk -max 5.0 [get_ports rst]
set_input_delay -clock sys_clk -min 1.0 [get_ports rst]

set_output_delay -clock sys_clk -max 5.0 [get_ports ns_red]
set_output_delay -clock sys_clk -min 1.0 [get_ports ns_red]
set_output_delay -clock sys_clk -max 5.0 [get_ports ns_yellow]
set_output_delay -clock sys_clk -min 1.0 [get_ports ns_yellow]
set_output_delay -clock sys_clk -max 5.0 [get_ports ns_green]
set_output_delay -clock sys_clk -min 1.0 [get_ports ns_green]
set_output_delay -clock sys_clk -max 5.0 [get_ports ew_red]
set_output_delay -clock sys_clk -min 1.0 [get_ports ew_red]
set_output_delay -clock sys_clk -max 5.0 [get_ports ew_yellow]
set_output_delay -clock sys_clk -min 1.0 [get_ports ew_yellow]
set_output_delay -clock sys_clk -max 5.0 [get_ports ew_green]
set_output_delay -clock sys_clk -min 1.0 [get_ports ew_green]
set_output_delay -clock sys_clk -max 5.0 [get_ports timer_val[*]]
set_output_delay -clock sys_clk -min 1.0 [get_ports timer_val[*]]

set_max_transition 0.8 [current_design]
set_max_fanout     16  [current_design]
