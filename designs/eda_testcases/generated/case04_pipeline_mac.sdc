###############################################################################
# Case 4: pipeline_mac — 4-stage pipelined multiply-accumulate
# Single clock, 200 MHz target (high-speed DSP)
###############################################################################

create_clock -name sys_clk -period 5.0 -waveform {0.0 2.5} [get_ports clk]

set_clock_uncertainty -setup 0.2 [get_clocks sys_clk]
set_clock_uncertainty -hold  0.05 [get_clocks sys_clk]
set_clock_transition  0.15       [get_clocks sys_clk]

set_input_delay -clock sys_clk -max 1.5 [get_ports a_in[*]]
set_input_delay -clock sys_clk -min 0.3 [get_ports a_in[*]]
set_input_delay -clock sys_clk -max 1.5 [get_ports b_in[*]]
set_input_delay -clock sys_clk -min 0.3 [get_ports b_in[*]]
set_input_delay -clock sys_clk -max 1.5 [get_ports valid_in]
set_input_delay -clock sys_clk -min 0.3 [get_ports valid_in]
set_input_delay -clock sys_clk -max 1.5 [get_ports acc_clear]
set_input_delay -clock sys_clk -min 0.3 [get_ports acc_clear]
set_input_delay -clock sys_clk -max 1.5 [get_ports saturate_mode]
set_input_delay -clock sys_clk -min 0.3 [get_ports saturate_mode]
set_input_delay -clock sys_clk -max 1.5 [get_ports rst]
set_input_delay -clock sys_clk -min 0.3 [get_ports rst]

set_output_delay -clock sys_clk -max 1.5 [get_ports result[*]]
set_output_delay -clock sys_clk -min 0.3 [get_ports result[*]]
set_output_delay -clock sys_clk -max 1.5 [get_ports valid_out]
set_output_delay -clock sys_clk -min 0.3 [get_ports valid_out]
set_output_delay -clock sys_clk -max 1.5 [get_ports accumulator[*]]
set_output_delay -clock sys_clk -min 0.3 [get_ports accumulator[*]]

set_max_transition 0.3 [current_design]
set_max_fanout     16  [current_design]
