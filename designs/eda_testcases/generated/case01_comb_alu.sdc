###############################################################################
# Case 1: comb_alu — Pure combinational ALU (no clock)
# No sequential elements; constrain max combinational delay only.
###############################################################################

# Virtual clock for constraining combinational-only paths
create_clock -name vclk -period 10.0 -waveform {0.0 5.0}

set_input_delay  -clock vclk -max 0.0 [get_ports a[*]]
set_input_delay  -clock vclk -min 0.0 [get_ports a[*]]
set_input_delay  -clock vclk -max 0.0 [get_ports b[*]]
set_input_delay  -clock vclk -min 0.0 [get_ports b[*]]
set_input_delay  -clock vclk -max 0.0 [get_ports op[*]]
set_input_delay  -clock vclk -min 0.0 [get_ports op[*]]

set_output_delay -clock vclk -max 0.0 [get_ports result[*]]
set_output_delay -clock vclk -min 0.0 [get_ports result[*]]
set_output_delay -clock vclk -max 0.0 [get_ports zero]
set_output_delay -clock vclk -min 0.0 [get_ports zero]
set_output_delay -clock vclk -max 0.0 [get_ports carry]
set_output_delay -clock vclk -min 0.0 [get_ports carry]

set_max_delay 10.0 -from [all_inputs] -to [all_outputs]

set_max_transition 0.5 [current_design]
set_max_fanout     20  [current_design]
