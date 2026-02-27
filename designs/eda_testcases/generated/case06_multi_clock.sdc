###############################################################################
# Case 6: multi_clock — Dual clock-domain design with CDC
# clk_fast = 200 MHz, clk_slow = 50 MHz (asynchronous)
###############################################################################

# --- Clock definitions ---
create_clock -name clk_fast -period 5.0  -waveform {0.0 2.5} [get_ports clk_fast]
create_clock -name clk_slow -period 20.0 -waveform {0.0 10.0} [get_ports clk_slow]

# Clocks are asynchronous (no phase relationship)
set_clock_groups -asynchronous \
  -group [get_clocks clk_fast] \
  -group [get_clocks clk_slow]

# --- Clock uncertainty ---
set_clock_uncertainty -setup 0.2 [get_clocks clk_fast]
set_clock_uncertainty -hold  0.05 [get_clocks clk_fast]
set_clock_uncertainty -setup 0.5 [get_clocks clk_slow]
set_clock_uncertainty -hold  0.1 [get_clocks clk_slow]

set_clock_transition 0.15 [get_clocks clk_fast]
set_clock_transition 0.3  [get_clocks clk_slow]

# --- Input delays (fast domain) ---
set_input_delay -clock clk_fast -max 1.5 [get_ports data_in[*]]
set_input_delay -clock clk_fast -min 0.3 [get_ports data_in[*]]
set_input_delay -clock clk_fast -max 1.5 [get_ports capture]
set_input_delay -clock clk_fast -min 0.3 [get_ports capture]
set_input_delay -clock clk_fast -max 1.5 [get_ports rst_fast]
set_input_delay -clock clk_fast -min 0.3 [get_ports rst_fast]
set_input_delay -clock clk_slow -max 5.0 [get_ports rst_slow]
set_input_delay -clock clk_slow -min 1.0 [get_ports rst_slow]

# --- Output delays ---
set_output_delay -clock clk_fast -max 1.5 [get_ports fast_count[*]]
set_output_delay -clock clk_fast -min 0.3 [get_ports fast_count[*]]
set_output_delay -clock clk_fast -max 1.5 [get_ports captured[*]]
set_output_delay -clock clk_fast -min 0.3 [get_ports captured[*]]
set_output_delay -clock clk_slow -max 5.0 [get_ports sync_valid]
set_output_delay -clock clk_slow -min 1.0 [get_ports sync_valid]
set_output_delay -clock clk_slow -max 5.0 [get_ports slow_accumulator[*]]
set_output_delay -clock clk_slow -min 1.0 [get_ports slow_accumulator[*]]

# --- CDC paths: max_delay for synchronizer chain ---
set_max_delay 5.0 -from [get_clocks clk_fast] -to [get_clocks clk_slow]
set_max_delay 5.0 -from [get_clocks clk_slow] -to [get_clocks clk_fast]

set_max_transition 0.3 [current_design]
set_max_fanout     16  [current_design]
