`include "pyc_handshake_pkg.sv"
`include "pyc_stream_if.sv"
`include "pyc_mem_if.sv"
`include "pyc_add.sv"
`include "pyc_mux.sv"
`include "pyc_and.sv"
`include "pyc_or.sv"
`include "pyc_xor.sv"
`include "pyc_not.sv"
`include "pyc_reg.sv"
`include "pyc_fifo.sv"

`include "pyc_byte_mem.sv"

`include "pyc_queue.sv"
`include "pyc_picker_onehot.sv"
`include "pyc_rr_arb.sv"
`include "pyc_sram.sv"

module FifoLoopback (
  input logic clk,
  input logic rst,
  input logic in_valid,
  input logic [7:0] in_data,
  input logic out_ready,
  output logic in_ready,
  output logic out_valid,
  output logic [7:0] out_data
);

logic v1;
logic v2;
logic [7:0] v3;

pyc_fifo #(.WIDTH(8), .DEPTH(2)) v1_inst (
  .clk(clk),
  .rst(rst),
  .in_valid(in_valid),
  .in_ready(v1),
  .in_data(in_data),
  .out_valid(v2),
  .out_ready(out_ready),
  .out_data(v3)
);
assign in_ready = v1;
assign out_valid = v2;
assign out_data = v3;

endmodule

