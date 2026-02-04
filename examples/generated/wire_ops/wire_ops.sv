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

module WireOps (
  input logic sys_clk,
  input logic sys_rst,
  input logic [7:0] a,
  input logic [7:0] b,
  input logic sel,
  output logic [7:0] y
);

logic [7:0] v1;
logic v2;
logic [7:0] v3;
logic [7:0] v4;
logic [7:0] v5;
logic [7:0] v6;
logic v7;
logic [7:0] v8;
logic [7:0] v9;

assign v1 = 8'd0;
assign v2 = 1'd1;
assign v3 = (a & b);
assign v4 = (a ^ b);
assign v5 = (sel ? v3 : v4);
assign v6 = v1;
assign v7 = v2;
assign v8 = v5;
pyc_reg #(.WIDTH(8)) v9_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v7),
  .d(v8),
  .init(v6),
  .q(v9)
);
assign y = v9;

endmodule

