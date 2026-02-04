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

module Counter (
  input logic clk,
  input logic rst,
  input logic en,
  output logic [7:0] count
);

logic [7:0] v1;
logic [7:0] v2;
logic [7:0] v3;
logic [7:0] v4;
logic [7:0] v5;
logic [7:0] v6;
logic [7:0] v7;

assign v1 = 8'd1;
assign v2 = 8'd0;
assign v3 = v1;
assign v4 = v2;
pyc_reg #(.WIDTH(8)) v5_inst (
  .clk(clk),
  .rst(rst),
  .en(en),
  .d(v4),
  .init(v4),
  .q(v5)
);
pyc_add #(.WIDTH(8)) v6_inst (
  .a(v5),
  .b(v3),
  .y(v6)
);
pyc_reg #(.WIDTH(8)) v7_inst (
  .clk(clk),
  .rst(rst),
  .en(en),
  .d(v6),
  .init(v4),
  .q(v7)
);
assign count = v7;

endmodule

