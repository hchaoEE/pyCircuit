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

module MultiClockRegs (
  input logic clk_a,
  input logic rst_a,
  input logic clk_b,
  input logic rst_b,
  output logic [7:0] a_count,
  output logic [7:0] b_count
);

logic v1;
logic [7:0] v2;
logic [7:0] v3;
logic v4;
logic [7:0] v5;
logic [7:0] v6;
logic [7:0] v7;
logic [7:0] v8;
logic [7:0] v9;
logic [7:0] v10;
logic [7:0] v11;
logic [7:0] v12;

assign v1 = 1'd1;
assign v2 = 8'd0;
assign v3 = 8'd1;
assign v4 = v1;
assign v5 = v2;
assign v6 = v3;
pyc_reg #(.WIDTH(8)) v7_inst (
  .clk(clk_a),
  .rst(rst_a),
  .en(v4),
  .d(v5),
  .init(v5),
  .q(v7)
);
pyc_add #(.WIDTH(8)) v8_inst (
  .a(v7),
  .b(v6),
  .y(v8)
);
pyc_reg #(.WIDTH(8)) v9_inst (
  .clk(clk_a),
  .rst(rst_a),
  .en(v4),
  .d(v8),
  .init(v5),
  .q(v9)
);
pyc_reg #(.WIDTH(8)) v10_inst (
  .clk(clk_b),
  .rst(rst_b),
  .en(v4),
  .d(v5),
  .init(v5),
  .q(v10)
);
pyc_add #(.WIDTH(8)) v11_inst (
  .a(v10),
  .b(v6),
  .y(v11)
);
pyc_reg #(.WIDTH(8)) v12_inst (
  .clk(clk_b),
  .rst(rst_b),
  .en(v4),
  .d(v11),
  .init(v5),
  .q(v12)
);
assign a_count = v9;
assign b_count = v12;

endmodule

