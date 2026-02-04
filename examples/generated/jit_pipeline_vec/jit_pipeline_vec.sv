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

module JitPipelineVec (
  input logic sys_clk,
  input logic sys_rst,
  input logic [15:0] a,
  input logic [15:0] b,
  input logic sel,
  output logic tag,
  output logic [15:0] data,
  output logic [7:0] lo8
);

logic [24:0] v1;
logic v2;
logic [15:0] v3;
logic [15:0] v4;
logic [15:0] v5;
logic v6;
logic [7:0] v7;
logic [24:0] v8;
logic [24:0] v9;
logic [24:0] v10;
logic [24:0] v11;
logic [24:0] v12;
logic [24:0] v13;
logic [24:0] v14;
logic [24:0] v15;
logic [24:0] v16;
logic v17;
logic [24:0] v18;
logic [24:0] v19;
logic [24:0] v20;
logic [24:0] v21;
logic [7:0] v22;
logic [15:0] v23;
logic v24;
logic [7:0] v25;
logic [15:0] v26;
logic v27;

assign v1 = 25'd0;
assign v2 = 1'd1;
assign v3 = (a + b);
assign v4 = (a ^ b);
assign v5 = (sel ? v3 : v4);
assign v6 = (a == b);
assign v7 = v5[7:0];
assign v8 = {{17{1'b0}}, v7};
assign v9 = (v1 | v8);
assign v10 = {{9{1'b0}}, v5};
assign v11 = (v10 << 8);
assign v12 = (v9 | v11);
assign v13 = {{24{1'b0}}, v6};
assign v14 = (v13 << 24);
assign v15 = (v12 | v14);
assign v16 = v1;
assign v17 = v2;
assign v18 = v15;
pyc_reg #(.WIDTH(25)) v19_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v17),
  .d(v18),
  .init(v16),
  .q(v19)
);
pyc_reg #(.WIDTH(25)) v20_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v17),
  .d(v19),
  .init(v16),
  .q(v20)
);
pyc_reg #(.WIDTH(25)) v21_inst (
  .clk(sys_clk),
  .rst(sys_rst),
  .en(v17),
  .d(v20),
  .init(v16),
  .q(v21)
);
assign v22 = v21[7:0];
assign v23 = v21[23:8];
assign v24 = v21[24];
assign v25 = v22;
assign v26 = v23;
assign v27 = v24;
assign tag = v27;
assign data = v26;
assign lo8 = v25;

endmodule

