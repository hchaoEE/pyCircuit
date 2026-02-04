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

module JitControlFlow (
  input logic [7:0] a,
  input logic [7:0] b,
  output logic [7:0] out
);

logic [7:0] v1;
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

assign v1 = 8'd2;
assign v2 = 8'd1;
assign v3 = (a + b);
assign v4 = (a == b);
assign v5 = (v3 + v2);
assign v6 = (v3 + v1);
assign v7 = (v4 ? v5 : v6);
assign v8 = (v7 + v2);
assign v9 = (v8 + v2);
assign v10 = (v9 + v2);
assign v11 = (v10 + v2);
assign v12 = v11;
assign out = v12;

endmodule

