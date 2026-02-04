// Combinational mux (prototype).
module pyc_mux #(
  parameter int WIDTH = 1
) (
  input  logic             sel,
  input  logic [WIDTH-1:0] a,
  input  logic [WIDTH-1:0] b,
  output logic [WIDTH-1:0] y
);
  assign y = sel ? a : b;
endmodule

