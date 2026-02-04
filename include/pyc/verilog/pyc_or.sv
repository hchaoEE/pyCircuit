// Combinational bitwise OR (prototype).
module pyc_or #(
  parameter int WIDTH = 1
) (
  input  logic [WIDTH-1:0] a,
  input  logic [WIDTH-1:0] b,
  output logic [WIDTH-1:0] y
);
  assign y = a | b;
endmodule

