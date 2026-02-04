// Combinational bitwise NOT (prototype).
module pyc_not #(
  parameter int WIDTH = 1
) (
  input  logic [WIDTH-1:0] a,
  output logic [WIDTH-1:0] y
);
  assign y = ~a;
endmodule

