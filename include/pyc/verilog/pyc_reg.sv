// Simple synchronous reset register (prototype).
module pyc_reg #(
  parameter int WIDTH = 1
) (
  input  logic             clk,
  input  logic             rst,
  input  logic             en,
  input  logic [WIDTH-1:0] d,
  input  logic [WIDTH-1:0] init,
  output logic [WIDTH-1:0] q
);
  always_ff @(posedge clk) begin
    if (rst) q <= init;
    else if (en) q <= d;
  end
endmodule

