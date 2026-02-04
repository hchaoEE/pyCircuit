// One-hot picker (prototype).
//
// Selects one of N inputs based on a one-hot `sel` vector.
module pyc_picker_onehot #(
  parameter int WIDTH = 1,
  parameter int N = 2
) (
  input  logic [N-1:0]              sel,
  input  logic [N-1:0][WIDTH-1:0]    in_data,
  output logic [WIDTH-1:0]           y
);
  always_comb begin
    y = '0;
    for (int i = 0; i < N; i++) begin
      if (sel[i]) y |= in_data[i];
    end
  end
endmodule

