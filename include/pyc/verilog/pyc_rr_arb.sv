// Round-robin ready/valid arbiter (prototype).
//
// - Chooses at most one input each cycle.
// - Consumes the chosen input when `out_valid && out_ready`.
// - Updates the RR pointer on a successful consume.
module pyc_rr_arb #(
  parameter int WIDTH = 1,
  parameter int N = 2
) (
  input  logic                      clk,
  input  logic                      rst,

  input  logic [N-1:0]              in_valid,
  output logic [N-1:0]              in_ready,
  input  logic [N-1:0][WIDTH-1:0]   in_data,

  output logic                      out_valid,
  input  logic                      out_ready,
  output logic [WIDTH-1:0]          out_data,
  output logic [$clog2((N <= 1) ? 2 : N)-1:0] out_sel
);
  localparam int SEL_W = $clog2((N <= 1) ? 2 : N);

  logic [SEL_W-1:0] rr_ptr;
  logic [N-1:0] grant;
  logic [SEL_W-1:0] sel;

  always_comb begin
    out_valid = 1'b0;
    out_data = '0;
    sel = '0;
    grant = '0;

    // Search for the first valid entry starting from rr_ptr.
    for (int off = 0; off < N; off++) begin
      int idx;
      idx = rr_ptr + off;
      if (idx >= N) idx -= N;
      if (!out_valid && in_valid[idx]) begin
        out_valid = 1'b1;
        out_data = in_data[idx];
        sel = idx[SEL_W-1:0];
        grant[idx] = 1'b1;
      end
    end

    out_sel = sel;

    in_ready = '0;
    if (out_valid) begin
      for (int i = 0; i < N; i++) begin
        in_ready[i] = grant[i] & out_ready;
      end
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      rr_ptr <= '0;
    end else if (out_valid && out_ready) begin
      if (N <= 1) rr_ptr <= '0;
      else rr_ptr <= (sel == (N - 1)) ? '0 : (sel + 1'b1);
    end
  end
endmodule

