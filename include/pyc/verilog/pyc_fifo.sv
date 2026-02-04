// Ready/valid FIFO with synchronous reset (prototype).
module pyc_fifo #(
  parameter int WIDTH = 1,
  parameter int DEPTH = 2
) (
  input  logic             clk,
  input  logic             rst,

  // Input (producer -> fifo)
  input  logic             in_valid,
  output logic             in_ready,
  input  logic [WIDTH-1:0] in_data,

  // Output (fifo -> consumer)
  output logic             out_valid,
  input  logic             out_ready,
  output logic [WIDTH-1:0] out_data
);
  initial begin
    if (DEPTH <= 0) $fatal(1, "pyc_fifo DEPTH must be > 0");
  end

  localparam int PTR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH);

  logic [WIDTH-1:0] storage [0:DEPTH-1];
  logic [PTR_W-1:0] rd_ptr;
  logic [PTR_W-1:0] wr_ptr;
  logic [PTR_W:0]   count;

  assign in_ready  = (count < DEPTH) || (out_ready && out_valid);
  assign out_valid = (count != 0);
  assign out_data  = storage[rd_ptr];

  function automatic [PTR_W-1:0] bump_ptr(input [PTR_W-1:0] p);
    if (DEPTH <= 1) begin
      bump_ptr = '0;
    end else if (p == (DEPTH - 1)) begin
      bump_ptr = '0;
    end else begin
      bump_ptr = p + 1'b1;
    end
  endfunction

  always_ff @(posedge clk) begin
    if (rst) begin
      rd_ptr <= '0;
      wr_ptr <= '0;
      count <= '0;
    end else begin
      logic do_pop;
      logic do_push;
      logic [PTR_W-1:0] rd_next;
      logic [PTR_W-1:0] wr_next;
      logic [PTR_W:0] count_next;

      do_pop  = out_valid && out_ready;
      do_push = in_valid && in_ready;

      rd_next = rd_ptr;
      wr_next = wr_ptr;
      count_next = count;

      if (do_pop) begin
        rd_next = bump_ptr(rd_next);
        count_next = count_next - 1'b1;
      end
      if (do_push) begin
        storage[wr_ptr] <= in_data;
        wr_next = bump_ptr(wr_next);
        count_next = count_next + 1'b1;
      end

      rd_ptr <= rd_next;
      wr_ptr <= wr_next;
      count <= count_next;
    end
  end
endmodule
