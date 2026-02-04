// Ready/valid queue (prototype).
//
// - DEPTH=1: fall-through "skid buffer" (0-cycle when ready, 1-entry storage when stalled)
// - DEPTH>1: wraps `pyc_fifo`
module pyc_queue #(
  parameter int WIDTH = 1,
  parameter int DEPTH = 1
) (
  input  logic             clk,
  input  logic             rst,

  input  logic             in_valid,
  output logic             in_ready,
  input  logic [WIDTH-1:0] in_data,

  output logic             out_valid,
  input  logic             out_ready,
  output logic [WIDTH-1:0] out_data
);
  generate
    if (DEPTH == 1) begin : gen_skid
      logic full;
      logic [WIDTH-1:0] storage;

      // Fall-through behavior when not full.
      assign out_valid = full ? 1'b1 : in_valid;
      assign out_data  = full ? storage : in_data;
      assign in_ready  = ~full || out_ready;

      always_ff @(posedge clk) begin
        if (rst) begin
          full <= 1'b0;
        end else if (full) begin
          // If downstream accepts, we can either become empty or immediately refill.
          if (out_ready) begin
            if (in_valid) begin
              storage <= in_data;
              full <= 1'b1;
            end else begin
              full <= 1'b0;
            end
          end
        end else begin
          // Empty: if downstream is stalled and we see valid, capture.
          if (in_valid && !out_ready) begin
            storage <= in_data;
            full <= 1'b1;
          end
        end
      end
    end else begin : gen_fifo
      pyc_fifo #(.WIDTH(WIDTH), .DEPTH(DEPTH)) u_fifo (
        .clk(clk),
        .rst(rst),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_data(in_data),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_data(out_data)
      );
    end
  endgenerate
endmodule

