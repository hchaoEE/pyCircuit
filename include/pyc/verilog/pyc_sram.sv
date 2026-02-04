// Single-outstanding request/response SRAM (prototype).
//
// - Word-addressed: `req_addr` indexes `mem[req_addr]` (no byte shifting).
// - One request may be accepted per cycle when the previous response is being accepted.
// - Read response is 1-cycle delayed relative to request accept.
module pyc_sram #(
  parameter int ADDR_WIDTH = 10,
  parameter int DATA_WIDTH = 32,
  parameter int DEPTH = 1024
) (
  input  logic                  clk,
  input  logic                  rst,

  input  logic                  req_valid,
  output logic                  req_ready,
  input  logic [ADDR_WIDTH-1:0] req_addr,
  input  logic                  req_write,
  input  logic [DATA_WIDTH-1:0] req_wdata,
  input  logic [(DATA_WIDTH+7)/8-1:0] req_wstrb,

  output logic                  resp_valid,
  input  logic                  resp_ready,
  output logic [DATA_WIDTH-1:0] resp_rdata
);
  localparam int STRB_WIDTH = (DATA_WIDTH + 7) / 8;

  logic [DATA_WIDTH-1:0] mem [0:DEPTH-1];

  logic pending;
  logic [DATA_WIDTH-1:0] resp_rdata_r;

  assign resp_valid = pending;
  assign resp_rdata = resp_rdata_r;
  assign req_ready  = ~pending || (resp_valid && resp_ready);

  function automatic [DATA_WIDTH-1:0] apply_wstrb(
    input [DATA_WIDTH-1:0] oldv,
    input [DATA_WIDTH-1:0] wdata,
    input [STRB_WIDTH-1:0] wstrb
  );
    automatic logic [DATA_WIDTH-1:0] v;
    v = oldv;
    for (int i = 0; i < STRB_WIDTH; i++) begin
      if (wstrb[i]) v[i*8 +: 8] = wdata[i*8 +: 8];
    end
    apply_wstrb = v;
  endfunction

  always_ff @(posedge clk) begin
    if (rst) begin
      pending <= 1'b0;
      resp_rdata_r <= '0;
    end else begin
      logic fire_resp;
      logic fire_req;
      fire_resp = pending && resp_ready;
      fire_req  = req_valid && req_ready;

      if (fire_resp && !fire_req) begin
        pending <= 1'b0;
      end
      if (fire_req) begin
        int unsigned addr_i;
        addr_i = req_addr;
        // Indexing is word-based; user is responsible for address mapping.
        if (addr_i < DEPTH) begin
          if (req_write) begin
            mem[addr_i] <= apply_wstrb(mem[addr_i], req_wdata, req_wstrb);
            resp_rdata_r <= '0;
          end else begin
            resp_rdata_r <= mem[addr_i];
          end
        end else begin
          resp_rdata_r <= '0;
        end
        pending <= 1'b1;
      end
    end
  end
endmodule
