// Byte-addressed memory (prototype).
//
// - `DEPTH` is in bytes.
// - Combinational little-endian read window.
// - Byte-enable write on posedge.
module pyc_byte_mem #(
  parameter int ADDR_WIDTH = 64,
  parameter int DATA_WIDTH = 64,
  parameter int DEPTH = 1024
) (
  input  logic                  clk,
  input  logic                  rst,

  input  logic [ADDR_WIDTH-1:0] raddr,
  output logic [DATA_WIDTH-1:0] rdata,

  input  logic                  wvalid,
  input  logic [ADDR_WIDTH-1:0] waddr,
  input  logic [DATA_WIDTH-1:0] wdata,
  input  logic [(DATA_WIDTH+7)/8-1:0] wstrb
);
  localparam int STRB_WIDTH = (DATA_WIDTH + 7) / 8;

  // Byte storage.
  logic [7:0] mem [0:DEPTH-1];

  // Combinational read: assemble DATA_WIDTH bits from successive bytes.
  always_comb begin
    int unsigned a;
    a = raddr[31:0];
    rdata = '0;
    for (int i = 0; i < STRB_WIDTH; i++) begin
      if ((a + i) < DEPTH)
        rdata[8 * i +: 8] = mem[a + i];
    end
  end

  // Byte-enable write.
  always_ff @(posedge clk) begin
    if (rst) begin
      // No implicit clear; memory contents are left as-is (TB may $readmemh()).
    end else if (wvalid) begin
      int unsigned a;
      a = waddr[31:0];
      for (int i = 0; i < STRB_WIDTH; i++) begin
        if (wstrb[i] && ((a + i) < DEPTH))
          mem[a + i] <= wdata[8 * i +: 8];
      end
    end
  end
endmodule

