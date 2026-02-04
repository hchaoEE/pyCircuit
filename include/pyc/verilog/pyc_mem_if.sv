// Memory request/response interface (prototype).
//
// One request channel and one response channel, both ready/valid.
interface pyc_mem_if #(
  parameter int ADDR_WIDTH = 32,
  parameter int DATA_WIDTH = 32
) ();
  localparam int STRB_WIDTH = (DATA_WIDTH + 7) / 8;

  // Request
  logic                  req_valid;
  logic                  req_ready;
  logic [ADDR_WIDTH-1:0] req_addr;
  logic                  req_write;
  logic [DATA_WIDTH-1:0] req_wdata;
  logic [STRB_WIDTH-1:0] req_wstrb;

  // Response
  logic                  resp_valid;
  logic                  resp_ready;
  logic [DATA_WIDTH-1:0] resp_rdata;

  modport master(
    output req_valid,
    input  req_ready,
    output req_addr,
    output req_write,
    output req_wdata,
    output req_wstrb,
    input  resp_valid,
    output resp_ready,
    input  resp_rdata
  );

  modport slave(
    input  req_valid,
    output req_ready,
    input  req_addr,
    input  req_write,
    input  req_wdata,
    input  req_wstrb,
    output resp_valid,
    input  resp_ready,
    output resp_rdata
  );
endinterface

