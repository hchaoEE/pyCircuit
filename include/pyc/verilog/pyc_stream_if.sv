// Ready/valid stream interface (prototype).
//
// This is an optional convenience wrapper; current primitives use explicit
// in_valid/in_ready/in_data style ports.
interface pyc_stream_if #(
  parameter int WIDTH = 1
) ();
  logic             valid;
  logic             ready;
  logic [WIDTH-1:0] data;

  modport producer(output valid, output data, input ready);
  modport consumer(input valid, input data, output ready);
endinterface

