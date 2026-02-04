package pyc_handshake_pkg;
  function automatic logic fire(input logic valid, input logic ready);
    fire = valid && ready;
  endfunction
endpackage

