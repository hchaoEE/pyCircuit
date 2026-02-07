module tb_janus_bcc_ooo_pyc;
  logic clk;
  logic rst;

  logic [63:0] boot_pc;
  logic [63:0] boot_sp;

  logic halted;
  logic [63:0] cycles;
  logic [63:0] pc;
  logic [63:0] fpc;
  logic [63:0] a0;
  logic [63:0] a1;
  logic [63:0] ra;
  logic [63:0] sp;

  logic [11:0] commit_op;
  logic commit_fire;
  logic [63:0] commit_value;
  logic [1:0] commit_dst_kind;
  logic [5:0] commit_dst_areg;
  logic [5:0] commit_pdst;
  logic [4:0] rob_count;
  logic [2:0] br_kind;
  logic commit_cond;
  logic [63:0] commit_tgt;
  logic [63:0] br_base_pc;
  logic [63:0] br_off;

  logic commit_store_fire;
  logic [63:0] commit_store_addr;
  logic [63:0] commit_store_data;
  logic [3:0] commit_store_size;

  logic commit_fire0;
  logic commit_fire1;
  logic commit_fire2;
  logic commit_fire3;
  logic [63:0] commit_pc0;
  logic [63:0] commit_pc1;
  logic [63:0] commit_pc2;
  logic [63:0] commit_pc3;
  logic [11:0] commit_op0;
  logic [11:0] commit_op1;
  logic [11:0] commit_op2;
  logic [11:0] commit_op3;
  logic [63:0] commit_value0;
  logic [63:0] commit_value1;
  logic [63:0] commit_value2;
  logic [63:0] commit_value3;

  logic [63:0] ct0;
  logic [63:0] cu0;
  logic [63:0] st0;
  logic [63:0] su0;

  logic issue_fire;
  logic [11:0] issue_op;
  logic [63:0] issue_pc;
  logic [3:0] issue_rob;
  logic [5:0] issue_sl;
  logic [5:0] issue_sr;
  logic [5:0] issue_sp;
  logic [5:0] issue_pdst;
  logic [63:0] issue_sl_val;
  logic [63:0] issue_sr_val;
  logic [63:0] issue_sp_val;
  logic issue_is_load;
  logic issue_is_store;
  logic store_pending;
  logic store_pending_older;
  logic [63:0] mem_raddr;
  logic dispatch_fire;
  logic [11:0] dec_op;

  logic mmio_uart_valid;
  logic [7:0] mmio_uart_data;
  logic mmio_exit_valid;
  logic [31:0] mmio_exit_code;

  janus_bcc_ooo_pyc dut (
      .clk(clk),
      .rst(rst),
      .boot_pc(boot_pc),
      .boot_sp(boot_sp),
      .halted(halted),
      .cycles(cycles),
      .pc(pc),
      .fpc(fpc),
      .a0(a0),
      .a1(a1),
      .ra(ra),
      .sp(sp),
      .commit_op(commit_op),
      .commit_fire(commit_fire),
      .commit_value(commit_value),
      .commit_dst_kind(commit_dst_kind),
      .commit_dst_areg(commit_dst_areg),
      .commit_pdst(commit_pdst),
      .br_kind(br_kind),
      .commit_cond(commit_cond),
      .commit_tgt(commit_tgt),
      .br_base_pc(br_base_pc),
      .br_off(br_off),
      .commit_store_fire(commit_store_fire),
      .commit_store_addr(commit_store_addr),
      .commit_store_data(commit_store_data),
      .commit_store_size(commit_store_size),
      .commit_fire0(commit_fire0),
      .commit_pc0(commit_pc0),
      .commit_op0(commit_op0),
      .commit_value0(commit_value0),
      .commit_fire1(commit_fire1),
      .commit_pc1(commit_pc1),
      .commit_op1(commit_op1),
      .commit_value1(commit_value1),
      .commit_fire2(commit_fire2),
      .commit_pc2(commit_pc2),
      .commit_op2(commit_op2),
      .commit_value2(commit_value2),
      .commit_fire3(commit_fire3),
      .commit_pc3(commit_pc3),
      .commit_op3(commit_op3),
      .commit_value3(commit_value3),
      .rob_count(rob_count),
      .ct0(ct0),
      .cu0(cu0),
      .st0(st0),
      .su0(su0),
      .issue_fire(issue_fire),
      .issue_op(issue_op),
      .issue_pc(issue_pc),
      .issue_rob(issue_rob),
      .issue_sl(issue_sl),
      .issue_sr(issue_sr),
      .issue_sp(issue_sp),
      .issue_pdst(issue_pdst),
      .issue_sl_val(issue_sl_val),
      .issue_sr_val(issue_sr_val),
      .issue_sp_val(issue_sp_val),
      .issue_is_load(issue_is_load),
      .issue_is_store(issue_is_store),
      .store_pending(store_pending),
      .store_pending_older(store_pending_older),
      .mem_raddr(mem_raddr),
      .dispatch_fire(dispatch_fire),
      .dec_op(dec_op),
      .mmio_uart_valid(mmio_uart_valid),
      .mmio_uart_data(mmio_uart_data),
      .mmio_exit_valid(mmio_exit_valid),
      .mmio_exit_code(mmio_exit_code)
  );

  always #5 clk = ~clk;

  function automatic logic [31:0] mem_read32(input int unsigned addr);
    mem_read32 = {dut.mem.mem[addr + 3], dut.mem.mem[addr + 2], dut.mem.mem[addr + 1], dut.mem.mem[addr + 0]};
  endfunction

  string memh_path;
  string vcd_path;
  string log_path;
  int log_fd;
  bit log_commits;

  longint unsigned max_cycles;
  longint unsigned expected_mem100;
  longint unsigned expected_a0;
  bit has_expected_mem100;
  bit has_expected_a0;
  bit has_expected_exit_pc;
  logic [31:0] got_mem100;
  longint unsigned i;
  longint unsigned expected_exit_pc;
  int unsigned exit_pc_stable_cycles;
  int unsigned exit_pc_stable;
  bit done;

  initial begin
    clk = 1'b0;
    rst = 1'b1;

    boot_pc = 64'h0000_0000_0001_0000;
    boot_sp = 64'h0000_0000_0002_0000;
    max_cycles = 400000;

    if (!$value$plusargs("memh=%s", memh_path)) begin
      memh_path = "janus/programs/test_or.memh";
    end

    void'($value$plusargs("boot_pc=%h", boot_pc));
    void'($value$plusargs("boot_sp=%h", boot_sp));
    void'($value$plusargs("max_cycles=%d", max_cycles));

    has_expected_mem100 = $value$plusargs("expected_mem100=%h", expected_mem100);
    has_expected_a0 = $value$plusargs("expected_a0=%h", expected_a0);
    has_expected_exit_pc = $value$plusargs("expected_exit_pc=%h", expected_exit_pc);
    exit_pc_stable_cycles = 8;
    void'($value$plusargs("exit_pc_stable=%d", exit_pc_stable_cycles));
    exit_pc_stable = 0;
    done = 0;

    vcd_path = "janus/generated/janus_bcc_ooo_pyc/tb_janus_bcc_ooo_pyc_sv.vcd";
    log_path = "janus/generated/janus_bcc_ooo_pyc/tb_janus_bcc_ooo_pyc_sv.log";
    void'($value$plusargs("vcd=%s", vcd_path));
    void'($value$plusargs("log=%s", log_path));
    log_commits = $test$plusargs("logcommits");

    if (!$test$plusargs("notrace")) begin
      $display("tb_janus_bcc_ooo_pyc: dumping VCD to %s", vcd_path);
      $dumpfile(vcd_path);
      $dumpvars(0, tb_janus_bcc_ooo_pyc);
    end

    if (!$test$plusargs("nolog")) begin
      log_fd = $fopen(log_path, "w");
      $fdisplay(log_fd, "tb_janus_bcc_ooo_pyc(SV): memh=%s", memh_path);
      $fdisplay(log_fd, "cycle,time,halted,slot,commit_pc,pc,fpc,cycles,rob_count,commit_op,commit_value,a0,a1,ra,sp,br_kind,commit_cond,commit_tgt,br_base_pc,br_off,st_fire,st_addr,st_data,st_size");
    end else begin
      log_fd = 0;
    end

    $display("tb_janus_bcc_ooo_pyc: memh=%s boot_pc=0x%016x max_cycles=%0d", memh_path, boot_pc, max_cycles);

    if ($test$plusargs("zeromem")) begin
      // Deterministic BSS/heap bring-up: clear backing RAM before loading memh.
      for (int unsigned k = 0; k < 1048576; k++) begin
        dut.mem.mem[k] = 8'h00;
      end
    end

    $readmemh(memh_path, dut.mem.mem);

    repeat (5) @(posedge clk);
    rst = 1'b0;

    i = 0;
    while (i < max_cycles && !halted && !done) begin
      @(posedge clk);
      if (mmio_uart_valid) begin
        $write("%c", mmio_uart_data);
      end
      if (mmio_exit_valid) begin
        done = 1;
      end
      if (log_fd != 0 && log_commits) begin
        if (commit_fire0) begin
          $fdisplay(log_fd,
                    "%0d,%0t,%0b,%0d,0x%016x,0x%016x,0x%016x,%0d,%0d,0x%03x,0x%016x,0x%016x,0x%016x,0x%016x,0x%016x,%0d,%0b,0x%016x,0x%016x,0x%016x,%0b,0x%016x,0x%016x,0x%01x",
                    i,
                    $time,
                    halted,
                    0,
                    commit_pc0,
                    pc,
                    fpc,
                    cycles,
                    rob_count,
                    commit_op0,
                    commit_value0,
                    a0,
                    a1,
                    ra,
                    sp,
                    br_kind,
                    commit_cond,
                    commit_tgt,
                    br_base_pc,
                    br_off,
                    commit_store_fire,
                    commit_store_addr,
                    commit_store_data,
                    commit_store_size);
        end
        if (commit_fire1) begin
          $fdisplay(log_fd,
                    "%0d,%0t,%0b,%0d,0x%016x,0x%016x,0x%016x,%0d,%0d,0x%03x,0x%016x,0x%016x,0x%016x,0x%016x,0x%016x,%0d,%0b,0x%016x,0x%016x,0x%016x,%0b,0x%016x,0x%016x,0x%01x",
                    i,
                    $time,
                    halted,
                    1,
                    commit_pc1,
                    pc,
                    fpc,
                    cycles,
                    rob_count,
                    commit_op1,
                    commit_value1,
                    a0,
                    a1,
                    ra,
                    sp,
                    br_kind,
                    commit_cond,
                    commit_tgt,
                    br_base_pc,
                    br_off,
                    commit_store_fire,
                    commit_store_addr,
                    commit_store_data,
                    commit_store_size);
        end
        if (commit_fire2) begin
          $fdisplay(log_fd,
                    "%0d,%0t,%0b,%0d,0x%016x,0x%016x,0x%016x,%0d,%0d,0x%03x,0x%016x,0x%016x,0x%016x,0x%016x,0x%016x,%0d,%0b,0x%016x,0x%016x,0x%016x,%0b,0x%016x,0x%016x,0x%01x",
                    i,
                    $time,
                    halted,
                    2,
                    commit_pc2,
                    pc,
                    fpc,
                    cycles,
                    rob_count,
                    commit_op2,
                    commit_value2,
                    a0,
                    a1,
                    ra,
                    sp,
                    br_kind,
                    commit_cond,
                    commit_tgt,
                    br_base_pc,
                    br_off,
                    commit_store_fire,
                    commit_store_addr,
                    commit_store_data,
                    commit_store_size);
        end
        if (commit_fire3) begin
          $fdisplay(log_fd,
                    "%0d,%0t,%0b,%0d,0x%016x,0x%016x,0x%016x,%0d,%0d,0x%03x,0x%016x,0x%016x,0x%016x,0x%016x,0x%016x,%0d,%0b,0x%016x,0x%016x,0x%016x,%0b,0x%016x,0x%016x,0x%01x",
                    i,
                    $time,
                    halted,
                    3,
                    commit_pc3,
                    pc,
                    fpc,
                    cycles,
                    rob_count,
                    commit_op3,
                    commit_value3,
                    a0,
                    a1,
                    ra,
                    sp,
                    br_kind,
                    commit_cond,
                    commit_tgt,
                    br_base_pc,
                    br_off,
                    commit_store_fire,
                    commit_store_addr,
                    commit_store_data,
                    commit_store_size);
        end
      end
      if (has_expected_exit_pc && pc === expected_exit_pc[63:0] && fpc === expected_exit_pc[63:0]) begin
        exit_pc_stable++;
        if (exit_pc_stable >= exit_pc_stable_cycles) begin
          done = 1;
        end
      end else begin
        exit_pc_stable = 0;
      end
      i++;
    end

    if (!halted && !done) begin
      $fatal(
          1,
          $sformatf(
              "FAIL: did not halt (pc=0x%016x fpc=0x%016x cycles=%0d rob_count=%0d halted=%0b done=%0b)\n  commit_fire=%0b commit_op=0x%03x commit_value=0x%016x commit_dst_kind=%0d commit_dst_areg=%0d commit_pdst=%0d\n  commit_store_fire=%0b commit_store_addr=0x%016x commit_store_data=0x%016x commit_store_size=0x%x\n  dispatch_fire=%0b dec_op=0x%03x\n  issue_fire=%0b issue_op=0x%03x issue_pc=0x%016x issue_rob=%0d issue_is_load=%0b issue_is_store=%0b store_pending=%0b store_pending_older=%0b mem_raddr=0x%016x\n  a0=0x%016x a1=0x%016x ra=0x%016x sp=0x%016x\n  mmio_exit_valid=%0b mmio_exit_code=0x%08x",
              pc,
              fpc,
              cycles,
              rob_count,
              halted,
              done,
              commit_fire,
              commit_op,
              commit_value,
              commit_dst_kind,
              commit_dst_areg,
              commit_pdst,
              commit_store_fire,
              commit_store_addr,
              commit_store_data,
              commit_store_size,
              dispatch_fire,
              dec_op,
              issue_fire,
              issue_op,
              issue_pc,
              issue_rob,
              issue_is_load,
              issue_is_store,
              store_pending,
              store_pending_older,
              mem_raddr,
              a0,
              a1,
              ra,
              sp,
              mmio_exit_valid,
              mmio_exit_code
          )
      );
    end

    got_mem100 = mem_read32(32'h0000_0100);

    if (has_expected_mem100 && got_mem100 !== expected_mem100[31:0]) begin
      $fatal(1, "FAIL: mem[0x100]=0x%08x expected=0x%08x", got_mem100, expected_mem100[31:0]);
    end

    if (has_expected_a0 && a0 !== expected_a0[63:0]) begin
      $fatal(1, "FAIL: a0=0x%016x expected=0x%016x", a0, expected_a0[63:0]);
    end

    if (log_fd != 0) begin
      $fdisplay(log_fd, "PASS: cycles=%0d pc=0x%016x fpc=0x%016x a0=0x%016x mem100=0x%08x", cycles, pc, fpc, a0, got_mem100);
      $fclose(log_fd);
    end

    $display("PASS: cycles=%0d pc=0x%016x a0=0x%016x mem[0x100]=0x%08x", cycles, pc, a0, got_mem100);
    $finish;
  end

endmodule
