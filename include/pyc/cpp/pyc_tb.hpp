#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "pyc_bits.hpp"

namespace pyc::cpp {

struct TbClock {
  Wire<1> *clk = nullptr;
  std::uint64_t half_period_steps = 1;
  std::uint64_t phase_steps = 0;

  void set(bool high) const {
    if (clk)
      *clk = Wire<1>(high ? 1u : 0u);
  }

  void toggle() const {
    if (!clk)
      return;
    bool high = clk->toBool();
    *clk = Wire<1>(high ? 0u : 1u);
  }

  bool shouldToggle(std::uint64_t step) const {
    if (!clk || half_period_steps == 0)
      return false;
    return ((step + phase_steps) % half_period_steps) == 0;
  }
};

template <typename Dut>
class Testbench {
public:
  explicit Testbench(Dut &dut) : dut_(dut) {}

  void addClock(Wire<1> &clk, std::uint64_t halfPeriodSteps = 1, std::uint64_t phaseSteps = 0, bool startHigh = false) {
    TbClock c;
    c.clk = &clk;
    c.half_period_steps = (halfPeriodSteps == 0) ? 1 : halfPeriodSteps;
    c.phase_steps = phaseSteps;
    c.set(startHigh);
    clocks_.push_back(c);
  }

  std::size_t numClocks() const { return clocks_.size(); }
  std::uint64_t timeSteps() const { return time_; }

  void step() {
    // Drive combinational logic before clock edges.
    dut_.eval();

    // Toggle all clocks that have an edge on this step.
    for (const auto &c : clocks_) {
      if (c.shouldToggle(time_))
        c.toggle();
    }

    // Sequential update (modules detect posedges internally).
    dut_.tick();

    // Re-evaluate combinational logic after state updates.
    dut_.eval();

    time_++;
  }

  void runSteps(std::uint64_t steps) {
    for (std::uint64_t i = 0; i < steps; i++)
      step();
  }

  void runCycles(std::uint64_t cycles) { runCycles(/*clockIdx=*/0, cycles); }

  void runCycles(std::size_t clockIdx, std::uint64_t cycles) {
    if (clockIdx >= clocks_.size())
      return;
    const auto hp = clocks_[clockIdx].half_period_steps;
    runSteps(cycles * 2u * hp);
  }

  void reset(Wire<1> &rst, std::uint64_t cyclesAsserted = 2, std::uint64_t cyclesDeasserted = 1) {
    rst = Wire<1>(1);
    runCycles(cyclesAsserted);
    rst = Wire<1>(0);
    runCycles(cyclesDeasserted);
  }

private:
  Dut &dut_;
  std::vector<TbClock> clocks_{};
  std::uint64_t time_ = 0;
};

} // namespace pyc::cpp

