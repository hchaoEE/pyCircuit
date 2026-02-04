#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace pyc {

struct VerilogEmitterOptions {
  bool includePrimitives = true;
};

::mlir::LogicalResult emitVerilog(::mlir::ModuleOp module, ::llvm::raw_ostream &os,
                                  const VerilogEmitterOptions &opts = {});

} // namespace pyc
