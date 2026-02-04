#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

namespace pyc {

struct CppEmitterOptions {};

::mlir::LogicalResult emitCpp(::mlir::ModuleOp module, ::llvm::raw_ostream &os,
                              const CppEmitterOptions &opts = {});

} // namespace pyc
