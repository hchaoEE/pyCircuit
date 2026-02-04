#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/StringRef.h"

namespace pyc {

class PYCDialect : public ::mlir::Dialect {
public:
  explicit PYCDialect(::mlir::MLIRContext *ctx);
  static ::llvm::StringRef getDialectNamespace() { return "pyc"; }

  void initialize();

  ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
  void printType(::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const override;
};

} // namespace pyc
