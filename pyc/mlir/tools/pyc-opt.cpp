#include "pyc/Dialect/PYC/PYCDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<pyc::PYCDialect, mlir::arith::ArithDialect, mlir::func::FuncDialect, mlir::scf::SCFDialect>();
  mlir::func::registerInlinerExtension(registry);
  registerAllPasses();
  return asMainReturnCode(MlirOptMain(argc, argv, "pyc-opt\n", registry));
}
