#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"

namespace pyc {

std::unique_ptr<::mlir::Pass> createFuseCombPass();
std::unique_ptr<::mlir::Pass> createLowerSCFToPYCStaticPass();

} // namespace pyc
