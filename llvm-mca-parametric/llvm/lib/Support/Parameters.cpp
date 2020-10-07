#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Parameters.h"

using namespace llvm;
using namespace cl;

namespace llvm {
  cl::opt<std::string> ParametersFilename("parameters", cl::desc("Parameters"), cl::value_desc("filename"));
  std::unique_ptr<mca::Parameters> Parameters;

  void ensureParametersInit() {
    if (Parameters) {
    } else {
      Parameters = llvm::make_unique<mca::Parameters>(ParametersFilename);
    }
  }
}
