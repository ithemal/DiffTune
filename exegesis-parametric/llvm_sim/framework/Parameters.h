//===--------------------- Parameters.h -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// A builder class for instructions that are statically analyzed by llvm-mca.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MCA_PARAMETERS_H
#define LLVM_MCA_PARAMETERS_H

#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <string>

namespace llvm {
namespace mca {

class Parameters {
  const std::string ParameterFile;
  bool Present;

public:
  std::unordered_map<std::string, std::string> paramMap;

  Parameters(StringRef parameterFile) : ParameterFile(parameterFile), Present(true) {
    /* std::cout << "ParamFile: " << ParameterFile << std::endl; */
    if (ParameterFile.compare("noop") == 0) {
      Present = false;
    } else {
      std::ifstream in(ParameterFile);

      std::string line;

      while (std::getline(in, line)) {
        std::istringstream iss(line);
        std::vector<std::string> results(std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>());
        paramMap.emplace(results[0], results[1]);
      }
    }
  }

  bool isPresent() const {
    return Present;
  }

  std::string read(const std::string parameter) const {
    auto res = paramMap.find(parameter);
    if (res != paramMap.end()) {
      return res->second;
    } else {
      return "";
    }
  }

  bool contains(std::string parameter) const {
    auto res = paramMap.count(parameter) > 0;
    return res;
  }

  int readInt(std::string parameter) const {
    return std::stoi(read(parameter));
  }

  void print(std::string parameter, int timing) const {
    std::cerr << parameter << " " << timing << std::endl;;
  }


};
} // namespace mca
} // namespace llvm

#endif // LLVM_MCA_PARAMETERS_H
