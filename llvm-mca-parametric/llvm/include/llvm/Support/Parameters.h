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

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"

#include <iostream>
#include <unordered_map>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <string>
#include <unistd.h>

extern char** environ;

namespace llvm {
  extern cl::opt<std::string> ParametersFilename;
  void ensureParametersInit();

  namespace mca {

class Parameters {
  const std::string ParameterFile;
  bool Present;
  std::unordered_map<std::string, std::string> paramMap;

public:
  Parameters(StringRef parameterFile) : ParameterFile(parameterFile), Present(true) {
    /* std::cout << "ParamFile: " << ParameterFile << std::endl; */
    if (ParameterFile.compare("noop") == 0) {
      Present = false;
    } else if (ParameterFile.compare("env") == 0) {
      int i = 1;
      char *s = *environ;

      for (; s; i++) {
	std::stringstream check1(s); 
	std::string intermediate; 
	std::vector <std::string> tokens; 
	while(std::getline(check1, intermediate, '=')) {
		tokens.push_back(intermediate); 
	}
	std::replace(tokens[0].begin(), tokens[0].end(), '_', '-');
	paramMap.emplace(tokens[0], tokens[1]);
	s = *(environ+i);
      }
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
      std::cerr << "Must contain: \"" << parameter << "\"" << std::endl;
      assert(false);
      return "";
    }
  }

  bool contains(std::string parameter) const {
    return paramMap.count(parameter) > 0;
  }

  int readInt(std::string parameter) const {
    // try {
      return std::stoi(read(parameter));
    // } catch (const std::invalid_argument& ia) {
    //   std::cerr << "Invalid argument: " << ia.what() << ", " << parameter << std::endl;
    //   exit(1);
    // }
  }

  void print(std::string parameter, int timing) const {
    std::cerr << parameter << " " << timing << std::endl;;
  }


};
} // namespace mca

  extern std::unique_ptr<mca::Parameters> Parameters;
} // namespace llvm

#endif // LLVM_MCA_PARAMETERS_H
