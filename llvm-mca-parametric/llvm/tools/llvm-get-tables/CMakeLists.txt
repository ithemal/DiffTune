include_directories(include)

set(LLVM_LINK_COMPONENTS
  AllTargetsAsmPrinters
  AllTargetsAsmParsers
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  MCA
  MC
  MCParser
  Support
  )

add_llvm_tool(llvm-get-tables
  llvm-get-tables.cpp
  )

set(LLVM_MCA_SOURCE_DIR ${CURRENT_SOURCE_DIR})
