library(RUnit)
library(MVNMixtureLRVB)

testsuite <- defineTestSuite("testsuite",
                             dirs = file.path(path.package(package="MVNMixtureLRVB"),
                                              "tests"),
                             testFileRegexp = "^runit.+\\.R",
                             testFuncRegexp = "^Test.+",
                             rngKind = "Mersenne-Twister",
                             rngNormalKind = "Inversion")

testResult <- runTestSuite(testsuite)
printTextProtocol(testResult)
