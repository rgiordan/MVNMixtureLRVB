library(RUnit)
library(MVNMixtureLRVB)

testsuite <- defineTestSuite("testsuite",
                             dirs = file.path(path.package(package="MVNMixtureLRVB"),
                                              "tests"),
                                 testFileRegexp = "^runit.+\\.r",
                                 testFuncRegexp = "^Test.+",
                                 rngKind = "Marsaglia-Multicarry",
                                 rngNormalKind = "Kinderman-Ramage")

testResult <- runTestSuite(testsuite)
printTextProtocol(testResult)
