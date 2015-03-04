# A file for quickly iterating on changes to the package.
library(devtools)
library(Rcpp)

kSourceLocation <- file.path(Sys.getenv("GIT_REPO_LOC"),  "MVNMixtureLRVB")
setwd(kSourceLocation)

kBuildPackage <- FALSE
if (kBuildPackage) {
  # Run this if you've changed which functions are exported.
  if (F) {
    Rcpp::compileAttributes()
    # After this you have to also manually copy the typedefs to the top of
    # the RcppExports.cpp file.
  }
  install(kSourceLocation)  
} else {  
  # Alternatively just compile it directly, which is slightly faster:
  sourceCpp(file.path(kSourceLocation, "src/build_matrices.cpp"))
  source(file.path(kSourceLocation, "R/fit_multivariate_normal_mixture_lib.R"))
}

# You may also need to restart R.
#detach("package:MVNMixtureLRVB")
library(MVNMixtureLRVB)

source(file.path(kSourceLocation, "inst/tests/runit_build_matrices.R"))
TestXVariance()
