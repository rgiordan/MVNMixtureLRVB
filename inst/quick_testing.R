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
  # You may also need to restart R.
  detach("package:MVNMixtureLRVB")
  library(MVNMixtureLRVB)
} else {  
  # Alternatively just compile it directly, which is slightly faster:
  sourceCpp(file.path(kSourceLocation, "src/build_matrices.cpp"))
  source(file.path(kSourceLocation, "R/fit_multivariate_normal_mixture_lib.R"))
}

#source(file.path(kSourceLocation, "inst/tests/runit_build_matrices.R"))


# TestUpdateFunctionsWithPriors

n <- 10
k <- 2
p <- 3
matrix.size <- p * (p + 1) / 2
par <- GenerateSampleParams(k=k, p=p, vars.scale=0.4^2)
par$true.probs <- rep(1 / k, k)
data <- GenerateMultivariateData(n,
                                 true.means = par$true.means,
                                 true.probs = par$true.probs,
                                 true.sigma = par$true.sigma)
x <- data$x
true.sigma.mat <- VectorizeMatrixList(par$true.sigma)
e.mu <- matrix(as.numeric(par$true.means), nrow(par$true.means), ncol(par$true.means))
e.mu2 <- (true.sigma.mat / rep(colSums(data$components), each=nrow(true.sigma.mat)) +
            GetVectorizedOuterProductMatrix(par$true.means))
z <- matrix(as.numeric(data$components), nrow(data$components), ncol(data$components))

lambda.par <- matrix(0, p * (p + 1) / 2, k)
n.par <- rep(0, k)

# Make some priors
priors <- GenerateSamplePriors(x, k, 10)

# Lambda update.
# Check the n prior:
lambda.update.1 <-
  UpdateLambdaPosterior(x=x, e_mu=e.mu, e_mu2=e.mu2, e_z=z,
                        TRUE, priors$lambda.prior.v.inv, rep(1.0, k))
lambda.update.2 <-
  UpdateLambdaPosterior(x=x, e_mu=e.mu, e_mu2=e.mu2, e_z=z,
                        TRUE, priors$lambda.prior.v.inv, rep(1001.0, k))

checkEqualsNumeric(lambda.update.2$n_par - lambda.update.1$n_par, rep(1000, k))
checkEqualsNumeric(lambda.update.2$lambda_par, lambda.update.1$lambda_par)

# Check the v.inv prior:
lambda.prior.v.inv <- matrix(ConvertSymmetricMatrixToVector(diag(p)), matrix.size, k)
lambda.update.1 <-
  UpdateLambdaPosterior(x=x, e_mu=e.mu, e_mu2=e.mu2, e_z=z,
                        TRUE, lambda.prior.v.inv, rep(1.0, k))
lambda.update.2 <-
  UpdateLambdaPosterior(x=x, e_mu=e.mu, e_mu2=e.mu2, e_z=z,
                        TRUE, 2 * lambda.prior.v.inv, rep(1.0, k))
v.par.1 <- solve(ConvertVectorToSymmetricMatrix(lambda.update.1$lambda_par[,1]))
v.par.2 <- solve(ConvertVectorToSymmetricMatrix(lambda.update.2$lambda_par[,1]))

checkEqualsNumeric(lambda.update.2$n_par, lambda.update.1$n_par)
checkEqualsNumeric(v.par.2 - v.par.1, diag(p))

# For the tests below just use one with no prior.
lambda.update <-
  UpdateLambdaPosterior(x=x, e_mu=e.mu, e_mu2=e.mu2, e_z=z,
                        FALSE, priors$lambda.prior.v.inv, rep(0.0, k))
lambda.par <- lambda.update$lambda_par
n.par <- lambda.update$n_par
e.log.det.lambda <- WishartELogDet(lambda.par, n.par)
e.lambda <- rep(n.par, each=p * (p + 1) / 2) * lambda.par
e.lambda.inv.mat <- InvertLinearizedMatrices(e.lambda)

# Pi update
e.log.pi <- GetELogDirichlet(colSums(z))

# Z update
GetZMatrixInPlace(z=z,
                  x=x, e_mu=e.mu, e_mu2=e.mu2,
                  e_lambda=e.lambda, e_log_det_lambda=e.log.det.lambda,
                  e_log_pi=e.log.pi)

# Mu update
mu.prior.info <- 0.1 * diag(p)
mu.prior.info.mat <- matrix(ConvertSymmetricMatrixToVector(mu.prior.info), matrix.size, k)
mu.update <- UpdateMuPosterior(x=x, e_lambda_inv_mat=e.lambda.inv.mat, e_z=z,
                               use_prior=TRUE, matrix(0, p, k), mu.prior.info.mat)
mu.update

e.mu <- mu.update$e_mu
e.mu2 <- mu.update$e_mu2

for (this.k in 1:k) {
  this.z <- z[, this.k]
  expected.mean <- colSums(this.z * x) / sum(this.z)
  checkEqualsNumeric(expected.mean, e.mu[, this.k])
  this.e.mu2 <- ConvertVectorToSymmetricMatrix(e.mu2[, this.k])
  this.mu.outer <- e.mu[, this.k] %*% t(e.mu[, this.k])
  this.mu.var <- ConvertVectorToSymmetricMatrix(e.lambda.inv.mat[, this.k]) / sum(this.z)
  checkEquals(this.mu.outer + this.mu.var, this.e.mu2)
}