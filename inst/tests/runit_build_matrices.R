library(MVNMixtureLRVB)
library(RUnit)
library(Matrix)
library(mvtnorm)
library(numDeriv)

TestXVariance <- function() {
  # Use a simulation to test the x covariance under infinitesimal perturbations.

  # Set a seed to avoid flakiness.
  set.seed(42)
  n <- 2
  p <- 3
  x <- matrix(1:(n * p), nrow=n, ncol=p)

  x.cov <- GetXVariance(x)
  checkEqualsNumeric(x.cov, t(x.cov))
  x.par <- PackXParameters(x)

  ConvertXRowToMatrix <- function(x.row) {
    x.row.mat <- matrix(x.row, nrow=n, ncol=p, byrow=TRUE)
    return(x.row.mat)
  }

  ExpandXRow <- function(x.row) {
    return(PackXParameters(ConvertXRowToMatrix(x.row)))
  }

  checkEqualsNumeric(x, ConvertXRowToMatrix(x.par[1:(n * p)]))
  checkEqualsNumeric(x.par, ExpandXRow(x.par[1:(n * p)]))

  n.sims <- 1e5
  epsilon <- 1e-4
  x.sim <- rmvnorm(n.sims, mean=x.par[1:(n * p)], sigma=epsilon * diag(n * p))
  x.par.sim <- t(apply(x.sim, MARGIN=1, ExpandXRow))
  x.cov.sim <- cov(x.par.sim)

  checkTrue(max(abs(x.cov.sim - epsilon * x.cov)) < epsilon)
}

TestXVarianceSubset <- function() {
  n <- 3
  p <- 2
  matrix.size <- p * (p + 1) / 2
  x <- matrix(1:(n * p), nrow=n, ncol=p)

  x.cov <- GetXVariance(x)

  # The subsets are zero-indexed
  x.cov.sub <- GetXVarianceSubset(x, c(0, 2))

  # Indices of the removed components.
  x2.indices <- c(p + 1:p, n * p + matrix.size + 1:matrix.size)

  checkEqualsNumeric(x.cov[-x2.indices, -x2.indices], x.cov.sub)
}




TestWishartELogDet <- function() {
  k <- 4
  p <- 5
  par <- GenerateSampleParams(k=k, p=p, vars.scale=0.4^2)
  lambda.par <- InvertLinearizedMatrices(VectorizeMatrixList(par$true.sigma))
  lambda.list <- ListifyVectorizedMatrix(lambda.par)
  n.par <- 10 + 10 * runif(k)

  e.log.det <- WishartELogDet(lambda.par, n.par)
  for (this.k in 1:k) {
    checkEqualsNumeric(p * log(2) + log(det(lambda.list[[this.k]])) +
                       MVDigamma(n.par[[this.k]] / 2, p),
                       e.log.det[[this.k]])
  }
}


TestGammaFunctions <- function() {
  p <- 5
  for (i in 10 * (1:10)) {
    checkEqualsNumeric(lgamma(i), CppLgamma(i))
    checkEqualsNumeric(digamma(i), CppDigamma(i))
    checkEqualsNumeric(trigamma(i), CppTrigamma(i))
    checkEqualsNumeric(MVLgamma(i, p), CppMultivariateLgamma(i, p))
    checkEqualsNumeric(MVDigamma(i, p), CppMultivariateDigamma(i, p))
    checkEqualsNumeric(MVTrigamma(i, p), CppMultivariateTrigamma(i, p))
  }
}


TestConvertSymmetricMatrixToVector <- function() {
  set.seed(42)
  n <- 10
  x <- matrix(runif(n * n), n, n)
  x <- x * t(x)
  x.vec <- ConvertSymmetricMatrixToVector(x)
  x.index <- 0
  for (j in 1:n) {
    for (i in 1:j) {
      x.index <- x.index + 1
      checkEquals(x.vec[x.index], x[i, j])
    }
  }
}

TestConvertVectorToSymmetricMatrix <- function() {
  set.seed(42)
  n <- 10
  x.vec <- runif(n * (n + 1) / 2)
  x <- ConvertVectorToSymmetricMatrix(x.vec)
  checkEqualsNumeric(x, t(x))
  x.index <- 0
  for (j in 1:n) {
    for (i in 1:j) {
      x.index <- x.index + 1
      checkEquals(x.vec[x.index], x[i, j])
    }
  }
}


CheckWishartMomentFormulas <- function() {
  # With simulations, confirm the formulas for the Wishart distribution
  # This doesn't test any of the C++.

  n <- 5
  v.par <- diag(n) + 0.1 * matrix(runif(n * n), n, n)
  v.par <- v.par + t(v.par)

  v.vec <- ConvertSymmetricMatrixToVector(v.par)
  sigma.par <- solve(v.par)
  n.par <- 10
  n.vec <- length(v.vec)

  # The theoretical covariances
  cov.mat <- matrix(0, n.vec, n.vec)
  for (j1 in 1:n) {
    for (i1 in 1:j1) {
      for (j2 in 1:n) {
        for (i2 in 1:j2) {
          index1 <- GetUpperTriangularIndex(i1 - 1, j1 - 1) + 1
          index2 <- GetUpperTriangularIndex(i2 - 1, j2 - 1) + 1
          cov.mat[index1, index2] <-
            n.par * (v.par[i2, j1] * v.par[i1, j2] + v.par[i2, i1] * v.par[j1, j2])
          cov.mat[index2, index1] <- cov.mat[index1, index2]
        }
      }
    }
  }

  n.samples <- 1e5
  x <- rWishart(n = n.samples, df = n.par, Sigma = v.par)
  x.mean <- apply(x, MARGIN=c(1, 2), mean)
  mean(abs(x.mean - n.par * v.par))

  x.vecs <- t(matrix(apply(x, MARGIN=3, ConvertSymmetricMatrixToVector), ncol=n.samples))
  x.log.dets <- apply(x, MARGIN=3, function(x.mat) { log(det(as.matrix(x.mat))) })

  # Check E(log(det(X)))
  checkEqualsNumeric(log(det(x[,,1])), x.log.dets[1])
  checkEqualsNumeric(mean(x.log.dets),
                     n * log(2) + log(det(v.par)) + MVDigamma(n.par / 2, n),
                     tolerance=1e-3)

  checkEqualsNumeric
  x.all <- cbind(x.vecs, x.log.dets)

  x.cov <- cov(x.all)

  mean(abs(x.cov[1:n.vec, 1:n.vec] - cov.mat))
  plot(2 * v.vec, x.cov[1:n.vec, n.vec + 1]); abline(0, 1)
  x.cov[n.vec + 1, n.vec + 1]
  CppMultivariateTrigamma(n.par / 2, n)
}



TestMultivariateNormalCovariance <- function() {
  # A somewhat time-consuming and crappy simulation test of the MVN covariance.

  ##############
  # THIS SHOULD BE IMPROVED

  set.seed(42)
  n <- 2
  matrix.size <- n * (n + 1) / 2
  sigma <- diag(n) + 0.1 * matrix(runif(n * n), n, n)
  sigma <- 0.5 * (sigma + t(sigma))
  mu <- runif(n)

  n.samples <- 10e4
  x <- rmvnorm(n=n.samples, mean=mu, sigma=sigma)
  GetProductTerms <- function(x.row) {
    return(ConvertSymmetricMatrixToVector(outer(x.row, x.row)))
  }
  GetProductTerms(x[1, ])
  x.prods <- apply(x, MARGIN=1, GetProductTerms)
  x.all <- cbind(x, t(x.prods))
  expected.x.cov <- cov(x.all)

  mean(abs(expected.x.cov[1:n, 1:n] - sigma))

  mu.mat <- matrix(mu, ncol=1)
  mu2.mat <- matrix(ConvertSymmetricMatrixToVector(sigma + mu %*% t(mu)), ncol=1)
  mu.cov <- GetMuVariance(mu.mat, mu2.mat)

  summary(as.numeric(abs(expected.x.cov[1:n, 1:n] - mu.cov[1:n, 1:n])))
  summary(as.numeric(abs(expected.x.cov[1:n, -(1:n)] - mu.cov[1:n, -(1:n)])))
  summary(as.numeric(abs(expected.x.cov[-(1:n), -(1:n)] - mu.cov[-(1:n), -(1:n)])))

  checkTrue(max(abs(expected.x.cov - mu.cov)) < 0.2)
  checkTrue(mean(abs(expected.x.cov - mu.cov)) < 0.03)

  mu.cov <- GetMuVariance(cbind(mu, mu), cbind(mu2.mat, mu2.mat))
  checkEqualsNumeric(mu.cov[1:n, 1:n], mu.cov[n + 1:n, n + 1:n])
  checkEqualsNumeric(mu.cov[2 * n  + 1:matrix.size, 2 * n  + 1:matrix.size],
                     mu.cov[2 * n + matrix.size + 1:matrix.size,
                            2 * n + matrix.size + 1:matrix.size])
}


TestGetWishartLinearCovariance <- function() {
  set.seed(42)
  n <- 5
  v.par <- diag(n) + 0.1 * matrix(runif(n * n), n, n)
  v.par <- v.par + t(v.par)

  v.vec <- ConvertSymmetricMatrixToVector(v.par)
  sigma.par <- solve(v.par)
  n.par <- 10

  # The theoretical covariances
  expected.cov.mat <- matrix(0, length(v.vec), length(v.vec))
  for (j1 in 1:n) {
    for (i1 in 1:j1) {
      for (j2 in 1:n) {
        for (i2 in 1:j2) {
          index1 <- GetUpperTriangularIndex(i1 - 1, j1 - 1) + 1
          index2 <- GetUpperTriangularIndex(i2 - 1, j2 - 1) + 1
          expected.cov.mat[index1, index2] <-
            n.par * (v.par[i2, j1] * v.par[i1, j2] + v.par[i2, i1] * v.par[j1, j2])
          expected.cov.mat[index2, index1] <- expected.cov.mat[index1, index2]
        }
      }
    }
  }
  cov.mat <- GetWishartLinearCovariance(v.vec, n.par)
  checkEqualsNumeric(expected.cov.mat, cov.mat)

  det.linear.cov <- GetWishartLinearLogDetCovariance(v.vec)
  checkEqualsNumeric(2 * v.vec, det.linear.cov)

  det.var <- GetWishartLogDetVariance(n.par, n)
  checkEqualsNumeric(det.var, MVTrigamma(n.par / 2, n))

  full.cov <- GetLambdaVariance(cbind(v.vec, v.vec), c(n.par, n.par))

  half.indices <- 1:length(v.vec)
  full.cov[half.indices, half.indices] /  cov.mat

  checkEqualsNumeric(full.cov, t(full.cov))
  checkEqualsNumeric(full.cov[half.indices, half.indices], cov.mat)
  checkEqualsNumeric(full.cov[half.indices, half.indices],
                     full.cov[half.indices + length(v.vec), half.indices + length(v.vec)])
  checkEqualsNumeric(full.cov[2 * length(v.vec) + 1, half.indices],
                     full.cov[2 * length(v.vec) + 2, half.indices + length(v.vec)])

  checkEqualsNumeric(full.cov[2 * length(v.vec) + 1, half.indices], det.linear.cov)
  checkEqualsNumeric(full.cov[2 * length(v.vec) + 1, 2 * length(v.vec) + 1], det.var)
  checkEqualsNumeric(full.cov, t(full.cov))
}

TestMVNLogLikelihoodAndGetZMat <-function() {
  set.seed(42)
  n <- 10
  p  <- 5
  k <- 3

  e.lambda <- diag(p) + 0.1 * matrix(runif(p * p), p, p)
  e.lambda <- e.lambda + t(e.lambda)
  sigma <- solve(e.lambda)
  e.mu <- runif(p)

  # The following assignments will make the variational log likelihood idential
  # to the ordinary log likelihood (up to a constant)
  e.mu2 <- e.mu %*% t(e.mu)
  e.log.det.lambda <- log(det(e.lambda))
  e.log.pi <- 0

  num.draws <- 30
  results <- matrix(NA, num.draws, 2)
  x <- rmvnorm(num.draws, mean=e.mu, sigma=sigma)
  for (draw in 1:num.draws) {
    expected.log.lik <- dmvnorm(x=x[draw, ], mean=e.mu, sigma=sigma, log = TRUE)
    cpp.log.lik <- MVNLogLikelihoodPoint(x[draw, ], e.mu,
                                         ConvertSymmetricMatrixToVector(e.mu2),
                                         ConvertSymmetricMatrixToVector(e.lambda),
                                         e.log.det.lambda, e.log.pi)
    results[draw, 1] <- expected.log.lik
    results[draw, 2]  <- cpp.log.lik
  }
  # Check that the difference is a constant.
  checkEqualsNumeric(sd(results[, 1] - results[, 2]), 0)

  # Also test GetZMatrix while we're here.
  e.mu.mat <- cbind(e.mu, e.mu)
  e.mu2.vec <- ConvertSymmetricMatrixToVector(e.mu2)
  e.mu2.mat <- cbind(e.mu2.vec, e.mu2.vec)
  e.lambda.vec <- ConvertSymmetricMatrixToVector(e.lambda)
  e.lambda.mat <- cbind(e.lambda.vec, e.lambda.vec)

  # Check with each component equally likely
  z.mat <- GetZMatrix(x, e.mu.mat, e.mu2.mat, e.lambda.mat, c(0, 0), c(1, 1))
  checkEqualsNumeric(rowSums(z.mat), rep(1, num.draws))
  checkEqualsNumeric(z.mat, rep(0.5, 2 * num.draws))

  # Check with a log pi change.
  log.pi.offset <- 0.5
  z.mat <- GetZMatrix(x, e.mu.mat, e.mu2.mat, e.lambda.mat, c(0, 0), c(0, log.pi.offset))
  checkEqualsNumeric(rowSums(z.mat), rep(1, num.draws))
  checkEqualsNumeric(z.mat[, 1]  / z.mat[, 2], rep(exp(-log.pi.offset), num.draws))

  # Check with a variance change.
  variance.offset <- 2
  new.e.lambda <- e.lambda
  new.e.lambda <- variance.offset * e.lambda
  new.e.lambda.mat <- cbind(e.lambda.vec, ConvertSymmetricMatrixToVector(new.e.lambda))
  e.log.det.lambda.vec <- c(log(det(e.lambda)), log(det(new.e.lambda)))
  z.mat <- GetZMatrix(x, e.mu.mat, e.mu2.mat, new.e.lambda.mat,
                      e.log.det.lambda.vec, c(0, 0))
  checkEqualsNumeric(rowSums(z.mat), rep(1, num.draws))

  # Check with an obvious assignment.
  k <- 2
  par <- GenerateSampleParams(k=k, p=2, vars.scale=0.001^2)
  data <- GenerateMultivariateData(50, par$true.means, par$true.sigma, par$true.probs)
  e.lambda <- InvertLinearizedMatrices(VectorizeMatrixList(par$true.sigma))
  e.mu2 <- GetVectorizedOuterProductMatrix(par$true.means)
  z.mat <- GetZMatrix(data$x,
                      e_mu=par$true.means,
                      e_mu2=e.mu2,
                      e_lambda=e.lambda,
                      e_log_det_lambda=rep(0, k),
                      e_log_pi=rep(0, k))
  checkEqualsNumeric(z.mat, data$components)
}


TestGetZCovariance <- function() {
  n <- 10
  k <- 3
  z.mat <- matrix(runif(n * k), n, k)
  z.mat <- z.mat / rep(rowSums(z.mat), k)

  checkEqualsNumeric(diag(z.mat[1, ]) - outer(z.mat[1, ], z.mat[1, ]),
                     GetSingleZCovariance(z.mat[1,]))
  z.cov <- GetZCovariance(z.mat)
  for (this.n in 1:n) {
    this.z <- z.mat[this.n, ]
    this.z.cov <- GetSingleZCovariance(this.z)
    offset <- (this.n - 1) * k + 1
    this.range <- offset:(offset + k - 1)
    checkEqualsNumeric(this.z.cov, z.cov[this.range, this.range])
  }
}



TestInvertLinearizedMatrices <- function() {
  k <- 3
  p <- 5
  x.list <- list()
  x.inv.list <- list()
  x.mat <- matrix(NA, p * (1 + p) / 2, k)
  for (this.k in 1:k) {
    x <- diag(p) + 0.2 * matrix(runif(p * p), p, p)
    x <- t(x) + x
    x.list[[this.k]] <- x
    x.inv.list[[this.k]] <- solve(x)
    x.mat[, this.k] <- ConvertSymmetricMatrixToVector(x)
  }

  x.mat.inv <- InvertLinearizedMatrices(x.mat)
  for (this.k in 1:k) {
    checkEqualsNumeric(ConvertSymmetricMatrixToVector(x.inv.list[[this.k]]),
                       x.mat.inv[, this.k])
  }

  # Also test LogDeterminantOfLinearlizeMatrices()
  x.mat.det <- LogDeterminantOfLinearizedMatrices(x.mat)
  for (this.k in 1:k) {
    checkEqualsNumeric(log(det(x.list[[this.k]])), x.mat.det[this.k])
  }
}

TestGetVectorizedOuterProductMatrix <- function() {
  k <- 4
  p <- 10
  x <- matrix(runif(k * p), p, k)

  test.result <- matrix(NA, nrow=p * (p + 1) / 2, ncol=k)
  for (this.k in 1:k) {
    this.vec <- x[, this.k]
    test.result[, this.k] <- ConvertSymmetricMatrixToVector(this.vec %*% t(this.vec))
  }
  result <- GetVectorizedOuterProductMatrix(x)
  checkEquals(test.result, result)
}

TestUpdateFunctions <- function() {
  n <- 10
  k <- 2
  p <- 3
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

  # Lambda update
  empty.prior <- matrix(0, 1, 1)
  lambda.update <- UpdateLambdaPosterior(x=x, e_mu=e.mu, e_mu2=e.mu2, e_z=z,
                                         FALSE, empty.prior, 0)
  lambda.par <- lambda.update$lambda_par
  n.par <- lambda.update$n_par
  e.log.det.lambda <- WishartELogDet(lambda.par, n.par)
  e.lambda <- rep(n.par, each=p * (p + 1) / 2) * lambda.par
  e.lambda.inv.mat <- InvertLinearizedMatrices(e.lambda)

  # Pi update
  e.log.pi <- GetELogDirichlet(colSums(z))

  for (this.k in 1:k) {
    this.mu <- matrix(rep(e.mu[, this.k], each=n), n, p)
    this.z <- z[, this.k]
    v.inv <- (t(this.z * x) %*% x - t(this.z * x) %*% this.mu - t(this.mu) %*% (this.z * x) +
                sum(this.z) * ConvertVectorToSymmetricMatrix(e.mu2[, this.k]))
    v <- solve(v.inv)
    checkEqualsNumeric(ConvertSymmetricMatrixToVector(v), lambda.par[, this.k])

    # TODO: this is wrong!  E(log(det(V))) != log(det(E(V))),
    checkEquals(log(det(v)) + p * log(2) + MVDigamma(n.par[[this.k]] / 2, p),
                e.log.det.lambda[this.k])
  }

  # Z update
  GetZMatrixInPlace(z=z,
                    x=x, e_mu=e.mu, e_mu2=e.mu2,
                    e_lambda=e.lambda, e_log_det_lambda=e.log.det.lambda,
                    e_log_pi=e.log.pi)

  # Mu update
  mu.update <- UpdateMuPosterior(x=x, e_lambda_inv_mat=e.lambda.inv.mat, e_z=z)
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
}


TestMLEPacking <- function() {
  k <- 5
  p <- 7
  par <- GenerateSampleParams(k=k, p=p)
  sigma.mat <- VectorizeMatrixList(par$true.sigma)
  lambda.mat <- InvertLinearizedMatrices(sigma.mat)
  linear.par <- LinearlyPackParameters(mu=par$true.means, lambda=lambda.mat, pi=par$true.probs)
  unpacked.par <- LinearlyUnpackParameters(linear.par, k_tot=k, p_tot=p)

  checkEqualsNumeric(par$true.means, unpacked.par$mu)
  checkEqualsNumeric(lambda.mat, unpacked.par$lambda)
  checkEqualsNumeric(par$true.probs, unpacked.par$pi)
}


TestMarginalLogLikelihood <- function() {
  set.seed(42)
  n <- 20
  k <- 2
  p <- 2
  par <- GenerateSampleParams(k=k, p=p, vars.scale=0.4^2)
  lambda.par <- InvertLinearizedMatrices(VectorizeMatrixList(par$true.sigma))

  samples <- 40
  r.log.lik <- rep(NA, samples)
  cpp.log.lik <- rep(NA, samples)
  for (sample in 1:samples) {
    data <- GenerateMultivariateData(n, par$true.means, par$true.sigma, par$true.probs)
    cpp.log.lik[sample] <- MarginalLogLikelihood(data$x,
                                                 par$true.means,
                                                 lambda.par,
                                                 par$true.probs)
    total.lik <- rep(0, n)
    for (this.k in 1:k) {
      this.lik <- dmvnorm(data$x, par$true.means[, this.k], par$true.sigma[[this.k]], log=FALSE)
      total.lik  <- total.lik + par$true.probs[this.k] * this.lik
    }
    r.log.lik[sample] <- sum(log(total.lik))
  }

  # They differ only by a constant.
  checkEqualsNumeric(sd(r.log.lik - cpp.log.lik), 0)
}


TestSensitivities <- function() {
  # Compare the analytic sensitivitites with numerical second derivatives.
  set.seed(42)
  n <- 10
  k <- 2
  p <- 2
  par <- GenerateSampleParams(k=k, p=p, vars.scale=0.4^2)
  lambda.par <- InvertLinearizedMatrices(VectorizeMatrixList(par$true.sigma))
  data <- GenerateMultivariateData(n, par$true.means, par$true.sigma, par$true.probs)
  priors <- GenerateSamplePriors(x=data$x, k=k, lambda.scale=0.01)

  e.mu <- par$true.means
  e.mu2 <- GetVectorizedOuterProductMatrix(e.mu)
  e.lambda <- lambda.par
  e.log.det.lambda <- LogDeterminantOfLinearizedMatrices(e.lambda)
  e.log.pi <- log(par$true.probs)
  e.z <- GetZMatrix(data$x, e.mu, e.mu2, e.lambda, e.log.det.lambda, e.log.pi)

  # Check the packing and unpacking.
  par <- PackVBParameters(e.z, e.mu, e.mu2, e.lambda, e.log.det.lambda, e.log.pi)
  unpacked.par <- UnpackVBParameters(par, n, p, k)
  checkEquals(e.mu, unpacked.par$e_mu)
  checkEquals(e.mu2, unpacked.par$e_mu2)
  checkEquals(e.lambda, unpacked.par$e_lambda)
  checkEquals(e.log.det.lambda, unpacked.par$e_log_det_lambda)
  checkEquals(e.log.pi, unpacked.par$e_log_pi)
  checkEquals(e.z, unpacked.par$e_z)

  LogLikelihoodForHessian <- function(par) {
    unpacked.par <- UnpackVBParameters(par, n, p, k)
    log.lik <- CompleteLogLikelihoodWithPriors(data$x,
                                               unpacked.par$e_z,
                                               unpacked.par$e_mu,
                                               unpacked.par$e_mu2,
                                               unpacked.par$e_lambda,
                                               unpacked.par$e_log_det_lambda,
                                               unpacked.par$e_log_pi,
                                               FALSE, FALSE, FALSE,
                                               priors$mu.prior.mean,
                                               priors$mu.prior.info,
                                               priors$lambda.prior.v.inv,
                                               priors$lambda.prior.n,
                                               priors$p.prior.alpha,
                                               FALSE)
  }

  htz <- GetHThetaZ(data$x, e.mu, e.mu2, e.lambda)
  htt <- GetHThetaTheta(data$x, e.z)
  checkEquals(htt, t(htt))

  theta.size <- nrow(htt)
  z.size <- ncol(htz)

  hess <- hessian(LogLikelihoodForHessian, par)
  num.htt <- hess[1:theta.size, 1:theta.size]
  num.htz <- hess[1:theta.size, theta.size + 1:z.size]

  checkEqualsNumeric(num.htt, as.matrix(htt), tolerance=1e-4)
  checkEqualsNumeric(num.htz, as.matrix(htz), tolerance=1e-4)
}


TestXPackingParameters <- function() {
  n <- 10
  p <- 2
  x <- matrix(runif(n * p), nrow=n, ncol=p)
  x2 <- t(apply(x, MARGIN=1,
                function(x.row) {
                  ConvertSymmetricMatrixToVector(x.row %*% t(x.row))
                }))
  x.par <- PackXParameters(x)
  unpacked.par <- UnpackXParameters(x.par, n, p)
  checkEqualsNumeric(x, unpacked.par$x)
  checkEqualsNumeric(x2, unpacked.par$x2)
}

TestSensitivitiesToX <- function() {
  # Compare the analytic sensitivitites with numerical second derivatives.

  # This currently doesn't work because the log likelihood doesn't separate
  # and and x^2 dependence.
  set.seed(42)
  n <- 3
  k <- 2
  p <- 2
  matrix.size <- p * (p + 1) / 2
  par <- GenerateSampleParams(k=k, p=p, vars.scale=0.4^2)
  # zero means make some sensitivities zero.
  par$true.means <- par$true.means + 1

  lambda.par <- InvertLinearizedMatrices(VectorizeMatrixList(par$true.sigma))
  data <- GenerateMultivariateData(n, par$true.means, par$true.sigma, par$true.probs)
  priors <- GenerateSamplePriors(x=data$x, k=k, lambda.scale=0.01)

  e.mu <- par$true.means
  e.mu2 <- GetVectorizedOuterProductMatrix(e.mu)
  e.lambda <- lambda.par
  e.log.det.lambda <- LogDeterminantOfLinearizedMatrices(e.lambda)
  e.log.pi <- log(par$true.probs)
  e.z <- GetZMatrix(data$x, e.mu, e.mu2, e.lambda, e.log.det.lambda, e.log.pi)

  # Check the packing and unpacking.
  x.par <- PackXParameters(data$x)
  unpacked.par <- UnpackXParameters(x.par, n, p)
  checkEqualsNumeric(data$x, unpacked.par$x)

  vb.par <- PackVBParameters(e.z, e.mu, e.mu2, e.lambda, e.log.det.lambda, e.log.pi)
  vb.size <- length(vb.par)
  vb.indices <- 1:vb.size
  x.size <- length(x.par)
  x.indices <- vb.size + 1:x.size

  OriginalLogLikelihoodForHessian <- function(par) {
    vb.par <- par[vb.indices]
    x.par <- par[x.indices]
    unpacked.par <- UnpackVBParameters(vb.par, n, p, k)
    unpacked.x <- UnpackXParameters(x.par, n, p)
    log.lik <- CompleteLogLikelihoodWithPriors(unpacked.x$x,
                                               unpacked.par$e_z,
                                               unpacked.par$e_mu,
                                               unpacked.par$e_mu2,
                                               unpacked.par$e_lambda,
                                               unpacked.par$e_log_det_lambda,
                                               unpacked.par$e_log_pi,
                                               FALSE, FALSE, FALSE,
                                               priors$mu.prior.mean,
                                               priors$mu.prior.info,
                                               priors$lambda.prior.v.inv,
                                               priors$lambda.prior.n,
                                               priors$p.prior.alpha,
                                               FALSE)
    return(log.lik)
  }

  LogLikelihoodForHessian <- function(par) {
    vb.par <- par[vb.indices]
    x.par <- par[x.indices]
    unpacked.par <- UnpackVBParameters(vb.par, n, p, k)
    unpacked.x <- UnpackXParameters(x.par, n, p)
    log.lik <- CompleteLogLikelihoodWithX2(unpacked.x$x,
                                               unpacked.x$x2,
                                               unpacked.par$e_z,
                                               unpacked.par$e_mu,
                                               unpacked.par$e_mu2,
                                               unpacked.par$e_lambda,
                                               unpacked.par$e_log_det_lambda,
                                               unpacked.par$e_log_pi,
                                               FALSE)
    return(log.lik)
  }

  checkEqualsNumeric(OriginalLogLikelihoodForHessian(c(vb.par, x.par)),
                     LogLikelihoodForHessian(c(vb.par, x.par)))

  hess <- hessian(LogLikelihoodForHessian, c(vb.par, x.par))

  htx <- GetHThetaX(e.z, e.mu, e.lambda)
  hzx <- GetHZX(n, e.mu, e.lambda)

  htz <- GetHThetaZ(data$x, e.mu, e.mu2, e.lambda)
  htt <- GetHThetaTheta(data$x, e.z)


  theta.size <- nrow(htt)
  z.size <- ncol(htz)
  x.indices <- 1:(n * p)

  num.htx <- hess[1:theta.size, vb.size + 1:x.size]
  num.hzx <- hess[theta.size + 1:z.size, vb.size + 1:x.size]

  # Just sanity check that the htz and htt matrices are still correct.  This is
  # also tested elsewhere.
  checkEqualsNumeric(hess[1:theta.size, 1:theta.size], as.matrix(htt), tolerance=1e-4)
  checkEqualsNumeric(hess[1:theta.size, theta.size + 1:z.size], as.matrix(htz), tolerance=1e-4)

  num.htx[abs(num.htx) < 1e-10] <- 0
  num.hzx[abs(num.hzx) < 1e-10] <- 0

  checkEqualsNumeric(num.hzx, as.matrix(hzx), tolerance=1e-6)
  checkEqualsNumeric(num.htx, as.matrix(htx), tolerance=1e-4)

  # Test the subset functions.  The indices are zero-based.
  htx.sub <- GetHThetaXSubset(e.z, e.mu, e.lambda, c(1, 3) - 1)
  hzx.sub <- GetHZXSubset(n, e.mu, e.lambda, c(1, 3) - 1)

  # Indices of the removed components.
  x2.indices <- c(p + 1:p, n * p + matrix.size + 1:matrix.size)
  checkEqualsNumeric(htx[, -x2.indices], htx.sub)
  checkEqualsNumeric(hzx[, -x2.indices], hzx.sub)

}

TestLRVBFunctions <- function() {
  set.seed(42)
  n <- 1000
  k <- 2
  p <- 2
  par <- GenerateSampleParams(k=k, p=p, vars.scale=0.4^2)
  lambda.par <- InvertLinearizedMatrices(VectorizeMatrixList(par$true.sigma))
  data <- GenerateMultivariateData(n, par$true.means, par$true.sigma, par$true.probs)
  priors <- GenerateSamplePriors(x=data$x, k=k, lambda.scale=0.01)

  e.mu <- par$true.means
  e.mu2 <- GetVectorizedOuterProductMatrix(e.mu)
  e.lambda <- lambda.par
  e.log.det.lambda <- LogDeterminantOfLinearizedMatrices(e.lambda)
  e.log.pi <- log(par$true.probs)
  e.z <- GetZMatrix(data$x, e.mu, e.mu2, e.lambda, e.log.det.lambda, e.log.pi)

  vb.optimum <-
    GetVariationalSolution(x=data$x,
                           e.mu=par$true.means,
                           e.lambda=true.lambda,
                           e.log.det.lambda=true.log.det.lambda,
                           e.p=par$true.probs,
                           e.log.pi=GetELogDirichlet(par$true.probs * n),
                           fit.mu=TRUE, fit.lambda=TRUE, fit.pi=TRUE,
                           e.z=data$components,
                           priors=priors, tolerance=1e-9, quiet=TRUE)

  theta.cov <- GetThetaCovariance(e.mu=vb.optimum$e.mu,
                                  e.mu2=vb.optimum$e.mu2,
                                  lambda.par=vb.optimum$lambda.par,
                                  n.par=vb.optimum$n.par,
                                  e.z=vb.optimum$e.z,
                                  fit.mu=TRUE, fit.lambda=TRUE, fit.pi=TRUE)

  # Make sure all the different ways of getting the LRVB correction are the same.
  lrvb.theta.cov <- CPPGetLRVBCovariance(x=data$x, e_mu=vb.optimum$e.mu,
                                         e_mu2=vb.optimum$e.mu2, e_lambda=vb.optimum$e.lambda,
                                         e_z=vb.optimum$e.z,
                                         theta_cov=theta.cov, verbose=FALSE)
  lrvb.correction <- GetLRVBCorrectionTerm(x=data$x, e_mu=vb.optimum$e.mu,
                                           e_mu2=vb.optimum$e.mu2, e_lambda=vb.optimum$e.lambda,
                                           e_z=vb.optimum$e.z,
                                           theta_cov=theta.cov, verbose=FALSE)
  lrvb.theta.cov.v2 <- CPPGetLRVBCovarianceFromCorrection(lrvb_correction=lrvb.correction,
                                                          theta_cov=theta.cov,
                                                          verbose=FALSE)
  lrvb.theta.cov.v3 <- GetLRVBCovariance(x=data$x,
                                         e.mu=vb.optimum$e.mu,
                                         e.mu2=vb.optimum$e.mu2,
                                         e.lambda=vb.optimum$e.lambda,
                                         e.z=vb.optimum$e.z,
                                         theta.cov=theta.cov)
  checkEqualsNumeric(lrvb.theta.cov, lrvb.theta.cov.v2)
  checkEqualsNumeric(lrvb.theta.cov, lrvb.theta.cov.v3)

  # Make sure the different ways of getting the leverage scores are the same.
  x.cov <- GetXVariance(x=data$x)
  htx <- GetHThetaX(e_z=vb.optimum$e.z, e_mu=vb.optimum$e.mu,
                    e_lambda=vb.optimum$e.lambda)
  hzx <- GetHZX(n_tot=n, e_mu=vb.optimum$e.mu, e_lambda=vb.optimum$e.lambda)
  htz <- GetHThetaZ(x=data$x, e_mu=vb.optimum$e.mu,
                    e_mu2=vb.optimum$e.mu2,
                    e_lambda=vb.optimum$e.lambda)
  z.cov <- GetZCovariance(z_mat=vb.optimum$e.z)
  tx.cov <- CPPGetLeverageScores(z_cov=z.cov, x_cov=x.cov, htx=htx,htz=htz, hzx=hzx,
                                 lrvb_correction=lrvb.correction, theta_cov=theta.cov)
  tx.cov.v2 <- GetLeverageScores(x=data$x, e.mu=vb.optimum$e.mu, e.mu2=vb.optimum$e.mu2,
                                 e.lambda=vb.optimum$e.lambda, e.z=vb.optimum$e.z,
                                 lrvb.theta.cov=lrvb.theta.cov)
  checkEqualsNumeric(tx.cov, tx.cov.v2)
}



TestGammaFunctions()
TestConvertSymmetricMatrixToVector()
TestConvertVectorToSymmetricMatrix()
TestGetWishartLinearCovariance()
TestMultivariateNormalCovariance()
TestMVNLogLikelihoodAndGetZMat()
TestGetZCovariance()
TestInvertLinearizedMatrices()
TestGetVectorizedOuterProductMatrix()
TestUpdateFunctions()
TestMLEPacking()
TestMarginalLogLikelihood()
TestSensitivities()
TestXPackingParameters()
TestXVariance()
TestXVarianceSubset()
TestSensitivitiesToX()
TestLRVBFunctions()
