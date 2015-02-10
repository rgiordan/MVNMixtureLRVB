library(Matrix)
library(mvtnorm)

MVLgamma <- function(x, p) {
  return(sum(lgamma(x + (1 - 1:p) / 2)))
}

MVDigamma <- function(x, p) {
  return(sum(digamma(x + (1 - 1:p) / 2)))
}

MVTrigamma <- function(x, p) {
  return(sum(trigamma(x + (1 - 1:p) / 2)))
}

VectorizeMatrixList <- function(mat.list) {
  k <- length(mat.list)
  p  <- nrow(mat.list[[1]])
  result.mat <- matrix(NA, (p * (1 + p)) / 2, k)

  for (this.k in 1:k) {
    this.mat <- mat.list[[this.k]]
    stopifnot(nrow(this.mat) == ncol(this.mat))
    stopifnot(nrow(this.mat) == p)
    result.mat[, this.k] <- ConvertSymmetricMatrixToVector(this.mat)
  }
  return(result.mat)
}

ListifyVectorizedMatrix <- function(mat.vec) {
  k <- ncol(mat.vec)
  mat.list <- list()
  for (this.k in 1:k) {
    mat.list[[this.k]] <- ConvertVectorToSymmetricMatrix(mat.vec[, this.k])
  }
  return(mat.list)
}

GetSymmetricMatrixVectorNames <- function(parameter, p, k=1, sep="_") {
  row.mat <- matrix(rep(as.numeric(1:p)), p, p)
  col.mat <- matrix(rep(as.numeric(1:p), each=p), p, p)
  row.vec <- ConvertSymmetricMatrixToVector(row.mat)
  col.vec <- ConvertSymmetricMatrixToVector(col.mat)
  indices <- paste(row.vec, col.vec, sep=sep)
  paramter.k <- parameter
  if (k > 1) {
    parameter.k <- paste(parameter, 1:k, sep=sep)
  }
  return(as.character(outer(indices, parameter.k, function(x, y) { paste(y, x, sep=sep) })))
}


GetMatrixVectorNames <- function(parameter, p, k=1, sep="_") {
  indices <- 1:p
  parameter.k <- parameter
  if (k > 1) {
    parameter.k <- paste(parameter, 1:k, sep=sep)
  }
  return(as.character(outer(indices, parameter.k, function(x, y) { paste(y, x, sep=sep) })))
}

GenerateSampleParams <- function(k, p, vars.scale = 0.4, anisotropy=0, random.rotation=FALSE) {
  true.means <- matrix(rep(0:(k-1), each=p), nrow=p, ncol=k)
  true.sigma <- list()
  for (this.k in 1:k) {
    this.sigma <- diag(1 + anisotropy * (1 : p - 1) / p)
    this.sigma <- t(this.sigma) %*% this.sigma
    if (random.rotation) {
      # Random rotation
      rotation <- matrix(runif(p * p), p, p)
      rotation <- rotation %*% t(rotation)
      rotation <- eigen(rotation)$vectors
      this.sigma <- rotation %*% this.sigma %*% t(rotation)
    }
    max.lambda <- max(eigen(this.sigma)$value)
    this.sigma <- this.sigma * sqrt(this.k) * vars.scale^2 / max.lambda
    true.sigma[[this.k]] <- this.sigma
  }
  true.probs <- k + 1:k
  true.probs <- true.probs / sum(true.probs)
  return(list(true.probs=true.probs, true.means=true.means, true.sigma=true.sigma))
}


GenerateSamplePriors <- function(x, k, lambda.scale=1) {
  p <- ncol(x)
  n <- nrow(x)
  matrix.size <- (p * (p + 1)) / 2
  mu.prior.mean  <- matrix(rep(0, p * k), nrow=p, ncol=k)
  x.scale <- diff(range(x))
  mu.prior.info <- matrix(3 / (x.scale ^ 2), matrix.size, k)

  lambda.prior.n <- rep(p, k) / 1000
  lambda.prior.v.inv.list <- list()
  for (this.k in 1:k) {
    lambda.prior.v.inv.list[[this.k]] <- lambda.scale * (x.scale ^ 2) * diag(p) / (1000 * p)
  }
  lambda.prior.v.inv <- VectorizeMatrixList(lambda.prior.v.inv.list)

  p.prior.alpha <- rep(1, k)
  return(list(mu.prior.mean=mu.prior.mean, mu.prior.info=mu.prior.info,
              lambda.prior.v.inv=lambda.prior.v.inv, lambda.prior.n=lambda.prior.n,
              p.prior.alpha=p.prior.alpha))
}



GenerateMultivariateData <- function(n, true.means, true.sigma, true.probs) {
  # Args:
  #   - n: The number of data points to simulate
  #   - true.means: A p by k matrix of the true means.
  #   - true.vars: A list of the true variances, each of which is p by p
  #   - true.probs: A p-length vector containing the true probabilities,
  #     which sum to one.
  #
  # Returns:
  #   A list containing
  #   - x: An n by p matrix containing draws from the mixture.
  #   - components: An n by k matrix containing the true components of x.
  #   - component.labels: An n-length vector of the component labels.

  p <- nrow(true.means)
  k <- length(true.probs)
  stopifnot(k == length(true.sigma))
  x <- matrix(NA, n, p)

  components <- t(rmultinom(n, prob=true.probs, size=1))
  component.n <- colSums(components)
  if (any(component.n == 0)) {
    print("Warning -- one component has no data points.")
  }

  for (this.k in 1:k) {
    stopifnot(ncol(true.sigma[[this.k]]) == p)
    stopifnot(nrow(true.sigma[[this.k]]) == p)
    this.x <- rmvnorm(component.n[this.k],
                      mean=true.means[, this.k],
                      sigma=true.sigma[[this.k]])
    x[components[, this.k] == 1, ] <- this.x
  }
  if (k == 1) {
    component.labels <- components
  } else {
    component.labels <- 1 + (k - colSums(apply(components, 1, cumsum)))
  }

  # Convert the components matrix to numeric for C++
  components <- matrix(as.numeric(components), nrow(components), ncol(components))
  return(list(x=x, components=components, component.labels=component.labels))
}

MakeSigmaMatrix <- function(sigma.list) {
  # Args:
  # - sigma.list: A list of p by p covariance matrices.
  #
  # Returns:
  #   A p * (p + 1) / 2 by k matrix of the vectorized matrices.
  result <- do.call(cbind, lapply(sigma.list, ConvertSymmetricMatrixToVector))
}


GetVariationalSolution <- function(x, e.mu, e.mu2=NULL,
                                   e.lambda=NULL, e.log.det.lambda=NULL,
                                   e.log.pi=NULL, e.pi=NULL, e.z,
                                   fit.mu=TRUE, fit.lambda=TRUE, fit.pi=TRUE, priors,
                                   tolerance=1e-5, max.iter=1000, elbo.every.n=Inf,
                                   debug=FALSE, keep.updates=FALSE, quiet=FALSE) {
  # priors should be a list like that returned by GenerateSamplePriors

  n <- nrow(x)
  p <- ncol(x)
  k <- ncol(e.mu)

  stopifnot(nrow(e.mu) == p)
  stopifnot(nrow(e.z) == n)
  stopifnot(ncol(e.z) == k)

  # Make sure everything is a numeric matrix.
  e.mu <- matrix(as.numeric(e.mu), nrow(e.mu), ncol(e.mu))
  e.z <- matrix(as.numeric(e.z), nrow(e.z), ncol(e.z))
  x <- as.matrix(x)

  # If not specified, fit with zero additional mu variance.
  if (is.null(e.mu2)) {
    e.mu2 <- GetVectorizedOuterProductMatrix(e.mu)
  }

  if (!fit.lambda) {
    if (is.null(e.lambda)) {
      stop("You must specify e.lambda if you do not fit lambda.")
    }
    if (is.null(e.log.det.lambda)) {
      stop("You must specify e.log.det.lambda if you do not fit lambda.")
    }
    e.lambda.inv.mat <- InvertLinearizedMatrices(e.lambda)

    # These values don't matter if we're not fitting lambda.
    n.par <- rep(10, k)
    lambda.par <- e.lambda / 10
  } else {
    # These will hold the lambda Wishart parameters.
    lambda.par <- matrix(0, p * (p + 1) / 2, k)
    n.par <- rep(0, k)
  }

  if (!fit.pi) {
    if (is.null(e.log.pi)) {
      stop("You must specify e.log.pi if you do not fit pi")
    }
    if (is.null(e.pi)) {
      stop("You must specify e.pi if you do not fit pi")
    }
  } else {
    if (is.null(e.log.pi)) {
      # Initialize the pi paraemeters.
      e.log.pi <- GetELogDirichlet(colSums(e.z))
    }
    if (is.null(e.pi)) {
      e.pi <- colMeans(e.z)
    }
  }

  updates <- list()
  total.diff <- Inf
  iter <- 0
  elbo.list <- list()
  log.lik.list <- list()
  entropy.list <- list()
  mu.diff <- lambda.diff <- pi.diff <- 0
  while (iter <= max.iter && total.diff > tolerance) {
    if (debug) browser()

    # Add zero to force R to allocate new memory.
    iter <- iter + 1

    old.e.mu <- e.mu + 1
    old.e.mu <- old.e.mu - 1
    old.e.mu2 <- e.mu2 + 1
    old.e.mu2 <- old.e.mu2 - 1

    old.lambda.par <- lambda.par + 1
    old.lambda.par <- old.lambda.par - 1

    old.n.par <- n.par + 1
    old.n.par <- old.n.par - 1

    old.e.log.pi <- e.log.pi + 1
    old.e.log.pi <- old.e.log.pi - 1

    # Lambda update
    if (fit.lambda) {
      lambda.update <- UpdateLambdaPosterior(x=x, e_mu=e.mu, e_mu2=e.mu2, e_z=e.z,
                                             use_prior=FALSE,
                                             lambda_prior_v_inv=priors$lambda.prior.v.inv,
                                             lambda_prior_n=priors$lambda.prior.n)
      lambda.par <- lambda.update$lambda_par
      n.par <- lambda.update$n_par
      if (!is.finite(lambda.par) || !is.finite(n.par)) {
        browser()
      }
      e.log.det.lambda <- WishartELogDet(lambda.par, n.par)
      e.lambda <- rep(n.par, each=p * (p + 1) / 2) * lambda.par
      e.lambda.inv.mat <- InvertLinearizedMatrices(e.lambda)
      lambda.diff <- sum(abs(lambda.par - old.lambda.par)) + sum(abs(n.par - old.n.par))
    }

    # Pi update
    if (fit.pi) {
      e.log.pi <- GetELogDirichlet(colSums(e.z))
      e.pi <- colMeans(e.z)
      pi.diff <- sum(abs(e.log.pi - old.e.log.pi))
    }

    # Mu update
    if (fit.mu) {
      mu.update <- UpdateMuPosterior(x=x, e_lambda_inv_mat=e.lambda.inv.mat, e_z=e.z)
      e.mu <- mu.update$e_mu
      e.mu2 <-  mu.update$e_mu2
      if (!is.finite(e.mu) || !is.finite(e.mu2)) {
        browser()
      }
      mu.diff <- sum(abs(e.mu - old.e.mu)) + sum(abs(e.mu2 - old.e.mu2))
    }

    # Z update
    GetZMatrixInPlace(z=e.z,
                      x=x, e_mu=e.mu, e_mu2=e.mu2,
                      e_lambda=e.lambda, e_log_det_lambda=e.log.det.lambda,
                      e_log_pi=e.log.pi)

    total.diff <- mu.diff + lambda.diff + pi.diff

    if (is.finite(elbo.every.n)) {
      if (iter %% elbo.every.n == 0) {
        elbo <- GetVariationalELBO(x=x, e.mu=e.mu, e.mu2=e.mu2,
                                   e.lambda=e.lambda, e.log.det.lambda=e.log.det.lambda,
                                   lambda.par=lambda.par, n.par=n.par,
                                   e.log.pi=e.log.pi, e.z=e.z,
                                   priors=priors)
        elbo.list[[length(elbo.list) + 1]] <- elbo$entropy + elbo$log.lik
        entropy.list[[length(entropy.list) + 1]] <- elbo$entropy
        log.lik.list[[length(log.lik.list) + 1]] <- elbo$log.lik

      }
    }

    if (keep.updates) {
      updates[[iter]] <- list(e.mu=e.mu, e.mu2=e.mu2, e.lambda=e.lambda,
                              e.log.det.lambda=e.log.det.lambda,
                              e.log.pi=e.log.pi, lambda.par=lambda.par, n.par=n.par,
                              z.counts=colSums(e.z))
    }

    if (!quiet) {
      print(sprintf("%d: mu: %f, lambda: %f, pi: %f, total: %f",
                    iter, mu.diff, lambda.diff, pi.diff, total.diff))
    }
  }
  return(list(e.mu=e.mu, e.mu2=e.mu2, e.lambda=e.lambda,
              e.log.det.lambda=e.log.det.lambda,
              e.pi=e.pi, e.log.pi=e.log.pi, e.z=e.z,
              lambda.par=lambda.par, n.par=n.par, updates=updates,
              elbo=unlist(elbo.list),
              entropy=unlist(entropy.list),
              log.lik=unlist(log.lik.list)))
}


GetVariationalELBO <- function(x, e.mu, e.mu2, e.lambda, e.log.det.lambda,
                               lambda.par, n.par, e.log.pi, e.z, priors) {
  entropy <- GetVariationalEntropy(e.z, e.mu, e.mu2, lambda.par, n.par, colSums(e.z))
  log.lik <- CompleteLogLikelihoodWithPriors(x,
                                             e.z,
                                             e.mu,
                                             e.mu2,
                                             e.lambda,
                                             e.log.det.lambda,
                                             e.log.pi,
                                             FALSE, FALSE, FALSE,
                                             priors$mu.prior.mean,
                                             priors$mu.prior.info,
                                             priors$lambda.prior.v.inv,
                                             priors$lambda.prior.n,
                                             priors$p.prior.alpha,
                                             FALSE)
  return(list(entropy=entropy, log.lik=log.lik))
}

PostProcessVBResults <- function(vb.optimum) {
  # Args:
  #  - vb.optimum: A list of results as returned by GetVariationalSolution
  #
  # Returns:
  #  - A list of the variational covariances.

  k <- ncol(vb.optimum$e.mu)
  p <- nrow(vb.optimum$e.mu)

  # Get the mu covariances.
  mu.covs <- ListifyVectorizedMatrix(vb.optimum$e.mu2)
  for (this.k in 1:k) {
    mu.covs[[this.k]] <- (mu.covs[[this.k]] -
                          vb.optimum$e.mu[, this.k] %*% t(vb.optimum$e.mu[, this.k]))
  }

  # Get the lambda covariance.
  lambda.covs <- list()
  for (this.k in 1:k) {
    lambda.covs[[this.k]] <- GetWishartLinearCovariance(vb.optimum$lambda.par[, this.k],
                                                        vb.optimum$n.par[this.k])
  }

  # Get the pi covariance.
  log.pi.cov <- GetLogDirichletCovariance(colSums(vb.optimum$e.z))

  return(list(mu.covs=mu.covs, lambda.covs=lambda.covs, log.pi.cov=log.pi.cov))
}

CoreParametersFromMLE <- function(par, k, p) {
  # Get a vector of mu, lambda, and log(pi) from the MLE paramaterization
  par.unpack <- LinearlyUnpackParameters(par, k, p)
  return(c(par.unpack$mu,
           par.unpack$lambda,
           log(par.unpack$pi)))
}

CoreParameterNamesFromMLE <- function(k, p) {
  # Get a vector of mu, lambda, and log(pi) from the MLE paramaterization
  mu.names <- outer(1:p, 1:k, function(x, y) { paste("mu", y, x, sep="_")})

  lambda.names <- matrix(NA, nrow=p * (p + 1) / 2, ncol=k)
  for (this.k in 1:k) {
    lambda.names.mat <-
      outer(1:p, 1:p, function(x, y) { paste("lambda", this.k, x, y, sep="_")})
    for (i in 1:p) {
      for (j in 1:i) {
        lambda.names[GetUpperTriangularIndex(i - 1, j - 1) + 1, this.k] <- lambda.names.mat[i, j]
      }
    }
  }

  pi.names <- paste("pi", 1:k, sep="_")

  return(c(mu.names,
           lambda.names,
           pi.names))
}

CoreTruthFromPar <- function(par) {
  # Get a vector of mu, lambda, and log(pi) from the output of GenerateSampleParams
  # or a similar list
  true.lambda <- InvertLinearizedMatrices(VectorizeMatrixList(par$true.sigma))
  return(c(par$true.means, true.lambda, log(par$true.probs)))
}


CoreMeansFromVB <- function(vb.optimum) {
  # Get a vector of mu, lambda, and log(pi) from the VB paramaterization
  return(c(vb.optimum$e.mu, vb.optimum$e.lambda, vb.optimum$e.log.pi))
}

CoreVarsFromVB <- function(vb.optimum) {
  # Get a vector of variances of mu, lambda, and log(pi) from the VB paramaterization

  # The variances of mu:
  var.mu <- vb.optimum$e.mu2 - GetVectorizedOuterProductMatrix(vb.optimum$e.mu)
  var.mu.list <- ListifyVectorizedMatrix(var.mu)
  var.mu.mat <- matrix(NA, nrow(vb.optimum$e.mu), ncol(vb.optimum$e.mu))
  for (this.k in 1:length(var.mu.list)) {
    var.mu.mat[, this.k] <- diag(var.mu.list[[this.k]])
  }

  var.lambda.mat <- matrix(NA, nrow(vb.optimum$e.lambda), ncol(vb.optimum$e.lambda))
  for (this.k in 1:ncol(vb.optimum$e.lambda)) {
    var.lambda.mat[, this.k] <-
      diag(GetWishartLinearCovariance(vb.optimum$lambda.par[, this.k],
                                      vb.optimum$n.par[this.k]))
  }

  var.log.pi.mat <- diag(GetLogDirichletCovariance(colSums(vb.optimum$e.z)))

  return(c(var.mu.mat, var.lambda.mat, var.log.pi.mat))
}



CoreParametersFromMLE <- function(par, k, p) {
  # Get a vector of mu, lambda, and log(pi) from the MLE paramaterization
  par.unpack <- LinearlyUnpackParameters(par, k, p)
  return(c(par.unpack$mu,
           par.unpack$lambda,
           log(par.unpack$pi)))
}


CoreCovFromMLE <- function(mle.means, mle.cov, k, p, n.sims=1e5) {
  # Args:
  #  - mle.means: The maximum likelihood estimate in the packed MLE parameters
  #  - mle.cov: The covariance of the packed MLE parameters (i.e. the negative inverse hessian
  #    of the log likelihood)
  #  - k:  The number of components.
  #  - p:  The dimension of the mean.
  #  - sims: The number of data points to simulate.  (NB: this should probably scale with p)
  #
  # Returns:
  #  The covariance of the core parameters, computed by simulation.
  #  Using the delta method to get the variance out of the log Cholesky parameterization
  #  seems not worth the trouble.

  draws <- rmvnorm(n.sims, mean=mle.means, sigma=mle.cov)
  core.mle.draws <- t(apply(draws, MARGIN=1,
                            function(par) { CoreParametersFromMLE(par, k, p) }))
  return(cov(core.mle.draws))
}

CoreVarsFromMLE <- function(mle.means, mle.cov, k, p, n.sims=1e5) {
 mle.cov <- CoreCovFromMLE(mle.means=mle.means, mle.cov=mle.cov, k=k, p=p, n.sims=n.sims)
 return(diag(mle.cov))
}

CoreParametersFromGibbs <- function(mu.draws, lambda.draws, pi.draws) {
  gibbs.mu <- colMeans(mu.draws)
  gibbs.lambda <- colMeans(lambda.draws)
  gibbs.log.pi <- colMeans(log(pi.draws))

  gibbs.mu.sd <- colSds(mu.draws)
  gibbs.lambda.sd <- colSds(lambda.draws)
  gibbs.log.pi.sd <- colSds(log(pi.draws))

  return(list(means=c(gibbs.mu, gibbs.lambda, gibbs.log.pi),
              sds=c(gibbs.mu.sd, gibbs.lambda.sd, gibbs.log.pi.sd)))
}

CoreCovarianceFromGibbs <- function(mu.draws, lambda.draws, pi.draws) {
  all.draws <- cbind(mu.draws, lambda.draws, log(pi.draws))
  return(cov(all.draws))
}


GetThetaCovariance <- function(e.mu, e.mu2, lambda.par, n.par, e.z,
                               fit.mu=TRUE, fit.lambda=TRUE, fit.pi=TRUE) {
  mu.cov <- GetMuVariance(e.mu, e.mu2)
  lambda.cov <- GetLambdaVariance(lambda.par, n.par)
  log.pi.cov <- GetLogPiVariance(colSums(e.z))

  # Zero out the matrices we won't fit.
  if (!fit.mu) {
    mu.cov <- Matrix(0, nrow(mu.cov), ncol(mu.cov))
  }
  if (!fit.lambda) {
    lambda.cov <- Matrix(0, nrow(lambda.cov), ncol(lambda.cov))
  }
  if (!fit.pi) {
    log.pi.cov <- Matrix(0, nrow(log.pi.cov), ncol(log.pi.cov))
  }

  return(bdiag(mu.cov, log.pi.cov, lambda.cov))
}


GetLRVBCovariance <- function(x, e.mu, e.mu2,
                              e.lambda, e.z,
                              theta.cov) {
  htz <- GetHThetaZ(x, e.mu, e.mu2, e.lambda)
  htt <- GetHThetaTheta(x, e.z)
  z.cov <- GetZCovariance(e.z)
  theta.id <- Diagonal(nrow(theta.cov))
  term1 <- theta.cov %*% htt
  term2 <- htz %*% z.cov %*% Matrix::t(htz)
  lrvb.correction <- (theta.id - term1 - theta.cov %*% term2)
  return(Matrix::solve(lrvb.correction, theta.cov))
}


GetLeverageScores <- function(x, e.mu, e.mu2, e.lambda, e.z, lrvb.theta.cov) {
  x.cov <- GetXVariance(x=x)
  n <- nrow(x)
  htx <- GetHThetaX(e_z=e.z, e_mu=e.mu,
                    e_lambda=e.lambda)
  hzx <- GetHZX(n_tot=n, e_mu=e.mu,
                e_lambda=e.lambda)
  htz <- GetHThetaZ(x=x, e_mu=e.mu,
                    e_mu2=e.mu2,
                    e_lambda=e.lambda)
  z.cov <- GetZCovariance(z_mat=e.z)
  return(as.matrix(lrvb.theta.cov %*% (htx + htz %*% z.cov %*% hzx) %*% x.cov))
}


CopyListToEnvironment <- function(my.list, my.env) {
  for (x in names(my.list)) {
    assign(x, my.list[[x]], my.env)
  }
}


##########
# Functions to extract meaning from the output of rnmixGibbs

GetMuFieldSizeFromCompdraw <- function(compdraw, k.vector, p) {
  GetMuSize <- function(i) {
    mu.i <- matrix(compdraw[[i]][["mu"]], nrow=1)
    return(sum(mu.i^2))
  }
  draws <- lapply(k.vector, GetMuSize)
  return(do.call(cbind, draws))
}

GetMuFieldFromCompdraw <- function(compdraw, k.vector, p) {
  draws <- lapply(k.vector, function(i) {
    matrix(compdraw[[i]][["mu"]], nrow=1)
  })
  return(do.call(cbind, draws))
}

GetLambdaFieldFromCompdraw <- function(compdraw, k.vector, p) {
  GetLambdaVec <- function(i) {
    rooti <- matrix(compdraw[[i]][["rooti"]], nrow=p, ncol=p)
    # This might be wrong.
    lambda <- rooti %*% t(rooti)
    return(matrix(ConvertSymmetricMatrixToVector(lambda), nrow=1))
  }
  draws <- lapply(k.vector, GetLambdaVec)
  return(do.call(cbind, draws))
}

GetFieldsFromnNmix <- function(nmix, FUN, k.vector, p) {
  return(do.call(rbind, lapply(nmix$compdraw,
                               function(compdraw) {
                                 FUN(compdraw, k.vector, p)
                               })))
}

colSds <- function(df) {
  apply(df, 2, sd)
}

