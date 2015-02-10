library(MVNMixtureLRVB)
library(Matrix)
library(mvtnorm)
library(ggplot2)
library(numDeriv)
library(reshape2)
library(dplyr)
library(coda)
library(digest)
library(gridExtra)
library(bayesm)

kShowPlots <- FALSE
kSaveResults <- FALSE

#######
# Generate data

n <- 4000
k <- 2
p <- 2
n.sims <- 100
vars.scale <- 0.4
anisotropy <- 1
#n.mh.draws <- 1e4
n.gibbs.draws <- 5e3
burnin <- 1000

analysis.name <- sprintf("n%d_k%d_p%d_sims%d_scale%0.1f_anisotropy%0.1f",
                         n, k, p, n.sims, vars.scale, anisotropy)

par <- GenerateSampleParams(k=k, p=p, vars.scale=vars.scale,
                            anisotropy=anisotropy, random.rotation=FALSE)

data <- GenerateMultivariateData(n, par$true.means, par$true.sigma, par$true.probs)
priors <- GenerateSamplePriors(x=data$x, k=k, lambda.scale=0.01)
analysis.hash <- digest(list(data, par))

true.lambda <- InvertLinearizedMatrices(VectorizeMatrixList(par$true.sigma))

df.col.names <- paste("X", 1:p, sep="")
x <- data$x
x.df <- data.frame(data$x)
names(x.df) <- df.col.names
x.df$label <- data$component.labels

means.df <- data.frame(t(par$true.means))
names(means.df) <- df.col.names


####################
# Compute the VB fit
vb.opt.time <- Sys.time()
vb.optimum <-
  GetVariationalSolution(x=x,
                         e.mu=par$true.means,
                         e.lambda=true.lambda,
                         e.log.det.lambda=true.log.det.lambda,
                         e.p=par$true.probs,
                         e.log.pi=GetELogDirichlet(par$true.probs * n),
                         fit.mu=TRUE, fit.lambda=TRUE, fit.pi=TRUE,
                         e.z=data$components,
                         priors=priors, tolerance=1e-9, quiet=FALSE)
vb.opt.time <- Sys.time() - vb.opt.time
theta.cov <- GetThetaCovariance(e.mu=vb.optimum$e.mu, e.mu2=vb.optimum$e.mu2,
                                lambda.par=vb.optimum$lambda.par, n.par=vb.optimum$n.par,
                                e.z=vb.optimum$e.z,
                                fit.mu=TRUE, fit.lambda=TRUE, fit.pi=TRUE)

lrvb.time <- Sys.time()
lrvb.theta.cov <- CPPGetLRVBCovariance(x=x,
                                       e_mu=vb.optimum$e.mu,
                                       e_mu2=vb.optimum$e.mu2,
                                       e_lambda=vb.optimum$e.lambda,
                                       e_z=vb.optimum$e.z,
                                       theta_cov=theta.cov, verbose=TRUE)
lrvb.time <- Sys.time() - lrvb.time
cat("VB time: ")
print(vb.time <- lrvb.time + vb.opt.time)


####################
# Get gibbs draws

gibbs.time <- Sys.time()
# Fit the normal mixture
prior.list <- list(ncomp=k)
out <- rnmixGibbs(Data=list(y=x),
                  Prior=prior.list,
                  Mcmc=list(R=n.gibbs.draws, keep=1))
print(gibbs.time <- Sys.time() - gibbs.time)

mu.sizes <- colMeans(GetFieldsFromnNmix(out$nmix,
                                        GetMuFieldSizeFromCompdraw,
                                        1:k, p)[-(1:burnin), ])
k.vector <- order(mu.sizes)
original.mu.draws <- GetFieldsFromnNmix(out$nmix, GetMuFieldFromCompdraw, k.vector, p)
original.lambda.draws <- GetFieldsFromnNmix(out$nmix, GetLambdaFieldFromCompdraw, k.vector, p)
original.pi.draws <- out$nmix$probdraw[, k.vector]

mu.draws <- data.frame(original.mu.draws[-(1:burnin),])
names(mu.draws) <- GetMatrixVectorNames("mu", p=p, k=k)

lambda.draws <- data.frame(original.lambda.draws[-(1:burnin),])
names(lambda.draws) <- GetSymmetricMatrixVectorNames("lambda", p, k)

pi.draws <- data.frame(original.pi.draws[-(1:burnin),])
names(pi.draws) <- paste("pi", 1:k, sep="_")

effectiveSize(pi.draws)
effectiveSize(mu.draws)
effectiveSize(lambda.draws)


##################
# Process results

core.names <- CoreParameterNamesFromMLE(k, p)

truth.means <- c(par$true.means, true.lambda, log(par$true.probs))

core.gibbs.results <- CoreParametersFromGibbs(mu.draws, lambda.draws, pi.draws)

core.vb.means <- CoreMeansFromVB(vb.optimum)
core.vb.sd <- sqrt(CoreVarsFromVB(vb.optimum))

lrvb.vars <- diag(lrvb.theta.cov)
lrvb.vars.list <- UnpackVBThetaParameters(lrvb.vars, n, p, k)
core.lrvb.vars <- c(lrvb.vars.list$e_mu, lrvb.vars.list$e_lambda, lrvb.vars.list$e_log_pi)
core.lrvb.sd <- sqrt(core.lrvb.vars)
core.vars <-  data.frame(var=core.names,
                         truth.mean=truth.means,
                         gibbs.mean=core.gibbs.results$means,
                         gibbs.sd=core.gibbs.results$sd,
                         vb.mean=core.vb.means,
                         vb.sd=core.vb.sd,
                         lrvb.sd=core.lrvb.sd)

core.vars.melt <- melt(core.vars, id.vars = c("var"))
core.vars.melt$measure <- sub("^.*\\.", "", core.vars.melt$variable)
core.vars.melt$method <- sub("\\..*$", "", core.vars.melt$variable)
core.vars.melt$parameter <- sub("\\_.*$", "", core.vars.melt$var)

core.vars.df <- dcast(core.vars.melt, var + parameter + measure ~ method)

this.parameter <- "mu"
grid.arrange(
  ggplot(filter(core.vars.df, parameter == this.parameter, measure == "mean")) +
    geom_point(aes(x=truth, y=gibbs, color="gibbs"), size=3) +
    geom_point(aes(x=truth, y=vb, color="vb"), size=3) +
    ggtitle(paste(this.parameter, "point estimates")) + xlab("Truth") + ylab("estimates") +
    expand_limits(x=0, y=0) +
    geom_abline(aes(slope=1, intercept=0), color="gray") +
    geom_hline(aes(yintercept=0), color="gray") +
    geom_vline(aes(xintercept=0), color="gray"),

  ggplot(filter(core.vars.df, parameter == this.parameter, measure == "sd")) +
    geom_point(aes(x=gibbs, y=lrvb, color="lrvb"), size=3) +
    geom_point(aes(x=gibbs, y=vb, color="vb"), size=3) +
    ggtitle(paste(this.parameter, "standard deviations")) + xlab("MH Stdev") + ylab("estimates") +
    expand_limits(x=0, y=0) +
    geom_abline(aes(slope=1, intercept=0), color="gray") +
    geom_hline(aes(yintercept=0), color="gray") +
    geom_vline(aes(xintercept=0), color="gray"),
  nrow=1)



# Visualize the VB fit
vb.probs <- colMeans(vb.optimum$e.z)
vb.sigma <- InvertLinearizedMatrices(vb.optimum$e.lambda)
vb.data <- GenerateMultivariateData(1e4, vb.optimum$e.mu,
                                    ListifyVectorizedMatrix(vb.sigma),
                                    vb.probs)
vb.x.df <- data.frame(vb.data$x)
names(vb.x.df) <- df.col.names
vb.means.df <- data.frame(t(vb.optimum$e.mu))
names(vb.means.df) <- df.col.names
if (p == 2) {
  ggplot() +
    geom_density2d(data=x.df, aes(x=X1, y=X2, color="true")) +
    geom_density2d(data=vb.x.df, aes(x=X1, y=X2, color="vb fit")) +
    geom_point(data=means.df, aes(x=X1, y=X2), color="red", size=5) +
    geom_point(data=vb.means.df, aes(x=X1, y=X2), color="green", size=5)
} else if (p == 1) {
  ggplot() +
    geom_density(data=x.df, aes(x=X1, color="true"), lwd=2) +
    geom_density(data=vb.x.df, aes(x=X1, color="vb fit"), lwd=2) +
    geom_vline(data=means.df, aes(xintercept=X1), color="red", size=1) +
    geom_vline(data=vb.means.df, aes(xintercept=X1), color="green", size=1)
}
