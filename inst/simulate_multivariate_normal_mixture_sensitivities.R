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

kShowPlots <- FALSE

#######
# Generate data

n <- 1000
k <- 2
p <- 2
vars.scale <- 0.5
analysis.name <- sprintf("n%d_k%d_p%d_vars%f", n, k, p, vars.scale)
par <- GenerateSampleParams(k=k, p=p, vars.scale=vars.scale)
data <- GenerateMultivariateData(n, par$true.means, par$true.sigma, par$true.probs)
priors <- GenerateSamplePriors(x=data$x, k=k)
analysis.hash <- digest(list(data, par))
true.lambda <- InvertLinearizedMatrices(VectorizeMatrixList(par$true.sigma))
true.log.det.lambda <- LogDeterminantOfLinearizedMatrices(true.lambda)

df.col.names <- paste("X", 1:p, sep="")
x <- data$x
x.df <- data.frame(data$x)
names(x.df) <- df.col.names
x.df$label <- data$component.labels

means.df <- data.frame(t(par$true.means))
names(means.df) <- df.col.names


####################
# Compute the VB fit and LRVB leverage scores

fit.mu <- TRUE
fit.pi <- TRUE
fit.lambda <- TRUE
vb.optimum <-
  GetVariationalSolution(x=x,
                         e.mu=par$true.means,
                         e.lambda=true.lambda,
                         e.log.det.lambda=true.log.det.lambda,
                         e.p=par$true.probs,
                         e.log.pi=GetELogDirichlet(par$true.probs * n),
                         fit.mu=fit.mu, fit.lambda=fit.lambda, fit.pi=fit.pi,
                         e.z=data$components,
                         priors=priors, tolerance=1e-9)

theta.cov <- GetThetaCovariance(e.mu=vb.optimum$e.mu, e.mu2=vb.optimum$e.mu2,
                                lambda.par=vb.optimum$lambda.par, n.par=vb.optimum$n.par,
                                pi.par=vb.optimum$pi.par,
                                fit.mu=fit.mu, fit.lambda=fit.lambda, fit.pi=fit.pi)

lrvb.theta.cov <- CPPGetLRVBCovariance(x=x,
                                       e_mu=vb.optimum$e.mu,
                                       e_mu2=vb.optimum$e.mu2,
                                       e_lambda=vb.optimum$e.lambda,
                                       e_z=vb.optimum$e.z,
                                       theta_cov=theta.cov, verbose=FALSE)

tx.cov <- GetLeverageScores(x=data$x,
                            e.mu=vb.optimum$e.mu, e.mu2=vb.optimum$e.mu2,
                            e.lambda=vb.optimum$e.lambda, e.z=vb.optimum$e.z,
                            lrvb.theta.cov=lrvb.theta.cov)



# VB Plots
vb.probs <- colMeans(vb.optimum$e.z)
vb.sigma <- InvertLinearizedMatrices(vb.optimum$e.lambda)
vb.data <- GenerateMultivariateData(1e4, vb.optimum$e.mu,
                                    ListifyVectorizedMatrix(vb.sigma),
                                    vb.probs)
vb.x.df <- data.frame(vb.data$x)
names(vb.x.df) <- df.col.names
vb.means.df <- data.frame(t(vb.optimum$e.mu))
names(vb.means.df) <- df.col.names
if (kShowPlots) {
  ggplot() +
    geom_density2d(data=x.df, aes(x=X1, y=X2, color="true")) +
    geom_density2d(data=vb.x.df, aes(x=X1, y=X2, color="vb fit")) +
    geom_point(data=means.df, aes(x=X1, y=X2), color="red", size=5) +
    geom_point(data=vb.means.df, aes(x=X1, y=X2), color="green", size=5)
}


####################
# Confirm that the LRVB sensitivities match the effects of actually perturbing the data.
# This is really slow because we are refitting the VB model many times, but the point is to
# illustrate that LRVB gives the same results much faster.

delta <- 0.0001

mu.diff.list <- list()
log.pi.diff.list <- list()
lambda.diff.list <- list()
mu.effect.list <- list()
log.pi.effect.list <- list()
lambda.effect.list <- list()
perturbation.list <- list()
iter <- 0
pb <- txtProgressBar(max=p * n, style=3)
for (x.col in 1:p) {
  for (x.row in 1:n) {
    setTxtProgressBar(pb, (x.row - 1) * p + x.col)
    iter <- iter + 1
    new.x <- x
    new.x[x.row, x.col] <- new.x[x.row, x.col] + delta

    x.index <- GetXCoordinate(x.row - 1, x.col - 1, n, p) + 1
    raw.effect <- UnpackVBThetaParameters(tx.cov[, x.index] * delta, n, p, k)

    new.vb.optimum <-
      GetVariationalSolution(x=new.x,
                             e.mu=vb.optimum$e.mu,
                             e.mu2=vb.optimum$e.mu2,
                             e.lambda=vb.optimum$e.lambda,
                             e.log.det.lambda=vb.optimum$e.log.det.lambda,
                             e.pi=vb.optimum$e.pi,
                             e.log.pi=vb.optimum$e.log.pi,
                             fit.mu=fit.mu, fit.lambda=fit.lambda, fit.pi=fit.pi,
                             e.z=vb.optimum$e.z,
                             priors=priors, tolerance=1e-9, quiet=TRUE)

    mu.diff.list[[iter]] <- as.numeric(new.vb.optimum$e.mu - vb.optimum$e.mu)
    log.pi.diff.list[[iter]] <- as.numeric(new.vb.optimum$e.log.pi - vb.optimum$e.log.pi)
    lambda.diff.list[[iter]] <- as.numeric(new.vb.optimum$e.lambda - vb.optimum$e.lambda)

    mu.effect.list[[iter]] <- as.numeric(raw.effect$e_mu)
    log.pi.effect.list[[iter]] <- as.numeric(raw.effect$e_log_pi)
    lambda.effect.list[[iter]] <- as.numeric(raw.effect$e_lambda)

    perturbation.list[[iter]] <- c(x.row, x.col)
  }
}
close(pb)


perturbations <- data.frame(do.call(rbind, perturbation.list))
names(perturbations)  <- c("row", "col")

mu.diff <- do.call(rbind, mu.diff.list)
log.pi.diff <- do.call(rbind, log.pi.diff.list)
lambda.diff <- do.call(rbind, lambda.diff.list)

mu.effect <- do.call(rbind, mu.effect.list)
log.pi.effect <- do.call(rbind, log.pi.effect.list)
lambda.effect <- do.call(rbind, lambda.effect.list)

base.title <- sprintf("\nn=%d, k=%d, p=%d", n, k, p)
grid.arrange(
  ggplot() +
    geom_point(aes(x=as.numeric(mu.diff), y=as.numeric(mu.effect)), size=2) +
    xlab("Actual change") + ylab("Leverage score") +
    geom_abline(aes(slope=1, intercept=0)) +
    ggtitle(paste("Mu leverage scores", base.title)),
  ggplot() +
    geom_point(aes(x=as.numeric(log.pi.diff), y=as.numeric(log.pi.effect)), size=2) +
    xlab("Actual change") + ylab("Leverage score") +
    geom_abline(aes(slope=1, intercept=0)) +
    ggtitle(paste("Log pi leverage scores", base.title)),
  ggplot() +
    geom_point(aes(x=as.numeric(lambda.diff), y=as.numeric(lambda.effect)), size=2) +
    xlab("Actual change") + ylab("Leverage score") +
    geom_abline(aes(slope=1, intercept=0)) +
    ggtitle(paste("Lambda leverage scores", base.title)),
  ncol=3
)


###################################
# Visualize the leverage scores.

mu.effect.df <- data.frame(mu.effect)
names(mu.effect.df) <- GetMatrixVectorNames("mu", p=p, k=k)

lambda.effect.df <- data.frame(lambda.effect)
names(lambda.effect.df) <- GetSymmetricMatrixVectorNames("lambda", p, k)

x.melt.df <- x.df
x.melt.df$row <- 1:nrow(x.melt.df)
x.melt.df <- melt(x.melt.df, id.vars=c("label", "row"))
x.melt.df$col <- as.numeric(sub("^X", "", x.melt.df$variable))

leverage.df <- 
  inner_join(cbind(mu.effect.df, lambda.effect.df, perturbations),
             mutate(x.df, row=1:nrow(x.df)), by=c("row"))

this.x.col <- 1
base.title <- sprintf("Effect of X%d perturbations on", x.col)
this.leverage.df <- filter(leverage.df, col == this.x.col)
grid.arrange(
  ggplot(this.leverage.df) +
    geom_density2d(aes(x=X1, y=X2)) +
    geom_point(aes(x=X1, y=X2, color=mu_1_1), size=4) +
    scale_colour_gradient2(low="red", high="yellow", mid="black", midpoint=0) +
    ggtitle(paste(base.title, "mu_1_1")),
  ggplot(this.leverage.df) +
    geom_density2d(aes(x=X1, y=X2)) +
    geom_point(aes(x=X1, y=X2, color=mu_1_2), size=4) +
    scale_colour_gradient2(low="red", high="yellow", mid="black", midpoint=0) +
    ggtitle(paste(base.title, "mu_1_2")),
  ggplot(this.leverage.df) +
    geom_density2d(aes(x=X1, y=X2)) +
    geom_point(aes(x=X1, y=X2, color=mu_2_1), size=4) +
    scale_colour_gradient2(low="red", high="yellow", mid="black", midpoint=0) +
    ggtitle(paste(base.title, "mu_2_1")),
  ggplot(this.leverage.df) +
    geom_density2d(aes(x=X1, y=X2)) +
    geom_point(aes(x=X1, y=X2, color=mu_2_2), size=4) +
    scale_colour_gradient2(low="red", high="yellow", mid="black", midpoint=0) +
    ggtitle(paste(base.title, "mu_2_2")),
  ncol=2)


grid.arrange(
  ggplot(this.leverage.df) +
    geom_density2d(aes(x=X1, y=X2)) +
    geom_point(aes(x=X1, y=X2, color=lambda_1_1_1), size=4) +
    scale_colour_gradient2(low="red", high="yellow", mid="black", midpoint=0) +
    ggtitle(paste(base.title, "lambda_1_1_1")),
  ggplot(this.leverage.df) +
    geom_density2d(aes(x=X1, y=X2)) +
    geom_point(aes(x=X1, y=X2, color=lambda_1_2_2), size=4) +
    scale_colour_gradient2(low="red", high="yellow", mid="black", midpoint=0) +
    ggtitle(paste(base.title, "lambda_1_2_2")),
  ggplot(this.leverage.df) +
    geom_density2d(aes(x=X1, y=X2)) +
    geom_point(aes(x=X1, y=X2, color=lambda_2_1_1), size=4) +
    scale_colour_gradient2(low="red", high="yellow", mid="black", midpoint=0) +
    ggtitle(paste(base.title, "lambda_2_1_1")),
  ggplot(this.leverage.df) +
    geom_density2d(aes(x=X1, y=X2)) +
    geom_point(aes(x=X1, y=X2, color=lambda_2_2_2), size=4) +
    scale_colour_gradient2(low="red", high="yellow", mid="black", midpoint=0) +
    ggtitle(paste(base.title, "lambda_2_2_2")),
  ncol=2)


grid.arrange(
  ggplot(this.leverage.df) +
    geom_density2d(aes(x=X1, y=X2)) +
    geom_point(aes(x=X1, y=X2, color=lambda_1_1_2), size=4) +
    scale_colour_gradient2(low="red", high="yellow", mid="black", midpoint=0) +
    ggtitle(paste(base.title, "lambda_1_1_2")),
  ggplot(this.leverage.df) +
    geom_density2d(aes(x=X1, y=X2)) +
    geom_point(aes(x=X1, y=X2, color=lambda_2_1_2), size=4) +
    scale_colour_gradient2(low="red", high="yellow", mid="black", midpoint=0) +
    ggtitle(paste(base.title, "lambda_2_1_2")),
  ncol=2)


