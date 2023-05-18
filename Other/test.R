### Required Libraries: --------------------------------------------------------
library(Rcpp)
library(RcppArmadillo)
library(devtools)
library(LaplacesDemon)
library(mvtnorm)
library(tidyverse)
library(DirichletReg)
library(salso)
library(rootSolve)
library(metRology)

### Required Commands for building the packages: -------------------------------
uninstall()
compileAttributes()
build()
install()
library(ClusterNormal)

### Sandbox: -------------------------------------------------------------------
rm(list = ls())
set.seed(43)
ci_true <- sample(1:5, 500, replace = TRUE)
dat <- rnorm(500, c(-100, -50, 0, 50, 100)[ci_true])
clus_init <- rep(1, 500)
K <- 5
xi_vec <- rep(0.01, K)
mu0_vec <- rep(0, K)
a_sigma_vec <- rep(5, K)
b_sigma_vec <- rep(1, K)
lambda_vec <- rep(1, K)
a_theta <- 1
b_theta <- 1
sm_iter <- 10

model <- SFDM_model(1000, K, clus_init, xi_vec, scale(dat), mu0_vec, 
                    a_sigma_vec, b_sigma_vec, lambda_vec, a_theta, b_theta, 
                    sm_iter, 250)
table(salso(model$iter_assign[-c(1:500), ]), ci_true)



u <- rgamma(1, length(clus_a), sum(alpha_vec))
u
SFDM_alpha(clus_a, xi_vec, alpha_vec, u)

hist(rgamma(10000, 500, sum(rgamma(1, xi_vec, 1))))
hist(rgamma(10000, 1, sum(rgamma(1, xi_vec, 1))))

rgamma(1, 250 + 0.01, )


rm(list = ls())
set.seed(31)
ci_actual <- rep(1:3, 5)
mu0 <- c(-1, 0, 1)
a_sigma <- c(1, 1, 1.5)
b_sigma <- c(1, 0.1, 0.1)
lambda <- c(1, 1, 1)
y <- rnorm(15)

log_marginal(ci_actual, y, a_sigma, b_sigma, lambda, mu0)

i <- 10

log(sqrt(2 * pi))



(0 + y[5])/(1 + 1)
