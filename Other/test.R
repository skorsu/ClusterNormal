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
