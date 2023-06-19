### Required Libraries: --------------------------------------------------------
library(Rcpp)
library(RcppArmadillo)
library(devtools)
library(LaplacesDemon)
library(mvtnorm)
library(tidyverse)
library(DirichletReg)
library(invgamma)
library(salso)
library(rootSolve)
library(metRology)
library(ggplot2)
library(gridExtra)
library(xtable)
library(mclustcomp)

### Required Commands for building the packages: -------------------------------
uninstall()
compileAttributes()
build()
install()
library(ClusterNormal)

### Sandbox: -------------------------------------------------------------------

rm(list = ls())
### Data Simulation: (1)
set.seed(1843)
N <- 500
K <- 5
ci_true <- sample(1:K, N, replace = TRUE)
dat_sim <- rnorm(500, (c(0, 7.5, 15, 25, 35)[ci_true])/2, 1)
ggplot(data.frame(x = dat_sim, ci_true), aes(x = x, fill = factor(ci_true))) +
  geom_histogram(bins = 100) +
  theme_bw()

set.seed(1843)
start_time <- Sys.time()
test_result <- SFCMM_model(iter = 10000, K_max = 10, init_assign = rep(0, 500),
                           y = dat_sim, a0 = 0.01, b0 = 0.01, mu0 = 0, s20 = 100,
                           xi0 = 1, launch_iter = 10, print_iter = 2000)
model_time <- Sys.time() - start_time
model_time
table(salso(test_result$iter_assign[-(1:5000), ]), ci_true)

set.seed(1843)
start_time <- Sys.time()
test_result <- SFDMM_model(iter = 10000, K_max = 10, init_assign = rep(0, 500),
                           y = dat_sim, a0 = 0.01, b0 = 0.01, mu0 = 0, s20 = 100,
                           xi0 = 1, a_theta = 1, b_theta = 1, launch_iter = 10, print_iter = 2000)
model_time <- Sys.time() - start_time
model_time
table(salso(test_result$iter_assign[-(1:5000), ]), ci_true)
