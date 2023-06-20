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

t <- SFCMM_realloc(dat_sim, K_max = 10, a0 = 0.1, b0 = 0.1, mu0 = 0, s20 = 100, xi0 = 1, 
              ci_init = rep(1, 500), mu = rnorm(10, 0, sqrt(100)), 
              s2 = 1/(rgamma(10, 0.1, 0.1)), alpha_vec = rgamma(10, 1, 1))

t <- SFCMM_SM(dat_sim, K_max = 10, a0 = 0.1, b0 = 0.1, mu0 = 0, s20 = 100, xi0 = 1, 
         ci_init = rep(0:9, 50), mu_init = rnorm(10, 0, sqrt(100)), 
         s2_init = 1/(rgamma(10, 0.1, 0.1)), alpha_init = rgamma(10, 1, 1),
         launch_iter = 10)

table(t$new_ci)

set.seed(1843)
start_time <- Sys.time()
test_result <- SFCMM_model(iter = 10000, K_max = 20, init_assign = rep(0, 500),
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
