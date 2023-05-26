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
set.seed(52)
dat <- rnorm(10, c(-5, 5))
K_max <- 5

fmm_iter(K_max, rep(1:2, 5), dat, mu0_cluster = rep(0, K_max), 
         lambda_cluster = rep(1, K_max), a_sigma_cluster = rep(1, K_max), 
         b_sigma_cluster = rep(1, K_max))

result <- fmm(100, K_max, rep(1:2, 5), dat, mu0_cluster = rep(0, K_max), 
    lambda_cluster = rep(1, K_max), a_sigma_cluster = rep(1, K_max), 
    b_sigma_cluster = rep(1, K_max))

salso(result)

rm(list = ls())
test_vec <- rep(NA, 1000)
for(i in 1:1000){
  test_vec[i] <- rmultinom_1(rep(0.1, 10), 10)
}
table(test_vec)
