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
K_m <- 5
set.seed(52)
ci_true <- sample(0:(K_m-1), 500, replace = TRUE)
dat <- rnorm(500, seq(-100, 100, length.out = K_m)[(ci_true + 1)])
hist(scale(dat), breaks = 100)
K_max <- 5

start_time <- Sys.time()
result <- fmm(10000, K_max, rep(0:0, 500), scale(dat), mu0_cluster = rep(0, K_max), 
    lambda_cluster = rep(1, K_max), a_sigma_cluster = rep(10, K_max), 
    b_sigma_cluster = rep(1, K_max), xi_cluster = rep(1, K_max))
total_time <- difftime(Sys.time(), start_time, units = "secs")
total_time

clus_assign <- salso(result[-c(1:5000), ], maxNClusters = K_max)
table(clus_assign, ci_true)

quantile(scale(dat)[clus_assign == 2])

data.frame(x = scale(dat)) %>%
  arrange(x) %>%
  mutate(previous = lag(x)) %>%
  mutate(diff = x - previous) %>%
  arrange(-diff) %>%
  head(15)

rm(list = ls())
test_vec <- rep(NA, 1000)
for(i in 1:1000){
  test_vec[i] <- rmultinom_1(rep(0.1, 10), 10)
}
table(test_vec)
