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

set.seed(32134)
### Simulate the data
ci_true <- sample(1:5, 500, replace = TRUE)
dat <- rnorm(500, c(-20, -10, 0, 10, 20)[ci_true])
K_max <- 10

start_time <- Sys.time()
test_result <- SFDM_model(iter = 5000, K = K_max, init_assign = rep(0, 500), y = dat, 
                          mu0_cluster = rep(0, K_max), lambda_cluster = rep(1, K_max), 
                          a_sigma_cluster = rep(1, K_max), b_sigma_cluster = rep(1, K_max), 
                          xi_cluster = rep(1, K_max), a_theta = 1, b_theta = 1, 
                          launch_iter = 10, print_iter = 500)
print(Sys.time() - start_time)

table(test_result$split_or_merge, test_result$sm_status)

table(salso(test_result$iter_assign[-c(1:2500), ]), ci_true)
(1791 + 212)/5000

rm(list = ls())
set.seed(2)
xi_clus <- rep(0.01, 5)
ci_true <- rep(0, 20)
alp <- c(rgamma(1, 0.01, 1), rep(0, 4))
SFDM_alpha(ci_true, xi_clus, alp, rgamma(1, length(ci_true), sum(alp)))


rm(list = ls())
set.seed(12)
xi_a <- rep(0.01, 5)
clus_vec <- sample(0:3, 30, replace = TRUE)
log_prior_cluster(clus_vec, xi_a)
table(clus_vec)
log(factorial(30)) - sum(log(factorial(table(clus_vec)))) +
  lgamma(0.04) - lgamma(30.04) + sum(lgamma(table(clus_vec) + 0.01)) - sum(lgamma(rep(0.01, 4)))

rm(list = ls())
set.seed(31)
ci_true <- rep(0:3, 125)
dat <- rnorm(500, c(-10, -5, 5, 10)[ci_true + 1])
K <- 5

test <- SFDM_SM(K, rep(0, 500), dat, alpha_vec = c(rgamma(1, 1, 1), rep(0, 4)),
        mu0_cluster = rep(0, K), lambda_cluster = rep(1, K), 
        a_sigma_cluster = rep(1, K), b_sigma_cluster = rep(1, K), 
        xi_cluster = rep(1, K), launch_iter = 5, a_theta = 1, b_theta = 1)

accept_vec <- rep(NA, 1000)
clus_assign <- matrix(NA, nrow = 500, ncol = 1000)
alpha_assign <- matrix(NA, nrow = K, ncol = 1000)
for(i in 1:1000){
  test <- SFDM_SM(K, rep(0, 500), dat, alpha_vec = c(rgamma(1, 1, 1), rep(0, 4)),
                  mu0_cluster = rep(0, K), lambda_cluster = rep(1, K), 
                  a_sigma_cluster = rep(1, K), b_sigma_cluster = rep(1, K), 
                  xi_cluster = rep(1, K), launch_iter = 5, a_theta = 1, b_theta = 1)
  accept_vec[i] <- test$accept_proposed
  clus_assign[, i] <- test$new_assign
  alpha_assign[, i] <- test$new_alpha
}

mean(accept_vec) * 100
(1:1000)[accept_vec == 1]

table(clus_assign[, 11])
clus_assign[, 1]
alpha_assign[, 11]

rm(list = ls())
set.seed(402)
alpha_vec <- rgamma(10, 3, 1)
alpha_vec
adjust_alpha(rep(c(0, 3, 2, 1, 6), 10), alpha_vec)

rm(list = ls())
set.seed(32)
ci_true <- rep(0:3, 5)
alpha_a <- c(rgamma(2, 1, 1), rep(0, 3))
K <- 5
dat <- rnorm(20, c(-10, -5, 5, 10)[ci_true + 1])

SFDM_realloc(c(0, rep(1, 19)), dat, alpha_vec = alpha_a, mu0_cluster = rep(0, K), lambda_cluster = rep(1, K), 
             a_sigma_cluster = rep(1, K), b_sigma_cluster = rep(1, K), 
             xi_cluster = rep(1, K))

rm(list = ls())
K_m <- 5
set.seed(12)
ci_true <- sample(0:(K_m-1), 500, replace = TRUE)
dat <- rnorm(500, c(0, 7.5, 15, 25, 35)[(ci_true + 1)])
hist(dat, breaks = 100)
K_max <- 10

data.frame(x = ci_true, y = scale(dat, center = TRUE, scale = FALSE)) %>%
  group_by(x) %>%
  summarise(mean(y), var(y))

start_time <- Sys.time()
result <- fmm(5000, K_max, rep(0:0, 500), 
              scale(dat, center = TRUE, scale = FALSE), mu0_cluster = rep(0, K_max), 
    lambda_cluster = rep(1, K_max), a_sigma_cluster = rep(1, K_max), 
    b_sigma_cluster = rep(1, K_max), xi_cluster = rep(1, K_max))
total_time <- difftime(Sys.time(), start_time, units = "secs")
total_time

clus_assign <- salso(result[-c(1:3500), ], maxNClusters = K_max)
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
