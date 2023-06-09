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
set.seed(31807)
ci_true <- rep(1:5, 10)
dat <- rnorm(50, c(0, 7.5, 15, 25, 35)[ci_true], 1)

test_result <- fmm_rcpp(iter = 10000, y = dat, K_max = 5, 
                        a0 = 1, b0 = 1, mu0 = 0, s20 = 100, xi0 = 1, 
                        ci_init = rep(1, 50))

table(salso(test_result$assign_mat[-c(1:7500), ], maxNClusters = 5), ci_true)
plot(test_result$mu[-c(1:7500), 2], type = "l")

hist(test_result$mu[-c(1:7500), 4])
apply(test_result$mu[-c(1:7500), ], 2, mean)
apply(test_result$sigma2[-c(1:7500), ], 2, mean)

plot(test_result$sigma2[-c(1:7500), 4], type = "l")

hist(test_result$sigma2)
hist(rnorm(10000, 0, sqrt(100)))
qqplot(test_result$sigma2, 1/rgamma(10000, 1, 1))
abline(0, 1)
rm(list = ls())
set.seed(1243)
mu <- rnorm(1, 0, sqrt(1000))
s2 <- 1/(rgamma(1, 1, 1))
y <- rnorm(1, mu, sqrt(s2))
log_marginal(y, mu0 = 0, s20 = 1000, a = 3, b = 4, mu, s2)
  
(-((y - mu)^2)/(2 * s2)) + (3.5 * log(4 + (0.5 * ((y - mu)^2)))) + ((y^2)/(2 * (s2 + 1000)))

#### Storing the intermediate Result
set.seed(3214)
dat <- scale(dat, scale = FALSE)
xin <- rep(0, K_max)
mun <- rep(0, K_max)
kn <- rep(0, K_max)
nun <- rep(0, K_max)
s2n <- rep(0, K_max)
SS_vec <- rep(0, K_max)

for(k in 1:3){
  nk <- sum(ci_true == k)
  ybar <- ifelse(nk > 0, mean(dat[ci_true == k]), 0)
  SS_vec[k] <- ifelse(nk > 1, (nk - 1) * var(dat[ci_true == k]), 0)
  kn[k] <- k0 + nk 
  nun[k] <- nu0 + nk
  mun[k] <- (k0*mu0 + nk*ybar)/kn[k]
  s2n[k] <- (nu0*s20 + SS_vec[k] + k0*nk*(ybar-mu0)^2/kn[k])/(nun[k])
  xin[k] <- xi0 + nk
}

mu_vec <- matrix(NA, nrow = 1000, ncol = 3)
s2_vec <- matrix(NA, nrow = 1000, ncol = 3)

for(i in 1:1000){
  s2_vec[i, ] <- 1/(rgamma(K_max, nun/2, (nun * s2n)/2))
  mu_vec[i, ] <- rnorm(K_max, mun, sqrt(s2_vec[i, ]/kn))
}

apply(s2_vec, 2, mean)


rm(list = ls())
elm <- list(result = matrix(NA, ncol = 3, nrow = 10), time = NA)
mylist_1 <- rep(list(elm), 2)
mylist_2 <- rep(list(elm), 2)
overall_result <- list(mylist_1, mylist_2)
overall_result[[1]]


rm(list = ls())
ci_true <- sample(rep(c(2, 3, 0), 10))
dat <- -(1:30)
K_max <- 10
log_likelihood(ci_true, dat, mu0_cluster = rep(0, K_max), lambda_cluster = rep(1, K_max), 
               a_sigma_cluster = rep(1, K_max), b_sigma_cluster = rep(1, K_max))


rm(list = ls())
set.seed(32134)
### Simulate the data
ci_true <- sample(1:5, 500, replace = TRUE)
dat <- rnorm(500, c(0, 7.5, 15, 25, 35)[ci_true], 1)
K_max <- 10

start_time <- Sys.time()
test_result <- SFDM_model(iter = 3000, K = K_max, init_assign = rep(0, 500), y = scale(dat, center = TRUE, scale = FALSE), 
                          mu0_cluster = rep(0, K_max), lambda_cluster = rep(1, K_max), 
                          a_sigma_cluster = rep(1, K_max), b_sigma_cluster = rep(1, K_max), 
                          xi_cluster = rep(1, K_max), a_theta = 1, b_theta = 1, 
                          launch_iter = 10, print_iter = 1000)
print(Sys.time() - start_time)

sp_status <- factor(test_result$split_or_merge)
levels(sp_status) <- c("Merge", "Split")

table(sp_status, test_result$sm_status)

table(salso(test_result$iter_assign[-c(1:1000), ]), ci_true)

plot(apply(test_result$iter_assign, 1, function(x) length(unique(x))), type = "l")

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
ci_true <- rep(0:0, 500)
dat <- rnorm(500, c(-10, -5, 5, 10)[ci_true + 1])
K <- 5

test <- SFDM_SM(K, rep(0, 500), dat, alpha_vec = c(rgamma(1, 1, 1), rep(0, 4)),
        mu0_cluster = rep(0, K), lambda_cluster = rep(1, K), 
        a_sigma_cluster = rep(1, K), b_sigma_cluster = rep(1, K), 
        xi_cluster = rep(1, K), launch_iter = 5, a_theta = 1, b_theta = 1)

test$old_lik
test$new_lik

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
