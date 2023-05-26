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

### User-defined functions: ----------------------------------------------------
rm(list = ls())
n_unique <- function(clus_vec){
  length(unique(clus_vec))
}

### Sandbox: -------------------------------------------------------------------
rm(list = ls())

set.seed(31)
dat <- rnorm(50)
ci_assign <- rep(1:2, 25)
K <- 5
mu0_vec <- rep(0, K)
lambda_vec <- rep(1, K)
a_sigma_vec <- rep(1, K)
b_sigma_vec <- rep(1, K)
xi_vec <- rep(0.1, K)
alpha_now <- c(rgamma(2, 0.01, 1), rep(0, 3))

SFDM_SM(K, ci_assign, alpha_now, xi_vec, dat, mu0_vec, a_sigma_vec,
        b_sigma_vec, lambda_vec, 1, 1, 10)


log_posterior(dat_new, dat, ci, mu0_vec, lambda_vec, a_sigma_vec, b_sigma_vec)

an <- 1 + 1
Vn_inv <- 1 + 2
mn <- sum(dat)/Vn_inv
bn <- 1 + (0.5 * var(dat)) + (0.5 * (2/3) * (mean(dat)^2))

scale_t <- sqrt(bn * (1 + (1/Vn_inv)) / an)

log((1/scale_t) * dt((dat_new - mn)/scale_t, 2*an))




set.seed(12)
dat_now <- rnorm(1)
dat_old <- rnorm(100)
mu0_vec <- rep(0, 5)
lambda_vec <- rep(1, 5)
a_sigma_vec <- rep(1, 5)
b_sigma_vec <- rep(1, 5)

log_posterior_predict(dat_now, dat_old, 1, mu0_vec, lambda_vec, a_sigma_vec, b_sigma_vec)

an <- 1 + (100/2)
vn_inv <- 1 + 100
mn <- sum(dat_old)/vn_inv
bn <- 1 + (0.5 * sum(dat_old^2) - mn^2 * vn_inv)

tdf <- 2 * an
scale_tt <- sqrt(bn * (1 + (1/vn_inv))/an)

dt((dat_now - mn)/scale_tt, df = tdf, log = TRUE) - log(scale_tt)

set.seed(12)
K <- 5
n <- 50
ci_true <- c(sample(c(2, 5, 1), n, replace = TRUE), 4)
dat <- rnorm(n+1, c(-5, -2.5, 0, 2.5, 5)[ci_true])
mu0_vec <- rep(0, K)
a_sigma_vec <- rep(5:1, 1)
b_sigma_vec <- (1:5)/10
lambda_vec <- rep(1:5, 1)

log_likelihood(dat, ci_true, mu0_vec, lambda_vec, a_sigma_vec, b_sigma_vec)
log_marginal(ci_true, dat, a_sigma_vec, b_sigma_vec, lambda_vec, mu0_vec)

-43.0332-26.0592-3.79354-42.1625

log_marginal_new(ci_true, dat, a_sigma_vec, b_sigma_vec, lambda_vec, mu0_vec)
sum(ci_true == 2)

data.frame(ci_true) %>%
  group_by(ci_true) %>%
  summarise(n = n()) %>%
  cbind(lb = lambda_vec[-3], a = a_sigma_vec[-3]) %>%
  group_by(ci_true) %>%
  summarise(step5 = (-(n/2) * log(2 * pi)) + (0.5 * log(lb + n)) - (0.5 * log(lb)) +
              lgamma(a + (0.5 * n)) - lgamma(a))

0.5 * ((1 * sum(ci_true == 4))/(1 + sum(ci_true == 4))) * (0 - mean(dat[ci_true == 4]))^2

dat_list <- list(list(NA, NA, NA, NA), list(NA, NA, NA, NA), 
                 list(NA, NA, NA, NA), list(NA, NA, NA, NA))
dat_list[[1]][[1]] <- c(-20, -10, 0, 10, 20)/2
dat_list[[2]][[1]] <- c(0, 7.5, 15, 25, 35)/2
dat_list[[3]][[1]] <- c(0, 7.5, 15, 25, 35)
dat_list[[4]][[1]] <- c(0, 7.5, 15, 25, 35)

dat_list[[1]][[2]] <- 1
dat_list[[2]][[2]] <- 1
dat_list[[3]][[2]] <- 1
dat_list[[4]][[2]] <- 3

dat_list[[1]][[3]] <- "Mean: c(-20, -10, 0, 10, 20)/2 Var = 1"
dat_list[[2]][[3]] <- "Mean: c(0, 7.5, 15, 25, 35)/2 Var = 1"
dat_list[[3]][[3]] <- "Mean: c(0, 7.5, 15, 25, 35) Var = 1"
dat_list[[4]][[3]] <- "Mean: c(0, 7.5, 15, 25, 35) Var = 9"

set.seed(2341)
for(i in 1:4){
  actual_clus <- sample(1:5, 500, replace = TRUE)
  dat <- rnorm(500, (dat_list[[i]][[1]])[actual_clus], dat_list[[i]][[2]])
  dat_list[[i]][[4]] <- data.frame(dat, actual_clus) %>%
    ggplot(aes(x = dat, group = factor(actual_clus))) +
    geom_histogram(aes(y = after_stat(density)), bins = 50, alpha = 0.25) +
    geom_density(linewidth = 0.75) +
    theme_bw() +
    labs(title = dat_list[[i]][[3]], x = "Data", y = "Density") +
    theme(axis.text = element_text(size = 16),
          axis.title = element_text(size = 20),
          plot.title = element_text(size = 28))
}

grid.arrange(dat_list[[1]][[4]], dat_list[[2]][[4]], 
             dat_list[[3]][[4]])

load(paste0("/Users/kevin-imac/Desktop/Result/Sensitivity/archive/sensitivity_2.RData"))
sum_tab(list_result)


set.seed(56327189)
ci_true <- sample(1:5, 500, replace = TRUE)
dat <- rnorm(500, c(0, 7.5, 15, 25, 35)[ci_true], 1)
ggplot(data.frame(x = ci_true, y = dat), aes(x = y, group = x)) +
  geom_density()
hist(dat, breaks = 100)
data.frame(x = ci_true, y = scale(dat)) %>%
  group_by(x) %>%
  summarise(mean = mean(y), var = var(y))

clus_init <- rep(1:1, 500)
K <- 10
xi_vec <- rep(0.1, K)
mu0_vec <- rep(0, K)
a_sigma_vec <- rep(1, K)
b_sigma_vec <- rep(1, K)
lambda_vec <- rep(1, K)
a_theta <- 1
b_theta <- 1
sm_iter <- 10

model <- SFDM_model(1000, K, clus_init, xi_vec, scale(dat), mu0_vec, 
                    a_sigma_vec, b_sigma_vec, lambda_vec, a_theta, b_theta, 
                    sm_iter, 250)

hist(model$log_A)

table(model$sm_status, model$split_or_merge)

table(salso(model$iter_assign[-c(1:500), ]), ci_true)

mclustcomp(as.numeric(salso(model$iter_assign[-c(1:500), ])), ci_true)


