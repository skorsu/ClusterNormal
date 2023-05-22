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


set.seed(143)
## c(-20, -10, 0, 10, 20)/2
ci_true <- sample(1:5, 500, replace = TRUE)
dat <- rnorm(500, c(0, 7.5, 15, 25, 35)[ci_true], 3)
ggplot(data.frame(x = ci_true, y = dat), aes(x = y, group = x)) +
  geom_density()
hist(dat, breaks = 100)
data.frame(x = ci_true, y = scale(dat)) %>%
  group_by(x) %>%
  summarise(mean = mean(y), var = var(y))

clus_init <- rep(1:500, 1)
K <- 500
xi_vec <- rep(0.1, K)
mu0_vec <- rep(0, K)
a_sigma_vec <- rep(100, K)
b_sigma_vec <- rep(1, K)
lambda_vec <- rep(1, K)
a_theta <- 1
b_theta <- 1
sm_iter <- 10

model <- SFDM_model(1000, K, clus_init, xi_vec, scale(dat), mu0_vec, 
                    a_sigma_vec, b_sigma_vec, lambda_vec, a_theta, b_theta, 
                    sm_iter, 250)

table(salso(model$iter_assign[-c(1:500), ]), ci_true)

apply(model$iter_assign, 1, n_unique)

(model$iter_assign[100, ])

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
