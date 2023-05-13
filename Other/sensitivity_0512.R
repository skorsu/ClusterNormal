rm(list = ls())

### Required Packages
library(salso)
library(ggplot2)
library(ggrepel)
library(gridExtra)
library(ClusterNormal)
library(tidyverse)
library(mclustcomp)
library(foreach)
library(doParallel)
library(doRNG)

### Function for calculating the number of active cluster
n_unique <- function(vec){
  length(unique(vec))
}

### Sensitivity Analysis
### Use the same dataset across the different sets of hyperparameter.
### (3.1) and (3.3) in draft. (sigma = 1 and 3 respectively.)
set.seed(2)
clus_actual <- rep(1:2, 250)[sample(1:500)]
dat <- rnorm(500, c(5, -5)[clus_actual], 3)
data.frame(clus_actual, dat) %>%
  ggplot(aes(x = dat, fill = factor(clus_actual))) +
  geom_histogram(alpha=0.6, position = 'identity', bins = 50) +
  scale_fill_manual(values=c("#69b3a2", "#404080")) +
  theme_bw() +
  labs(fill="", x = "Data", y = "")

### Set of hyperparameter (K, xi, lambda, a_sigma, b_sigma, b_theta, iter_launch)
### Default Case: (10, 0.1, 1, 1, 1 , 1, 1, 10)
hyper_set <- data.frame(c(10, 0.1, 1, 1, 1, 1, 1, 10), 
                        c(5, 0.1, 1, 1, 1, 1, 1, 10),
                        c(20, 0.1, 1, 1, 1, 1, 1, 10),
                        c(10, 0.01, 1, 1, 1, 1, 1, 10),
                        c(10, 1, 1, 1, 1, 1, 1, 10),
                        c(10, 0.1, 0.1, 1, 1, 1, 1, 10),
                        c(10, 0.1, 10, 1, 1, 1, 1, 10),
                        c(10, 0.1, 1, 10, 1, 1, 1, 10), 
                        c(10, 0.1, 1, 100, 1, 1, 1, 10),
                        c(10, 0.1, 1, 1, 10, 1, 1, 10),
                        c(10, 0.1, 1, 1, 100, 1, 1, 10),
                        c(10, 0.1, 1, 1, 1, 1, 4, 10),
                        c(10, 0.1, 1, 1, 1, 1, 9, 10),
                        c(10, 0.1, 1, 1, 1, 1, 1, 5),
                        c(10, 0.1, 1, 1, 1, 1, 1, 25)) %>% t()

rownames(hyper_set) <- NULL
colnames(hyper_set) <- c("K", "xi", "lambda", "as", "bs", "at", "bt", "iter")

### Run the model
set.seed(45785)
overall_start <- Sys.time()
registerDoParallel(detectCores() - 1)
result_mat <- foreach(i = 1:(dim(hyper_set)[1])) %dorng%{
  K <- hyper_set[i, "K"]
  iter <- 1000
  ci_init <- sample(1:1, 500, replace = TRUE)
  xi_vec <- rep(hyper_set[i, "xi"], K)
  mu0_vec <- rep(0, K)
  a_sigma_vec <- rep(hyper_set[i, "as"], K)
  b_sigma_vec <- rep(hyper_set[i, "bs"], K)
  lambda_vec <- rep(hyper_set[i, "lambda"], K)
  a_theta <- hyper_set[i, "at"]
  b_theta <- hyper_set[i, "bt"]
  sm_iter <- hyper_set[i, "iter"]
  
  start_time <- Sys.time()
  result <- SFDM_model(iter, K, ci_init, xi_vec, scale(dat), mu0_vec, 
                       a_sigma_vec, b_sigma_vec, lambda_vec, a_theta, b_theta, 
                       sm_iter, 250)
  run_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
  
  clus_result <- as.numeric(salso(result$iter_assign[-(1:500), ]))
  table(clus_actual, salso(result$iter_assign[-(1:500), ], maxNClusters = K))
  jacc_score <- mclustcomp(clus_actual, clus_result, types = "jaccard")$scores
  vi_score <- mclustcomp(clus_actual, clus_result, types = "vi")$scores
  
  result_status <- factor(result$sm_status)
  levels(result_status) <- c("Reject", "Accept")
  result_sm <- factor(result$split_or_merge)
  levels(result_sm) <- c("Merge", "Split")
  result_tab <- table(result_status, result_sm)
  prob_accept <- result_tab[2, ]/table(result_sm)
  
  active_clus <- apply(result$iter_assign, 1, n_unique)
  
  return(list(run_time = run_time, jacc = jacc_score, vi = vi_score,
              prob = c(prob_accept, mean(result$sm_status)), 
              n_unique = active_clus))
}
stopImplicitCluster()
Sys.time() - overall_start

### Summary
run_time_vec <- rep(NA, dim(hyper_set)[1])
jacc_vec <- rep(NA, dim(hyper_set)[1])
vi_vec <- rep(NA, dim(hyper_set)[1])
accept_prob <- matrix(NA, ncol = 3, nrow = dim(hyper_set)[1])

for(i in 1:15){
  run_time_vec[i] <- result_mat[[i]]$run_time
  accept_prob[i, ] <- result_mat[[i]]$prob
  jacc_vec[i] <- result_mat[[i]]$jacc
  vi_vec[i] <- result_mat[[i]]$vi
}

colnames(accept_prob) <- c("Merge", "Split", "Overall")

dat1 <- data.frame(comp_time = run_time_vec, ac_prob = accept_prob[, 3], 
                   jacc_vec, vi_vec)

cbind(hyper_set, dat1)
p1 <- ggplot(dat1, aes(x = comp_time, y = ac_prob, color = factor(group_launch))) +
  geom_point() +
  theme_bw() +
  labs(x = "Computational Time (Second)", y = "Acceptance Probability", 
       color = "Number of Launch Iteration") +
  xlim(0, 60) +
  ylim(0, 1) +
  theme(legend.position = "bottom")
  
dat2 <- data.frame(jacc = jacc_vec, vi = vi_vec)
p2 <- dat2 %>% pivot_longer(cols = c("jacc", "vi"),
                      names_to = "Metric",
                      values_to = "Value") %>%
  ggplot(aes(x = Metric, y = Value)) +
  geom_boxplot() +
  geom_dotplot(binaxis = 'y', stackdir = 'center', dotsize = 0.5, binwidth = 0.05) +
  scale_x_discrete(labels = c("Jaccard Index", "VI")) +
  theme_bw() +
  labs(x = "Metric", y = " ")

grid.arrange(p1, p2, nrow = 1)
