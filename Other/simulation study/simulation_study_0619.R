rm(list = ls())

### FIX ME: --------------------------------------------------------------------
overall_seed <- 31807
n_para <- 30
scenario_now <- 1 ## Scenario
alg_iter <- 10000
burn_in <- 5000
save_path <- NULL ## The location where we will save the result
scale_dat <- FALSE ## scale the data
save_name_analysis <- paste0(save_path, "simu_study_", scenario_now, 
                             "_scale_", scale_dat, ".RData") 
###: ---------------------------------------------------------------------------

### Required Packages: ---------------------------------------------------------
library(ClusterNormal)
library(tidyverse)
library(factoextra)
library(cluster)
library(EMCluster)
library(AntMAN)
library(dirichletprocess)
library(salso)

em_option <- .EMControl(short.iter = 1) ### Option for the EM algorithm

library(tidyverse)
library(mclustcomp)
library(foreach)
library(doParallel)
library(doRNG)
###: ---------------------------------------------------------------------------

### User-defined functions: ----------------------------------------------------
### Function: Simulating the data based on the scenario
f_data_sim <- function(sim_seed, scenario_index){
  
  ### place for storing result.
  actual_clus <- NULL
  dat <- NULL
  
  set.seed(sim_seed)
  
  if(! scenario_index %in% 1:4){
    warning("invalid scenario. we have only 4 scenarios")
  } else {
    if(scenario_index == 1){
      actual_clus <- sample(1:2, 500, replace = TRUE)
      dat <- rnorm(500, c(-5, 5)[actual_clus])
    } else if(scenario_index == 2){
      actual_clus <- sample(1:5, 500, replace = TRUE)
      dat <- rnorm(500, (c(0, 7.5, 15, 25, 35))[actual_clus])
    } else if(scenario_index == 3){
      actual_clus <- sample(1:2, 500, replace = TRUE)
      dat <- rnorm(500, c(-5, 5)[actual_clus], 3)
    } else {
      actual_clus <- sample(1:5, 500, replace = TRUE)
      dat <- rnorm(500, (c(0, 7.5, 15, 25, 35)[actual_clus])/2, 1)
    }
  }
  
  ### return the simulated data
  result <- data.frame(actual_clus, dat)
  return(result)
}

### Function: Compute average silhouette for k clusters
### https://uc-r.github.io/kmeans_clustering
avg_sil <- function(k, data_clus) {
  km.res <- kmeans(data_clus, centers = k)
  ss <- silhouette(km.res$cluster, dist(data_clus))
  mean(ss[, 3])
}

### Function: Calculate the BIC for EM algorithm
k_EM_BIC <- function(data_clus, k, em_opt){
  ### Initialize the model
  init_EM <- init.EM(data_clus, nclass = k, EMC = em_opt,
                     stable.solution = TRUE, min.n = 1, min.n.iter = 10,
                     method = c("Rnd.EM"))
  ### Calculate BIC
  em.bic(scale(data_list$dat), init_EM)
}

###: ---------------------------------------------------------------------------

### Simulated the data
set.seed(overall_seed)
registerDoParallel(15)
list_data <- foreach(i = 1:n_para) %dopar% {
  return(f_data_sim(overall_seed + i, scenario_now))
}
stopImplicitCluster()

### Start a simulation study
overall_start <- Sys.time()

registerDoParallel(15)
list_result <- foreach(i = 1:n_para) %dopar% {
  
  ### Get the data
  dat_used <- as.numeric(scale(list_data[[i]]$dat, 
                               center = scale_dat, scale = scale_dat))
  
  assign_result <- matrix(NA, nrow = 500, ncol = 7)
  colnames(assign_result) <- c("Kmeans", "PAM", "EM", "AntMAN", "DP", "SFDMM", "SFCMM")
  comp_time <- rep(NA, 7)
  
  ### K-mean
  k_means_sil <- rep(NA, 9)
  km_start <- Sys.time()
  for(i in 2:10){
    k_means_sil[(i-1)] <- avg_sil(i, dat_used)
  }
  km_method <- kmeans(dat_used, which.max(k_means_sil) + 1)
  km_time <- difftime(Sys.time(), km_start, units = "secs")
  assign_result[, 1] <- km_method$cluster
  comp_time[1] <- as.numeric(km_time)
  
  ### PAM
  pam_sil <- rep(NA, 9)
  pam_start <- Sys.time()
  for(i in 2:10){
    pam_sil[(i-1)] <- mean(silhouette(pam(dat_used, i))[, 3])
  }
  pam_method <- pam(dat_used, which.max(pam_sil) + 1)
  pam_time <- difftime(Sys.time(), pam_start, units = "secs")
  assign_result[, 2] <- pam_method$clustering
  comp_time[2] <- as.numeric(pam_time)
  
  ### EM
  em_option <- .EMControl(short.iter = 1)
  em_BIC <- rep(NA, 9)
  em_start <- Sys.time()
  for(i in 2:10){
    em_BIC[(i-1)] <- k_EM_BIC(data.frame(dat_used), i, em_option)
  }
  EM_opt <- which.min(em_BIC) + 1
  em_method <- emcluster(data.frame(dat_used), emobj = init.EM(data.frame(dat_used), nclass = EM_opt, 
                                                               EMC = em_option, stable.solution = TRUE,
                                                               min.n = 1, min.n.iter = 10, 
                                                               method = c("Rnd.EM")), 
                         EMC = em_option, assign.class = TRUE)$class
  em_time <- difftime(Sys.time(), em_start, units = "secs")
  assign_result[, 3] <- em_method
  comp_time[3] <- as.numeric(em_time)
  
  ### AntMAN
  AntMAN_MCMC <- AM_mcmc_parameters(niter = alg_iter, burnin = burn_in, thin = 1,
                                    verbose = 1, output = c("CI", "K"), 
                                    parallel = FALSE, output_dir = NULL)
  data_hyper <- AM_mix_hyperparams_uninorm(m0 = 0, k0 = 1, nu0 = 0.1, sig02 = 0.1)
  cluster_hyper <- AM_mix_weights_prior_gamma(a = 1, b = 1)
  am_start <- Sys.time()
  AntMAN_mod<- AntMAN::AM_mcmc_fit(y = dat_used, initial_clustering = rep(1, 500),
                                   mix_kernel_hyperparams = data_hyper,
                                   mix_weight_prior = cluster_hyper,
                                   mcmc_parameters = AntMAN_MCMC)
  AntMAN_method <- as.numeric(salso(AM_clustering(AntMAN_mod), maxNClusters = 10))
  am_time <- difftime(Sys.time(), am_start, units = "secs")
  assign_result[, 4] <- AntMAN_method
  comp_time[4] <- as.numeric(am_time)
  
  ### DP
  dp_mod <- DirichletProcessGaussian(as.matrix(dat_used), g0Priors = c(0, 1, 0.1, 0.1), alphaPriors = c(1, 1))
  dp_start <- Sys.time()
  dp_fit <- Fit(dp_mod, alg_iter, updatePrior = FALSE, progressBar = TRUE) 
  dp_clus <- matrix(NA, nrow = burn_in, ncol = 500)
  for(i in 1:burn_in){
    dp_clus[i, ] <- dp_fit$labelsChain[[(burn_in + i)]]
  }
  dp_assign <- as.numeric(salso(dp_clus, maxNClusters = 10))
  dp_time <- difftime(Sys.time(), dp_start, units = "secs")
  assign_result[, 5] <- dp_assign
  comp_time[5] <- as.numeric(dp_time)
  
  ### SFDMM
  sfdmm_start <- Sys.time()
  SFDMM_mod <- SFDMM_model(iter = alg_iter, K_max = 10, init_assign = rep(0, 500), 
                           y = dat_used, a0 = 0.1, b0 = 0.1, mu0 = 0, s20 = 100, 
                           xi0 = 1, a_theta = 1, b_theta = 1, launch_iter = 10, 
                           print_iter = 2500)
  SFDMM_assign <- as.numeric(salso(SFDMM_mod$iter_assign[-(1:burn_in), ], maxNClusters = 10))
  sfdmm_time <- difftime(Sys.time(), sfdmm_start, units = "secs")
  assign_result[, 6] <- SFDMM_assign
  comp_time[6] <- as.numeric(sfdmm_time)
  
  ### SFCMM
  sfcmm_start <- Sys.time()
  SFCMM_mod <- SFCMM_model(iter = alg_iter, K_max = 10, init_assign = rep(0, 500), 
                           y = dat_used, a0 = 0.1, b0 = 0.1, mu0 = 0, s20 = 100, 
                           xi0 = 1, launch_iter = 10, print_iter = 2500)
  SFCMM_assign <- as.numeric(salso(SFCMM_mod$iter_assign[-(1:burn_in), ], maxNClusters = 10))
  sfcmm_time <- difftime(Sys.time(), sfcmm_start, units = "secs")
  assign_result[, 7] <- SFCMM_assign
  comp_time[7] <- as.numeric(sfcmm_time)
  
  return(list(assign_result = assign_result, comp_time = comp_time))
  
}

stopImplicitCluster()
print(Sys.time() - overall_start)

### Save the result
final_list <- list(dat = list_data, result = list_result)
save(final_list, file = save_name_analysis)

