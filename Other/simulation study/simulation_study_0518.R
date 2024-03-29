rm(list = ls())

### FIX ME: --------------------------------------------------------------------
overall_seed <- 30184
n_para <- 5
scenario_now <- 2 ## Scenario
save_path <- NULL ## The location where we will save the result
save_name <- paste0(save_path, "simulation_", scenario_now, ".RData")
###: ---------------------------------------------------------------------------

### Required Packages: ---------------------------------------------------------
library(ClusterNormal)

library(cluster) ### For calculating silhouette and pam function
library(EMCluster) ### For EM algorithm
library(AntMAN) ### For finite of finite mixture
library(PReMiuM) ### For the Dirichlet Process
library(salso)
em_option <- .EMControl(short.iter = 1) ### Option for the EM algorithm

library(tidyverse)
library(mclustcomp)
library(foreach)
library(doParallel)
library(doRNG)
###: ---------------------------------------------------------------------------

### Note: ----------------------------------------------------------------------
### PReMiuM is no longer avialble on CRAN. You need to download from 
### https://www.jstatsoft.org/article/view/v064i07 .
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

### Function: Compute an average silhouette value for k-means/pam method
avg_sil <- function(data_list, clus_method, k) {
  clus_result <- NULL
  if(clus_method == "kmeans"){
    clus_result <- kmeans(scale(data_list$dat), centers = k)$cluster
  } else if(clus_method == "pam"){
    clus_result <- pam(scale(data_list$dat), k)$clustering
  }
  
  ss <- silhouette(clus_result, dist(scale(data_list$dat)))
  mean(ss[, 3])
}

### Function: Calculate the BIC for EM algorithm
k_EM_BIC <- function(data_list, k, em_opt){
  ### Initialize the model
  init_EM <- init.EM(scale(data_list$dat), nclass = k, EMC = em_opt,
                     stable.solution = TRUE, min.n = 1, min.n.iter = 10,
                     method = c("Rnd.EM"))
  ### Calculate BIC
  em.bic(scale(data_list$dat), init_EM)
}

###: ---------------------------------------------------------------------------

### Simulated the data
set.seed(overall_seed)
registerDoParallel(detectCores() - 1)
list_data <- foreach(i = 1:n_para) %dorng%{
  return(f_data_sim(overall_seed + i, scenario_now))
}
stopImplicitCluster()

### Choose the hyperparameters for each scenario based on the 
### sensitivity analysis
hyper_set <- list(list(K_max = 10, xi = 0.1, lambda = 1, a_sigma = 1, 
                       b_sigma = 1, b_theta = 1, sm_iter = 10),
                  list(K_max = 10, xi = 0.1, lambda = 1, a_sigma = 100, 
                       b_sigma = 1, b_theta = 1, sm_iter = 10),
                  list(K_max = 10, xi = 0.1, lambda = 1, a_sigma = 1, 
                       b_sigma = 1, b_theta = 1, sm_iter = 10),
                  list(K_max = 10, xi = 0.1, lambda = 1, a_sigma = 100, 
                       b_sigma = 1, b_theta = 1, sm_iter = 10))
hyper_set <- hyper_set[[scenario_now]]

### Begin the process
overall_time <- Sys.time()
result_test <- foreach(i = 1:n_para) %dorng%{
  
  ### Prepare for the algorithms
  result_mat <- matrix(NA, ncol = 7, nrow = 500)
  seed_num <- round(overall_seed/2) + i
  scale_dat <- scale(list_data[[i]]$dat)
  result_mat[, 1] <- list_data[[i]]$actual_clus
  
  set.seed(seed_num)
  
  ### K-Mean
  kmeans_start <- Sys.time()
  kmean_opt <- which.max(apply(data.frame(k = 2:hyper_set$K_max), 1, avg_sil, 
                               data_list = list_data[[i]], 
                               clus_method = "kmeans")) + 1
  result_mat[, 2] <- kmeans(scale_dat, kmean_opt)$cluster
  kmeans_time <- Sys.time() - kmeans_start
  
  ### PAM
  pam_start <- Sys.time()
  pam_opt <- which.max(apply(data.frame(k = 2:hyper_set$K_max), 1, avg_sil, 
                             data_list = list_data[[i]], 
                             clus_method = "pam")) + 1
  result_mat[, 3] <- pam(scale_dat, pam_opt)$cluster
  pam_time <- Sys.time() - pam_start
  
  ### EM
  EM_start <- Sys.time()
  EM_opt <- which.min(apply(data.frame(k = 2:hyper_set$K_max), 1, k_EM_BIC, 
                            data_list = list_data[[i]], em_opt = em_option)) + 1
  result_mat[, 4] <- emcluster(scale_dat, emobj = init.EM(scale_dat, 
                                                          nclass = EM_opt, 
                                                          EMC = em_option,
                                                          stable.solution = TRUE, 
                                                          min.n = 1, 
                                                          min.n.iter = 10, 
                                                          method = c("Rnd.EM")), 
                               EMC = em_option, assign.class = TRUE)$class
  EM_time <- Sys.time() - EM_start
  
  ### AntMAN
  AntMAN_MCMC <- AM_mcmc_parameters(niter = 1000, burnin = 500, thin = 1,
                                    verbose = 1, output = c("CI", "K"), 
                                    parallel = FALSE, output_dir = NULL)
  data_hyper <- AM_mix_hyperparams_uninorm(m0 = 0, k0 = 1, 
                                           nu0 = 1, 
                                           sig02 = 1)
  Kmax_dist <- AM_mix_components_prior_dirac(5)
  cluster_hyper <- AM_mix_weights_prior_gamma(a = hyper_set$xi, b = 1)
  AntMAN_start <- Sys.time()
  AntMAN_mod <- AntMAN::AM_mcmc_fit(y = as.numeric(scale_dat),
                                    ## initial_clustering = rep(1, 500),
                                    mix_kernel_hyperparams = data_hyper,
                                    ## mix_components_prior = Kmax_dist,
                                    mix_weight_prior = cluster_hyper,
                                    mcmc_parameters = AntMAN_MCMC)
  result_mat[, 5] <- as.numeric(salso(AM_clustering(AntMAN_mod), 
                                      maxNClusters = hyper_set$K_max))
  AntMAN_time <- Sys.time() - AntMAN_start
  
  ### DP
  DP_output_prefix <- paste0("DP_", seed_num)
  DP_start <- Sys.time()
  DP_mod <- profRegr(covNames = "dat_scale", data = data.frame(dat_scale = scale_dat), 
                     nSweeps = 500, nBurn = 500, yModel = "Categorical", xModel = "Normal", 
                     excludeY = TRUE, nClusInit = 1, seed = seed_num,
                     output = DP_output_prefix)
  file.remove(list.files(pattern = DP_output_prefix))
  DP_clus <- calcOptimalClustering(calcDissimilarityMatrix(DP_mod), 
                                   maxNClusters = hyper_set$K_max)
  result_mat[, 6] <- DP_clus$clustering
  DP_time <- Sys.time() - DP_start
  
  ### SFDM
  SFDM_start <- Sys.time()
  SFDM_result <- SFDM_model(1000, hyper_set$K_max, rep(1, 500), 
                       rep(hyper_set$xi, hyper_set$K_max), scale_dat, 
                       rep(0, hyper_set$K_max), rep(hyper_set$a_sigma, hyper_set$K_max), 
                       rep(hyper_set$b_sigma, hyper_set$K_max), rep(hyper_set$lambda, hyper_set$K_max), 
                       1, hyper_set$b_theta, hyper_set$sm_iter, 250)
  result_mat[, 7] <- as.numeric(salso(SFDM_result$iter_assign[-(1:500), ]))
  SFDM_time <- Sys.time() - SFDM_start
  
  ### Return the result
  comp_time <- list(k_means = kmeans_time, pam = pam_time, EM = EM_time,
                    AntMAN = AntMAN_time, DP = DP_time, SFDM = SFDM_time)
  n_clus = list(k_means = kmean_opt, pam = pam_opt, EM = EM_opt,
                AntMAN = length(unique(result_mat[, 5])),
                DP = length(unique(result_mat[, 6])),
                SFDM = length(unique(result_mat[, 7])))
  return(list(AntMAN_mod = AntMAN_mod, cluster = result_mat, time = comp_time, n_clus = n_clus))
}
stopImplicitCluster()

print(Sys.time() - overall_time)

### Save the result
save(result_test, file = save_name)


salso(result_test[[1]]$AntMAN_mod)
AM_salso(AntMAN::AM_clustering(result_test[[1]]$AntMAN_mod), loss = "VI")
 

