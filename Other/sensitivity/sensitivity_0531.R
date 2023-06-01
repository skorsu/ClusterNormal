rm(list = ls())

### FIX ME: --------------------------------------------------------------------
overall_seed <- 30184
n_para <- 30
scenario_now <- 1 ## Scenario
alg_iter <- 3000
save_path <- NULL ## The location where we will save the result
save_name_analysis <- paste0(save_path, "sensitivity_", scenario_now, ".RData") 
centered <- FALSE ## center the data
###: ---------------------------------------------------------------------------

### Required Packages: ---------------------------------------------------------
library(salso)
library(ggplot2)
library(ClusterNormal)
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

###: ---------------------------------------------------------------------------

### Sensitivity Analysis: ------------------------------------------------------
### Updated 5/31/2023: Using the original data and let the program save the raw 
###                    result as well.
### Updated 5/18/2023: Perform a sensitivity analysis for all four scenarios.

### Step 1: Simulate the data based on the scenarios
### Using the "f_data_sim" function.

### Simulate the data for paralleled
set.seed(overall_seed)
registerDoParallel(detectCores() - 1)
list_data <- foreach(i = 1:n_para) %dorng%{
  return(f_data_sim(overall_seed + i, scenario_now))
}
stopImplicitCluster()

### Step 2: Create a set of hyperparameter 
### Hyperparameter: (K, xi, lambda, a_sigma, b_sigma, b_theta, iter_launch)
### Default Case: (10, 0.1, 1, 1, 1 , 1, 1, 10)
### All scenarios share the same set of hyperparameters
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

### Step 3: Run the model paralleled
### Each parallel represents each setting. We will loop through each simulated data.
### Define global parameters

ci_init <- sample(0:0, 500, replace = TRUE)

### Start the algorithm
set.seed(31807)
overall_start <- Sys.time()
registerDoParallel(detectCores() - 2)

list_result <- foreach(i = 1:nrow(hyper_set)) %dorng%{
  
  ### Create an object for storing the result
  list_store <- rep(list(result = NA, time = NA, actual_clus = NA,
                         split_or_merge = NA, result_status = NA), n_para)
  
  ### Hyperparameters
  K <- hyper_set[i, "K"]
  xi_vec <- rep(hyper_set[i, "xi"], K)
  mu0_vec <- rep(0, K)
  a_sigma_vec <- rep(hyper_set[i, "as"], K)
  b_sigma_vec <- rep(hyper_set[i, "bs"], K)
  lambda_vec <- rep(hyper_set[i, "lambda"], K)
  a_theta <- hyper_set[i, "at"]
  b_theta <- hyper_set[i, "bt"]
  sm_iter <- hyper_set[i, "iter"]
  
  ### Matrix for storing the final result
  ### result_mat <- data.frame(matrix(NA, ncol = 7, nrow = n_para))
  ### colnames(result_mat) <- c("Run Time", "Clusters", "Jaccard", "VI", 
  ###                           "P(Merge|Accept)", "P(Split|Accept)", "P(Accept)")

  ### Loop through each data set
  for(t in 1:n_para){
    
    ### Data for each parallel
    list_store[[t]]$actual_clus <- list_data[[t]]$actual_clus
    ### clus_index <- list_data[[t]]$actual_clus
    scaled_dat <- scale(list_data[[t]]$dat, center = centered, scale = FALSE)
    start_time <- Sys.time()
    result <- SFDM_model(alg_iter, K, ci_init, scaled_dat, mu0_vec, lambda_vec,
                         a_sigma_vec, b_sigma_vec, xi_vec, a_theta, b_theta, 
                         sm_iter, 500)
    run_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    list_store[[t]]$time <- run_time
    list_store[[t]]$result <- result$iter_assign
    list_store[[t]]$split_or_merge <- result$split_or_merge
    list_store[[t]]$result_status <- result$sm_status
    
    ### clus_result <- as.numeric(salso(result$iter_assign[-(1:alg_burn_in), ],
    ###                                 maxNClusters = K))
    ### jacc_score <- mclustcomp(clus_index, clus_result, types = "jaccard")$scores
    ### vi_score <- mclustcomp(clus_index, clus_result, types = "vi")$scores
    
    ### result_status <- factor(result$sm_status)
    ### levels(result_status) <- c("Reject", "Accept")
    ### result_sm <- factor(result$split_or_merge)
    ### levels(result_sm) <- c("Merge", "Split")
    ### result_tab <- table(result_status, result_sm)
    ### prob_accept <- result_tab[2, ]/table(result_sm)
    
    ### result_mat[t, ] <- c(run_time, length(unique(clus_result)), jacc_score, 
    ###                      vi_score, prob_accept, mean(result$sm_status))
    
  }
  return(list_store)
}
stopImplicitCluster()
print("-----------------------------------------------------")
print(Sys.time() - overall_start)

### Step 4: Save the result
save(list_result, file = save_name_analysis)

###: ---------------------------------------------------------------------------

