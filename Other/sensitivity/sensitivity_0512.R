rm(list = ls())

### Required Packages
library(salso)
library(ggplot2)
library(ClusterNormal)
library(tidyverse)
library(mclustcomp)
library(foreach)
library(doParallel)
library(doRNG)

### User-defined functions
### Function: Simulating the data based on the scenario
f_data_sim <- function(sim_seed, actual_K, overlap){
  ### place for storing result.
  actual_clus <- NULL
  dat <- NULL
  
  ### simulate the data
  set.seed(sim_seed)
  if(actual_K == 2){ ### Scenario 1 and 3
    actual_clus <- rep(1:2, 250)[sample(1:500)]
    if(overlap == FALSE){
      ## Scenario 1
      print("Scenario 1")
      dat <- rnorm(500, c(5, -5)[actual_clus], 1)
    } else {
      ### Scenario 3
      print("Scenario 3")
      dat <- rnorm(500, c(5, -5)[actual_clus], 3)
    }
  } else if(actual_K == 5){ ### Scenario 2 and 4
    actual_clus <- rep(1:5, 100)[sample(1:500)]
    if(overlap == FALSE){
      ## Scenario 2
      print("Scenario 2")
      dat <- rnorm(500, c(-100, -50, 0, 50, 100)[actual_clus], 1)
    } else {
      ### Scenario 4
      print("Scenario 4")
      dat <- rnorm(500, c(-10, -5, 0, 20, 40)[actual_clus], 
                   c(1.5, 1.5, 1.5, 3, 3)[actual_clus])
    }
  } else {
    warning("invalid values of the actual clusters. (actual_K)")
  }
  
  ### return the simulated data
  result <- data.frame(actual_clus, dat)
  return(result)
}

### Sensitivity Analysis
### Updated 3/14/2023: Perform a sensitivity analysis for all four scenarios.

### Step 1: Simulate the data based on the scenarios
### Using the "f_data_sim" function.

n_cluster <- 2 ## **FIX HERE: We can choose only 2 or 5.
group_overlap <- FALSE ## **FIX HERE

### Simulate the data for paralleled
n_para <- 30
set.seed(31807)
registerDoParallel(detectCores() - 1)
list_data <- foreach(i = 1:n_para) %dorng%{
  return(f_data_sim(2, n_cluster, group_overlap))
}
stopImplicitCluster()

### (Optional) Step O1: Create a plot to see the data
### Choose the color for each group
optional_plot <- FALSE
if(optional_plot == TRUE){
  col_code <- c("salmon", "#321aba", "#69b3a2", "#908121", "#E2134A")
  
  ggplot(list_data[[1]], aes(x = dat, fill = factor(actual_clus))) +
    geom_histogram(alpha = 0.6, position = 'identity', bins = 50) +
    scale_fill_manual(values = col_code[unique(list_data[[1]]$actual_clus)]) +
    theme_bw() +
    labs(fill="", x = "Data", y = "")
}

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

iter <- 1000
ci_init <- sample(1:1, 500, replace = TRUE)

### Start the algorithm
set.seed(31807)
overall_start <- Sys.time()
registerDoParallel(detectCores() - 5)
list_result <- foreach(i = 1:nrow(hyper_set)) %dorng%{
  
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
  result_mat <- data.frame(matrix(NA, ncol = 6, nrow = n_para))
  colnames(result_mat) <- c("Run Time", "Jaccard", "VI", "P(Merge|Accept)", 
                            "P(Split|Accept)", "P(Accept)")

  ### Loop through each data set
  for(t in 1:n_para){
    
    ### Data for each parallel
    clus_index <- list_data[[t]]$actual_clus
    scaled_dat <- scale(list_data[[t]]$dat)
    
    start_time <- Sys.time()
    result <- SFDM_model(iter, K, ci_init, xi_vec, scaled_dat, mu0_vec, 
                         a_sigma_vec, b_sigma_vec, lambda_vec, a_theta, b_theta, 
                         sm_iter, 250)
    run_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    
    clus_result <- as.numeric(salso(result$iter_assign[-(1:500), ]))
    jacc_score <- mclustcomp(clus_index, clus_result, types = "jaccard")$scores
    vi_score <- mclustcomp(clus_index, clus_result, types = "vi")$scores
    
    result_status <- factor(result$sm_status)
    levels(result_status) <- c("Reject", "Accept")
    result_sm <- factor(result$split_or_merge)
    levels(result_sm) <- c("Merge", "Split")
    result_tab <- table(result_status, result_sm)
    prob_accept <- result_tab[2, ]/table(result_sm)
    
    result_mat[t, ] <- c(run_time, jacc_score, vi_score, prob_accept, 
                         mean(result$sm_status))
    
  }
  return(result_mat)
}
stopImplicitCluster()
print(Sys.time() - overall_start)

### Step 4: Save the result
save(list_result, file = "simu_result_scenario_x.RData") ## **FIX HERE

