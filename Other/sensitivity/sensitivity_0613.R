rm(list = ls())

### FIX ME: --------------------------------------------------------------------
overall_seed <- 31807
n_para <- 30
scenario_now <- 1 ## Scenario
alg_iter <- 10000
burn_in <- 5000
save_path <- NULL ## The location where we will save the result
scale_dat <- FALSE ## scale the data
save_name_analysis <- paste0(save_path, "sensitivity_", scenario_now, 
                             "_scale_", scale_dat, ".RData") 
###: ---------------------------------------------------------------------------

### Required Packages: ---------------------------------------------------------
library(salso)
library(ClusterNormal)
library(tidyverse)
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
registerDoParallel(25)
list_data <- foreach(i = 1:n_para) %dopar%{
  return(f_data_sim(overall_seed + i, scenario_now))
}
stopImplicitCluster()

### Step 2: Create a set of hyperparameter 
### Hyperparameter: (K_max, s20, as, bs, xi, a_theta, b_theta, iter_launch)
### Default Case: (10, 10, 1, 1, 0.1, 1, 1, 10)
### All scenarios share the same set of hyperparameters
hyper_set <- data.frame(c(10, 10, 1, 1, 0.1, 1, 1, 10), 
                        c(5, 10, 1, 1, 0.1, 1, 1, 10),
                        c(20, 10, 1, 1, 0.1, 1, 1, 10),
                        c(10, 1, 1, 1, 0.1, 1, 1, 10),
                        c(10, 100, 1, 1, 0.1, 1, 1, 10),
                        c(10, 10, 0.1, 1, 0.1, 1, 1, 10),
                        c(10, 10, 10, 1, 0.1, 1, 1, 10),
                        c(10, 10, 1, 0.1, 0.1, 1, 1, 10),
                        c(10, 10, 1, 10, 0.1, 1, 1, 10),
                        c(10, 10, 1, 1, 1, 1, 1, 10), 
                        c(10, 10, 1, 1, 0.01, 1, 1, 10),
                        c(10, 10, 1, 1, 0.1, 4, 1, 10),
                        c(10, 10, 1, 1, 0.1, 9, 1, 10),
                        c(10, 10, 1, 1, 0.1, 1, 4, 10),
                        c(10, 10, 1, 1, 0.1, 1, 9, 10),
                        c(10, 10, 1, 1, 0.1, 1, 1, 5),
                        c(10, 10, 1, 1, 0.1, 1, 1, 25)) %>% t()

rownames(hyper_set) <- NULL
colnames(hyper_set) <- c("K_max", "s20", "as", "bs", "xi", "at", "bt", "iter")

### Step 3: Run the model paralleled
### Each parallel represents each setting. We will loop through each simulated data.
### Define global parameters

ci_init <- sample(0:0, 500, replace = TRUE)

### Start the algorithm
set.seed(overall_seed)
overall_start <- Sys.time()
registerDoParallel(25)

x <- foreach(i = 1:nrow(hyper_set)) %:% ## Hyperparameters
  foreach(t = 1:n_para) %dopar% { ## Dataset 
    
    dat <- as.numeric(scale(list_data[[t]]$dat, 
                            center = scale_dat, scale = scale_dat))
    start_time <- Sys.time()
    model <- SFDMM_model(iter = 10000, K_max = hyper_set[i, "K_max"], 
                         init_assign = rep(0, 500), y = dat, a0 = hyper_set[i, "as"],
                         b0 = hyper_set[i, "bs"], mu0 = 0, s20 = hyper_set[i, "s20"],
                         xi0 = hyper_set[i, "xi"], a_theta = hyper_set[i, "at"], 
                         b_theta = hyper_set[i, "bt"], launch_iter = hyper_set[i, "iter"],
                         print_iter = 2000)
    comp_time <- difftime(Sys.time(), start_time, units = "secs")
    
    ### Cluster assignment (from salso)
    result_salso <- as.numeric(salso(model$iter_assign[-(1:burn_in), ], 
                                     maxNClusters = hyper_set[i, "K_max"]))
    
    ### Thining (for storing the result)
    assign_iter <- model$iter_assign[(1:2000)*5, ]
    
    ### Result
    list(assign_iter = assign_iter, result_salso = result_salso, 
         comp_time = comp_time, sm_status = model$split_or_merge,
         accept_sm = model$sm_status)
  }

stopImplicitCluster()
print("-----------------------------------------------------")
print(Sys.time() - overall_start)

final_list <- list(data_set = list_data, result = x)

### Step 4: Save the result
save(final_list, file = save_name_analysis)

###: ---------------------------------------------------------------------------
