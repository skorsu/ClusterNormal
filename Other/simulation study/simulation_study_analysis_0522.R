rm(list = ls())

### FIX ME: --------------------------------------------------------------------
scenario_now <- 2
load_path <- "/Users/kevinkvp/Desktop/" ## The location where we will save the result
load_name <- paste0(load_path, "simulation_", scenario_now, ".RData")
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
### Function: Calculate the Jaccard Index from the result
jacc_result <- function(cluster_mat){
  ### Actual cluster
  actual_clus <- cluster_mat[, 1]
  result_assign <- cluster_mat[, -1]
  
  result_quan <- apply(result_assign, 2, mclustcomp, y = actual_clus, 
                       type = c("jaccard", "vi"))
  result_quan
}

###: ---------------------------------------------------------------------------

### Import the result
load(load_name)
n_method <- 6
time_mat <- matrix(NA, nrow = length(result_test), ncol = n_method)
n_clus_mat <- matrix(NA, nrow = length(result_test), ncol = n_method)

for(i in 1:length(result_test)){
  ### Time
  time_mat[i, ] <- as.numeric(result_test[[i]]$time)
  
  ### Number of the cluster
  n_clus_mat[i, ] <- as.numeric(result_test[[i]]$n_clus)
}

as.numeric(result_test[[1]]$n_clus)

jacc_result(result_test[[1]]$cluster)

table(result_test[[1]]$cluster[, 5], result_test[[1]]$cluster[, 1])


















