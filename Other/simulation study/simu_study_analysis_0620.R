rm(list = ls())

### FIX ME: --------------------------------------------------------------------
scenario_now <- 4 ## Scenario
scale_dat <- FALSE ## scale the data
file_path <- "/Users/kevin-imac/Desktop/Simulation_Study/"
save_name_analysis <- paste0(file_path, "simu_study_", scenario_now, 
                             "_scale_", scale_dat, ".RData") 
n_param <- 30
###: ---------------------------------------------------------------------------

### Required Packages: ---------------------------------------------------------
library(xtable)
library(tidyverse)
library(mclustcomp)
###: ---------------------------------------------------------------------------

### user-defined functions: ----------------------------------------------------
### Function: Summary Quantities "Mean (SD)".
bal_quan <- function(num_vec, rounding = 4){
  mean_val <- round(mean(num_vec), 4)
  sd_val <- round(sd(num_vec), 4)
  paste0(mean_val, " (", sd_val, ")")
}

### Function: Create the latex final result
sum_tab <- function(res_mat){
  
  row_name <- c("K-means", "PAM", "EM", "MFM", "DP",
                "SFDMM", "SFCMM")
  col_name <- c("Time", "Adj Rand", "Jaccard", "VI", "Active Clusters")
  
  colnames(res_mat) <- col_name
  rownames(res_mat) <- row_name
  
  # xtable(result_2) %>% print()
  
  res_mat

}
###: ---------------------------------------------------------------------------

### Import the result
load(save_name_analysis)

time_mat <- matrix(NA, nrow = n_param, ncol = 7)
jaac_mat <- matrix(NA, nrow = n_param, ncol = 7)
rand_mat <- matrix(NA, nrow = n_param, ncol = 7)
vi_mat <- matrix(NA, nrow = n_param, ncol = 7)
active_num <- matrix(NA, nrow = n_param, ncol = 7)

for(i in 1:n_param){
  
  ### Get the number of the active cluster
  active_num[i, ] <- apply(final_list$result[[i]]$assign_result, 2, 
                           function(x){length(unique(x))})
  
  ### Jaccard
  jaac_mat[i, ] <- apply(final_list$result[[i]]$assign_result, 2, 
                         function(x){mclustcomp(x, y = final_list$dat[[i]]$actual_clus, 
                                                types = "jaccard")[, 2]}) %>% as.numeric()
  
  ### Adj Rand Index
  rand_mat[i, ] <- apply(final_list$result[[i]]$assign_result, 2, 
                         function(x){mclustcomp(x, y = final_list$dat[[i]]$actual_clus, 
                                                types = "adjrand")[, 2]}) %>% as.numeric()
  ### VI
  vi_mat[i, ] <- apply(final_list$result[[i]]$assign_result, 2, 
                       function(x){mclustcomp(x, y = final_list$dat[[i]]$actual_clus, 
                                              types = "vi")[, 2]}) %>% as.numeric()
  ### Get the computational time
  time_mat[i, ] <- final_list$result[[i]]$comp_time
}

data.frame("Time" = apply(time_mat, 2, bal_quan),
           "Rand" = apply(rand_mat, 2, bal_quan),
           "Jaccard" = apply(jaac_mat, 2, bal_quan),
           "VI" = apply(vi_mat, 2, bal_quan),
           "Ac" = apply(active_num, 2, bal_quan)) %>% sum_tab() %>% xtable()



