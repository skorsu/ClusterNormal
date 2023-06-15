rm(list = ls())

### FIX ME: --------------------------------------------------------------------
scenario_now <- 3 ## Scenario
scale_dat <- FALSE ## scale the data
file_path <- "/Users/kevin-imac/Desktop/Result/Sensitivity/"
save_name_analysis <- paste0(file_path, "sensitivity_", scenario_now, 
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

### Function: Merge all 15 cases into one table
sum_tab <- function(res_list){
  
  row_name <- c("Default", "$\\\\text{K}_{\\\\text{max}} = 5$", 
                "$text{K}_{\\text{max}} = 20$", "$\\\\sigma_0^2 = 1$", 
                "$\\sigma_0^2 = 100$", "a_{\\sigma} = 0.1", "a_{\\sigma} = 10",
                "b_{\\sigma} = 0.1", "b_{\\sigma} = 10", "\\xi = 1", 
                "\\xi = 0.01", "a_{\\theta} = 4", "a_{\\theta} = 9", 
                "b_{\\theta} = 4", "b_{\\theta} = 9", 
                "$\\text{sm}_{\\text{iter}} = 5$", 
                "$\\text{sm}_{\\text{iter}} = 25$")
  
  result_1 <- res_list[, 1:5]
  colnames(result_1) <- c("Time", "Adj Rand", "Jaccard", "VI", "Active Clusters")
  
  result_2 <- res_list[, 6:8]
  colnames(result_2) <- c("P(Accept)", "P(Accept|Merge)", "P(Accept|Split)")
  
  xtable(result_1) %>% print()
  print("----------------------------------")
  xtable(result_2) %>% print()

}
###: ---------------------------------------------------------------------------

### Import the result
load(save_name_analysis)

### Calculate the quantities
sum_table <- matrix(NA, nrow = 17, ncol = 8)

for(k in 1:17){
  
  result_quan <- matrix(NA, nrow = n_param, ncol = 8)
  for(i in 1:n_param){
    result_quan[i, 1] <- final_list$result[[k]][[i]]$comp_time
    result_quan[i, 2:4] <- mclustcomp(final_list$data_set[[i]]$actual_clus, 
                                      final_list$result[[k]][[i]]$result_salso,
                                      type = c("adjrand", "jaccard", "vi"))[, 2]
    result_quan[i, 5] <- length(unique(final_list$result[[k]][[i]]$result_salso))
    result_quan[i, 6] <- mean(final_list$result[[k]][[i]]$accept_sm)
    result_quan[i, 7] <- mean(final_list$result[[k]][[i]]$accept_sm[final_list$result[[k]][[i]]$sm_status == 0])
    result_quan[i, 8] <- mean(final_list$result[[k]][[i]]$accept_sm[final_list$result[[k]][[i]]$sm_status == 1])
  }
  
  sum_table[k, ] <- apply(result_quan, 2, bal_quan)
  
}

sum_tab(sum_table)



