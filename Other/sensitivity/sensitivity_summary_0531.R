rm(list = ls())

### Setting
burn_in <- 2000

### Import the data
path <- "/Users/kevin-imac/Desktop/"
load(paste0(path, "sensitivity_2_center_TRUE.RData"))

### Libraries
library(salso)
library(tidyverse)
library(mclustcomp)
library(ggplot2)
library(foreach)
library(doParallel)

### User-defined function
### Function: Calculate the unique number of active cluster
n_unique <- function(cvec){
  length(unique(cvec))
}

### Function: Summary Quantities "Mean (SD)".
bal_quan <- function(num_vec, rounding = 4){
  mean_val <- round(mean(num_vec), 4)
  sd_val <- round(sd(num_vec), 4)
  paste0(mean_val, " (", sd_val, ")")
}

### Function: Merge all 15 cases into one table
sum_tab <- function(res_list, rr = 4){
  result <- data.frame(matrix(NA, nrow = 15, ncol = 8))
  for(i in 1:15){
    result[i, ] <- apply(res_list[[i]], 2, bal_quan, rounding = rr)
  }
  colnames(result) <- c("Time", "Adj Rand", "Jaccard", "VI",
                        "P(Accept)", "P(Accept|Merge)", "P(Accept|Split)", "# Cluster")
  xtable(result)
}

start_collect <- Sys.time()
registerDoParallel(detectCores() - 1)
result_list <- foreach(c = 1:15) %dopar%{
  result_mat <- matrix(NA, nrow = 30, ncol = 8)
  colnames(result_mat) <- c("Time", "Adj Rand", "Jaccard", "VI",
                            "P(Accept)", "P(Accept|Merge)", "P(Accept|Split)", "# Cluster")
  for(i in 1:30){
    ci_true <- list_result[[c]][[i]]$actual_clus
    ci_result <- as.numeric(salso(list_result[[c]][[i]]$result[-(1:burn_in), ]))
    result_mat[i, 1] <- list_result[[c]][[i]]$time
    result_mat[i, 2:4] <- mclustcomp(ci_true, ci_result)[c(1, 5, 22), 2]
    result_mat[i, 5] <- mean(list_result[[c]][[i]]$result_status)
    sm <- factor(list_result[[c]][[i]]$split_or_merge)
    levels(sm) <- c("Merge", "Split")
    status <- factor(list_result[[c]][[i]]$result_status)
    levels(status) <- c("Reject", "Accept")
    result_mat[i, 6:7] <- table(sm, status)[, 2]/table(sm)
    result_mat[i, 8] <- length(unique(ci_result))
  }
  
  return(result_mat)
}
stopImplicitCluster()
print(Sys.time() - start_collect)

sum_tab(result_list)

result_list[[9]]
list_result[[9]][[3]]$split_or_merge

### Plot for the assignment
unique_clus <- matrix(NA, nrow = 30, ncol = 3000)
for(i in 1:30){
  unique_clus[i, ] <- apply(list_result[[1]][[i]]$result, 1, n_unique)
}

data.frame(iter = 1:3000, t(unique_clus)) %>%
  ggplot(aes(x = iter)) +
  geom_line(aes(y = X1)) +
  geom_line(aes(y = X2)) +
  geom_line(aes(y = X3)) +
  geom_line(aes(y = X4)) +
  theme_bw()
