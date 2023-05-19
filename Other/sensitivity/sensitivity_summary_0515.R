rm(list = ls())

### FIX ME: --------------------------------------------------------------------
overall_seed <- 24535
save_path <- "/Users/kevin-imac/Desktop/Result/Sensitivity/" ## The location where we will save the result
###: ---------------------------------------------------------------------------

### Required Packages: ---------------------------------------------------------
library(ggplot2)
library(gridExtra)
library(xtable)
library(tidyverse)
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
      dat <- rnorm(500, c(0, 7.5, 15, 25, 35)[actual_clus], 3)
    }
  }
  
  ### return the simulated data
  result <- data.frame(actual_clus, dat)
  return(result)
}

### Function: Summary Quantities "Mean (SD)".
bal_quan <- function(num_vec, rounding = 4){
  mean_val <- round(mean(num_vec), 4)
  sd_val <- round(sd(num_vec), 4)
  paste0(mean_val, " (", sd_val, ")")
}

### Function: Merge all 15 cases into one table
sum_tab <- function(res_list, rr = 4){
  result <- data.frame(matrix(NA, nrow = 15, ncol = 6))
  for(i in 1:15){
    result[i, ] <- apply(list_result[[i]], 2, bal_quan, rounding = rr)
  }
  result <- result[, -c(4:5)]
  colnames(result) <- c("Time", "Jaccard", "VI", "P(Accept)")
  xtable(result)
}

###: ---------------------------------------------------------------------------

### Plot for the example dataset
p1 <- ggplot(f_data_sim(overall_seed, 1), aes(x = dat, group = factor(actual_clus))) +
  geom_histogram(aes(y = after_stat(density)), bins = 50, alpha = 0.25) +
  geom_density(linewidth = 0.75) +
  theme_bw() +
  labs(title = "Scenario 1: 2 separated clusters", x = "Data", y = "Density") +
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        plot.title = element_text(size = 28))

p2 <- ggplot(f_data_sim(overall_seed, 2), aes(x = dat, group = factor(actual_clus))) +
  geom_histogram(aes(y = after_stat(density)), bins = 50, alpha = 0.25) +
  geom_density(linewidth = 0.75) +
  theme_bw() +
  labs(title = "Scenario 2: 5 separated clusters", x = "Data", y = "Density") +
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        plot.title = element_text(size = 28))

p3 <- ggplot(f_data_sim(overall_seed, 3), aes(x = dat, group = factor(actual_clus))) +
  geom_histogram(aes(y = after_stat(density)), bins = 50, alpha = 0.25) +
  geom_density(linewidth = 0.75) +
  theme_bw() +
  labs(title = "Scenario 3: 2 mixing clusters", x = "Data", y = "Density") +
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        plot.title = element_text(size = 28))

p4 <- ggplot(f_data_sim(overall_seed, 4), aes(x = dat, group = factor(actual_clus))) +
  geom_histogram(aes(y = after_stat(density)), bins = 50, alpha = 0.25) +
  geom_density(linewidth = 0.75) +
  theme_bw() +
  labs(title = "Scenario 4: 5 mixing clusters", x = "Data", y = "Density") +
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        plot.title = element_text(size = 28))

grid.arrange(p1, p2, p3, p4)

### Sensitivity Analysis

for(scenario_now in 1:4){
  load(paste0(save_path, "sensitivity_", scenario_now, ".RData"))
  print(paste0("Start printing the latex table for the scenario ", scenario_now))
  print(sum_tab(list_result))
  print("---------------------------------------------------------------------")
  rm(list_result)
}
