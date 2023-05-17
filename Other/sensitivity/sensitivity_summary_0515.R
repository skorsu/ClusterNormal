rm(list = ls())

### Required Packages
library(ggplot2)
library(gridExtra)
library(xtable)
library(tidyverse)

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

### Direction for the result file
result_source <- "/Users/kevin-imac/Desktop/Result/"
file_prefix <- "simu_result_scenario_"
load(paste0(result_source, file_prefix, "4", ".RData"))
sum_tab(list_result)


### Plot
dat <- f_data_sim(1, 2, FALSE)
p1 <- ggplot(dat, aes(x = dat, group = factor(actual_clus))) +
  geom_density() +
  theme_bw() +
  labs(title = "Scenario 1: 2 separated clusters", x = "Data", y = "Density") +
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        plot.title = element_text(size = 28))

dat <- f_data_sim(1, 5, FALSE)
p2 <- ggplot(dat, aes(x = dat, group = factor(actual_clus))) +
  geom_density() +
  theme_bw() +
  labs(title = "Scenario 2: 5 separated clusters", x = "Data", y = "Density") +
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        plot.title = element_text(size = 28))

dat <- f_data_sim(1, 2, TRUE)
p3 <- ggplot(dat, aes(x = dat, group = factor(actual_clus))) +
  geom_density() +
  theme_bw() +
  labs(title = "Scenario 3: 2 mixing clusters", x = "Data", y = "Density") +
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        plot.title = element_text(size = 28))

dat <- f_data_sim(1, 5, TRUE)
p4 <- ggplot(dat, aes(x = dat, group = factor(actual_clus))) +
  geom_density() +
  theme_bw() +
  labs(title = "Scenario 4: 5 mixing clusters", x = "Data", y = "Density") +
  theme(axis.text = element_text(size = 16),
        axis.title = element_text(size = 20),
        plot.title = element_text(size = 28))

grid.arrange(p1, p2, p3, p4)


