rm(list = ls())

### Required Packages
library(ClusterNormal)
library(cluster)
library(factoextra)
library(NbClust)

library(salso)
library(ggplot2)

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

### Simulation Study
dat <- f_data_sim(23, 5, TRUE)
fviz_nbclust(data.frame(dat$dat), kmeans, method = "wss")
mod_kmean <- NbClust(scale(dat$dat), distance = "euclidean", min.nc=2, 
                     max.nc=10, method = "kmeans", index = "silhouette")
table(mod_kmean$Best.partition, dat$actual_clus)


