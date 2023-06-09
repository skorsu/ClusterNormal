rm(list = ls())

### Packages
library(dplyr)
library(ggplot2)
library(bmixture)
library(salso)

### Model

fmm_model_R <- function(iter, dat, K_max, a0, b0, mu0, s20, xi0, ci_init){
  
  #### Storing the result
  mu_mat <- matrix(NA, nrow = iter, ncol = K_max)
  s2_mat <- matrix(NA, nrow = iter, ncol = K_max)
  mixing_mat <- matrix(NA, nrow = iter, ncol = K_max)
  assign_mat <- matrix(NA, nrow = iter, ncol = N)
  
  #### Intermediate Storage
  mun <- rep(NA, K_max)
  s2n <- rep(NA, K_max)
  an <- rep(NA, K_max)
  bn <- rep(NA, K_max)
  xin <- rep(NA, K_max)
  
  #### Initial mu and s2
  mu <- rnorm(K_max, mu0, sqrt(s20))
  s2 <- 1/rgamma(K_max, a0, b0)
  
  ### Perform the FMM
  for(t in 1:iter){
    
    ### Update the posterior parameter
    for(k in 1:K_max){
      nk <- sum(ci_init == k)
      an[k] <- a0 + (nk/2)
      bn[k] <- b0 + ifelse(nk > 0, 0.5 * sum((dat[ci_init == k] - mu[k])^2), 0)
      mun[k] <- ((s20 * sum(dat[ci_init == k])) + (mu0 * s2[k]))/((nk * s20) + s2[k])
      s2n[k] <- (s2[k] * s20)/((nk * s20) + s2[k])
      xin[k] <- xi0 + nk
    }
    
    ### Update mu
    mu <- rnorm(K_max, mun, sqrt(s2n))
    
    ### Update s2
    s2 <- 1/(rgamma(K_max, an, bn))
    
    ### Update the mixing proportion
    mix_p <- as.numeric(rdirichlet(1, xin))
    
    ### Update the cluster assignment
    for(i in 1:N){
      w <- mix_p * dnorm(dat[i], mu, sqrt(s2))
      ci_init[i] <- which.max(rmultinom(1, 1, w/sum(w)))
    }
    
    ### Label Switching protection
    sort_mu <- sort(mu)
    sort_s2 <- rep(NA, K_max)
    sort_mix <- rep(NA, K_max)
    sort_clus <- rep(NA, length(dat))
    
    for(k in 1:K_max){
      ci_old <- which(mu == sort_mu[k])
      sort_s2[k] <- s2[ci_old]
      sort_mix[k] <- mix_p[ci_old]
      sort_clus[ci_init == ci_old] <- k
    }
    
    mu <- sort_mu
    s2 <- sort_s2
    mix_p <- sort_mix
    ci_init <- sort_clus
    
    mu_mat[t, ] <- mu
    s2_mat[t, ] <- s2
    mixing_mat[t, ] <- mix_p
    assign_mat[t, ] <- ci_init
    
  }
  
  return(list(assign_mat = assign_mat, mu_mat = mu_mat, 
              s2_mat = s2_mat, mixing_mat = mixing_mat))
  
}

