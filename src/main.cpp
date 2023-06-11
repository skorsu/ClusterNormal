#include "RcppArmadillo.h"
#include "Rmath.h"

// [[Rcpp::depends(RcppArmadillo)]]

#define pi 3.141592653589793238462643383280

// Note: -----------------------------------------------------------------------
// * Cluster index starts from 0 to (K-1)

// -----------------------------------------------------------------------------

// User-defined function: ------------------------------------------------------
// [[Rcpp::export]]
arma::vec log_sum_exp(arma::vec log_unnorm_prob){
  
  /* Description: This function will calculate the normalized probability 
   *              by applying log-sum-exp trick on the log-scale probability.
   * Credit: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
   */
  
  double max_elem = log_unnorm_prob.max();
  double t = log(0.00000000000000000001) - log(log_unnorm_prob.size());          
  
  for(int k = 0; k < log_unnorm_prob.size(); ++k){
    double prob_k = log_unnorm_prob.at(k) - max_elem;
    if(prob_k > t){
      log_unnorm_prob.row(k).fill(std::exp(prob_k));
    } else {
      log_unnorm_prob.row(k).fill(0.0);
    }
  }
  
  // Normalize the vector
  return arma::normalise(log_unnorm_prob, 1);
}

// [[Rcpp::export]]
double log_marginal(double yi, double mu0, double s20, double a, double b,
                    double mu, double s2){
  
  /* Description: This function will calculate the proportional of the log 
   *              marginal distribution for the inactive clusters. 
   */
  
  double result = 0.0;
  
  result -= (0.5 * (1/s2) * std::pow(yi - mu, 2.0));
  result += ((a + 0.5) * std::log(b + (0.5 * std::pow(yi - mu, 2.0))));
  result += (0.5 * (1/(s2 + s20)) * std::pow(yi - mu0, 2.0));
  
  return result;
}

// [[Rcpp::export]]
Rcpp::IntegerVector rmultinom_1(Rcpp::NumericVector &probs, unsigned int &N){
  
  /* Description: sample from the multinomial(N, probs).
   * Credit: https://gallery.rcpp.org/articles/recreating-rmultinom-and-rpois-with-rcpp/
   */
  
  Rcpp::IntegerVector outcome(N);
  rmultinom(1, probs.begin(), N, outcome.begin());
  return outcome;
}

// [[Rcpp::export]]
arma::vec adjust_alpha(int K_max, arma::vec clus_assign, arma::vec alpha_vec){
  arma::vec a_alpha = arma::zeros(K_max);
  
  /* Description: To adjust the alpha vector. Keep only the element with at 
   *              least 1 observation is allocated to.
   */
  
  arma::vec new_active_clus = arma::unique(clus_assign);
  arma::uvec index_active = arma::conv_to<arma::uvec>::from(new_active_clus);
  a_alpha.elem(index_active) = alpha_vec.elem(index_active);
  
  return a_alpha;
}

// [[Rcpp::export]]
Rcpp::List SFDMM_rGibbs(arma::vec y, arma::vec sm_clus, double a0, double b0, 
                        double mu0, double s20, arma::vec ci_init, 
                        arma::vec mu, arma::vec s2, arma::vec S){
  
  /* Description: This function will perform a restricted Gibbs sampler based on
   *              Neal (3.14)
   */
  
  Rcpp::List result;
  
  unsigned int sm_size = sm_clus.size();
  
  // Reallocate the observation to one of the active cluster.
  for(int i = 0; i < S.size(); ++i){
    
    int s = S[i];
    
    arma::vec log_realloc(2, arma::fill::value(0.0));
    
    double yi = y[s];
    arma::vec c_not_i(ci_init);
    c_not_i.shed_row(s);
    
    for(int k = 0; k < 2; ++k){
      int cc = sm_clus[k];
      arma::uvec nk_vec = arma::find(c_not_i == cc);
      double l_re = std::log(nk_vec.size());
      
      if(nk_vec.size() == 0){
        l_re += log_marginal(yi, mu0, s20, a0, b0, mu[cc], s2[cc]);
      } else {
        l_re += R::dnorm4(yi, mu[cc], std::sqrt(s2[cc]), 1);
      }
      
      log_realloc.row(k).fill(l_re);
      
    }
    
    std::cout << "current_obs: " << s << std::endl;
  
    Rcpp::NumericVector realloc_prob = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(log_sum_exp(log_realloc)));
    
    std::cout << "realloc_prob: " << realloc_prob << std::endl;
    
    Rcpp::IntegerVector ind_vec = rmultinom_1(realloc_prob, sm_size);
    
    std::cout << "ind_vec: " << ind_vec << std::endl;
    
    arma::vec ind_arma = Rcpp::as<arma::vec>(Rcpp::wrap(ind_vec));
    arma::uword new_assign_i = arma::index_max(ind_arma);
    
    std::cout << "new_assign_i: " << new_assign_i << std::endl;
    
    ci_init.row(i).fill(sm_clus[new_assign_i]);
    
  }
  
  return result;
  
}

// Finite Mixture Model: -------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List fmm_rcpp(int iter, arma::vec y, unsigned int K_max, double a0, 
                    double b0, double mu0, double s20, double xi0, 
                    arma::vec ci_init){
  
  /* Description: This is a function for performing a Bayesian FMM. 
   */
  
  Rcpp::List result;
  
  // Storing the result
  arma::mat mu_mat(iter, K_max, arma::fill::value(0.0));
  arma::mat s2_mat(iter, K_max, arma::fill::value(0.0));
  arma::mat assign_mat(iter, y.size(), arma::fill::value(0.0));
    
  // Intermediate Storage
  arma::vec mun(K_max, arma::fill::value(0.0));
  arma::vec s2n(K_max, arma::fill::value(0.0));
  arma::vec an(K_max, arma::fill::value(0.0));
  arma::vec bn(K_max, arma::fill::value(0.0));
  arma::vec xin(K_max, arma::fill::value(0.0));
  
  // Initial mu and s2
  arma::vec mu(K_max, arma::fill::value(0.0));
  arma::vec s2(K_max, arma::fill::value(0.0));
  
  for(int k = 0; k < K_max; ++k){
    mu.row(k).fill(R::rnorm(mu0, std::sqrt(s20)));
    s2.row(k).fill(1/(R::rgamma(a0, (1/b0))));
  }
  
  for(int t = 0; t < iter; ++t){
    
    // Update the hyperparameter and update mu and s2
    for(int k = 0; k < K_max; ++k){
      arma::uvec nk_vec = arma::find(ci_init == k);
      double nk = nk_vec.size();
      if(nk == 0){
        an.row(k).fill(a0);
        bn.row(k).fill(b0);
        mun.row(k).fill(mu0);
        s2n.row(k).fill(s20);
        xin.row(k).fill(xi0);
      } else {
        arma::vec yk = y.rows(nk_vec);
        an.row(k).fill(a0 + (nk/2));
        bn.row(k).fill(b0 + (0.5 * arma::accu(arma::pow(yk - mu[k], 2.0))));
        mun.row(k).fill(((s20 * arma::accu(yk)) + (mu0 * s2[k]))/((nk * s20) + s2[k]));
        s2n.row(k).fill((s2[k] * s20)/((nk * s20) + s2[k]));
        xin.row(k).fill(xi0 + nk);
      }
      
      // Update mu and s2
      mu.row(k).fill(R::rnorm(mun[k], std::sqrt(s2n[k])));
      s2.row(k).fill(1/(R::rgamma(an[k], (1/bn[k]))));
    }
    
    // Update the cluster assignment
    for(int i = 0; i < y.size(); ++i){
      arma::vec log_realloc(K_max, arma::fill::value(0.0));
      
      double yi = y[i];
      arma::vec c_not_i(ci_init);
      c_not_i.shed_row(i);
      
      for(int k = 0; k < K_max; ++k){
        arma::uvec nk_vec = arma::find(c_not_i == k);
        log_realloc.row(k).fill(std::log(nk_vec.size() + xi0) + R::dnorm4(yi, mu[k], std::sqrt(s2[k]), 1));
      }
      
      Rcpp::NumericVector realloc_prob = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(log_sum_exp(log_realloc)));
      Rcpp::IntegerVector ind_vec = rmultinom_1(realloc_prob, K_max);
      arma::vec ind_arma = Rcpp::as<arma::vec>(Rcpp::wrap(ind_vec));
      arma::uword new_assign_i = arma::index_max(ind_arma);
      ci_init.row(i).fill(new_assign_i);

    }
    
    // Relabel: Label Switching protection
    arma::uvec mu_sort_order = arma::sort_index(mu);
    mu = mu.rows(mu_sort_order);
    s2 = s2.rows(mu_sort_order);
    
    arma::vec clus_order(y.size(), arma::fill::value(-1));
    for(int k = 0; k < K_max; ++k){ // New index
      int old_c = mu_sort_order[k]; // get the old index
      arma::uvec old_ci = arma::find(ci_init == old_c);
      clus_order.rows(old_ci).fill(k);
    }
    ci_init = clus_order;
    
    mu_mat.row(t) = mu.t();
    s2_mat.row(t) = s2.t();
    assign_mat.row(t) = ci_init.t();
    
    }

  result["mu"] = mu_mat;
  result["sigma2"] = s2_mat;
  result["assign_mat"] = assign_mat;
  
  return result;
}

// Sparse Finite Discrete Mixture Model (SFDMM): -------------------------------
// [[Rcpp::export]]
Rcpp::List SFDMM_realloc(arma::vec y, unsigned int K_max, double a0, double b0, 
                         double mu0, double s20, double xi0, arma::vec ci_init, 
                         arma::vec mu, arma::vec s2, arma::vec alpha_vec){
  
  /* Description: This function will perform the reallocation step, the first 
   *              step of the SFDMM.
   */
  
  Rcpp::List result;
  
  // Obtain the active cluster's index
  arma::vec clus_active = arma::unique(ci_init);
  unsigned int K_pos = clus_active.size();
  
  // Reallocate the observation to one of the active cluster.
  for(int i = 0; i < y.size(); ++i){
    
    arma::vec log_realloc(clus_active.size(), arma::fill::value(0.0));
    
    double yi = y[i];
    arma::vec c_not_i(ci_init);
    c_not_i.shed_row(i);
    
    for(int k = 0; k < K_pos; ++k){
      int cc = clus_active[k];
      arma::uvec nk_vec = arma::find(c_not_i == cc);
      double l_re = std::log(nk_vec.size() + xi0);
      
      if(nk_vec.size() == 0){
        l_re += log_marginal(yi, mu0, s20, a0, b0, mu[cc], s2[cc]);
      } else {
        l_re += R::dnorm4(yi, mu[cc], std::sqrt(s2[cc]), 1);
      }
      
      log_realloc.row(k).fill(l_re);
      
    }
    
    Rcpp::NumericVector realloc_prob = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(log_sum_exp(log_realloc)));
    Rcpp::IntegerVector ind_vec = rmultinom_1(realloc_prob, K_pos);
    arma::vec ind_arma = Rcpp::as<arma::vec>(Rcpp::wrap(ind_vec));
    arma::uword new_assign_i = arma::index_max(ind_arma);
    ci_init.row(i).fill(clus_active[new_assign_i]);
    
  }
  
  // Update mu and s2
  for(int k = 0; k < K_pos; ++k){
    int cc = clus_active[k];
    arma::uvec nk_vec = arma::find(ci_init == cc);
    
    double nk = nk_vec.size();
    double an = 0.0;
    double bn = 0.0;
    double mun = 0.0;
    double s2n = 0.0;
    
    if(nk == 0){
      an = a0;
      bn = b0;
      mun = mu0;
      s2n = s20;
    } else {
      arma::vec yk = y.rows(nk_vec);
      an = (a0 + (nk/2));
      bn = (b0 + (0.5 * arma::accu(arma::pow(yk - mu[cc], 2.0))));
      mun = (((s20 * arma::accu(yk)) + (mu0 * s2[cc]))/((nk * s20) + s2[cc]));
      s2n = ((s2[cc] * s20)/((nk * s20) + s2[cc]));
    }
    
    mu.row(cc).fill(R::rnorm(mun, std::sqrt(s2n)));
    s2.row(cc).fill(1/(R::rgamma(an, (1/bn))));
  }
  
  // Update the alpha vector
  alpha_vec = adjust_alpha(K_max, ci_init, alpha_vec);
  
  // Relabel: Label Switching protection
  arma::uvec mu_sort_order = arma::sort_index(mu);
  mu = mu.rows(mu_sort_order);
  s2 = s2.rows(mu_sort_order);
  alpha_vec = alpha_vec.rows(mu_sort_order);
  
  arma::vec clus_order(y.size(), arma::fill::value(-1));
  for(int k = 0; k < K_max; ++k){ // New index
    int old_c = mu_sort_order[k]; // get the old index
    arma::uvec old_ci = arma::find(ci_init == old_c);
    clus_order.rows(old_ci).fill(k);
  }
  ci_init = clus_order;
  
  result["new_mu"] = mu;
  result["new_s2"] = s2;
  result["new_alpha"] = alpha_vec;
  result["new_ci"] = ci_init;
  
  return result;
}

// [[Rcpp::export]]
Rcpp::List SFDMM_SM(arma::vec y, unsigned int K_max, double a0, double b0, 
                    double mu0, double s20, double xi0, arma::vec ci_init, 
                    arma::vec mu_init, arma::vec s2_init, arma::vec alpha_init,
                    int launch_iter, double a_theta, double b_theta){
  
  /* Description: This function will perform the split-merge MCMC, the second 
   *              step of the SFDMM.
   */
  
  Rcpp::List result;
  
  int n = y.size();
  arma::vec active_clus = arma::unique(ci_init);
  int K_pos = active_clus.size();
  int split_ind = -1; // Indicate that we will perform split or merge
  int accept_proposed = 0; // indicator for accepting the proposed assignment.
  double log_A = 0.0; // log of acceptance probability
  
  arma::vec launch_assign(ci_init);
  arma::vec launch_alpha(alpha_init);
  arma::vec launch_mu(mu_init);
  arma::vec launch_s2(s2_init);
  
  // Select two observations to determine whether we will split or merge.
  arma::uvec samp_obs = arma::randperm(n, 2);
  
  // If all cluster is already active, we can perform only merge.
  while((K_pos == K_max) and (ci_init[samp_obs[0]] == ci_init[samp_obs[1]])){
    samp_obs = arma::randperm(n, 2); // sample two observations again until we get merge.
  }
  arma::vec samp_clus = ci_init.rows(samp_obs);
  
  int new_clus = -1;
  // Create the split indicator. If we split, split_ind = 1. Otherwise, split_ind = 0.
  if(samp_clus[0] == samp_clus[1]){
    split_ind = 1; // Split
    
    arma::uvec inactive_index = arma::find(alpha_init == 0);
    arma::uvec new_clus_index = arma::randperm(inactive_index.size(), 1);
    new_clus = inactive_index[new_clus_index[0]]; // new active cluster
    samp_clus.row(0).fill(new_clus);
    
    launch_assign.row(samp_obs[0]).fill(new_clus); // set ci_launch to be a new cluster
    launch_alpha.row(new_clus).fill(R::rgamma(xi0, 1.0));
    launch_mu.row(new_clus).fill(R::rnorm(mu0, std::sqrt(s20)));
    launch_s2.row(new_clus).fill(1/(R::rgamma(a0, 1/b0)));
    
  } else {
    split_ind = 0;
  }
  
  // Create a set S := {same cluster as samp_obs, but not index_samp}
  arma::uvec S = arma::find(ci_init == ci_init[samp_obs[0]] or ci_init == ci_init[samp_obs[1]]);
  arma::uvec samp_obs_index = arma::find(S == samp_obs[0] or S == samp_obs[1]);
  S.shed_rows(samp_obs_index);
  
  // Perform a launch step
  // Randomly assign observation in S to be ci_launch or cj_launch.
  arma::vec init_ind = arma::randu(S.size(), arma::distr_param(0, 1));
  launch_assign.rows(S) = samp_clus.rows(arma::conv_to<arma::uvec>::from(init_ind > 0.5));
  
  std::cout << "split/merge: " << split_ind << std::endl;
  std::cout << "samp_obs: " << samp_obs << std::endl;
  std::cout << "samp_clus" << samp_clus << std::endl;
  std::cout << "assign: " << arma::join_horiz(ci_init, launch_assign) << std::endl; 
  std::cout << "alpha: " << arma::join_horiz(alpha_init, launch_alpha) << std::endl; 
  std::cout << "mu: " << arma::join_horiz(mu_init, launch_mu) << std::endl; 
  std::cout << "s2: " << arma::join_horiz(s2_init, launch_s2) << std::endl; 
  
  result["split_ind"] = split_ind;
  return result;
  
}

// END: ------------------------------------------------------------------------