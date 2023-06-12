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
      log_unnorm_prob.row(k).fill(0.00000000000000000001);
    }
  }
  
  // Normalize the vector
  return log_unnorm_prob/arma::accu(log_unnorm_prob);
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
Rcpp::List SFDMM_rGibbs(arma::vec y, arma::vec sm_clus, arma::vec ci_init, 
                        arma::vec mu, arma::vec s2, arma::uvec S, 
                        double a0, double b0, double mu0, double s20){
  
  /* Description: This function will perform a restricted Gibbs sampler based on
   *              Neal (3.14). Also, we will update mu and s2 in each iteration.
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
      log_realloc.row(k).fill(std::log(nk_vec.size()) + R::dnorm4(yi, mu[cc], std::sqrt(s2[cc]), 1));
    }
    
    // Sample from the rmultinomial
    Rcpp::NumericVector realloc_prob = Rcpp::as<Rcpp::NumericVector>(Rcpp::wrap(log_sum_exp(log_realloc)));
    Rcpp::IntegerVector ind_vec = rmultinom_1(realloc_prob, sm_size);
    arma::vec ind_arma = Rcpp::as<arma::vec>(Rcpp::wrap(ind_vec));
    arma::uword new_assign_i = arma::index_max(ind_arma);
    
    ci_init.row(s).fill(sm_clus[new_assign_i]);
    
  }
  
  // Update mu and s2
  for(int k = 0; k < 2; ++k){
    int cc = sm_clus[k];
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
    
    // Update mu and s2
    mu.row(cc).fill(R::rnorm(mun, std::sqrt(s2n)));
    s2.row(cc).fill(1/(R::rgamma(an, (1/bn))));
  }
  
  result["assign"] = ci_init;
  result["new_mu"] = mu;
  result["new_s2"] = s2;
  
  return result;
  
}

// [[Rcpp::export]]
double log_inv_gamma(double s2_k, double a0, double b0){
  
  /* Description: This is a function calculate the density of the inverse gamma 
   *              distribution in a log scale. 
   */

  double result = 0.0;
  result += (a0 * std::log(b0));
  result -= std::lgamma(a0);
  result += ((-a0 - 1) * std::log(s2_k));
  result -= (b0/s2_k);
  
  return result;
  
}

// [[Rcpp::export]]
double log_cluster_prior(arma::vec ci, double xi0){
  
  /* Description: NA
   */
  
  double result = 0.0;
  double xi_total = 0.0;
  
  arma::vec active_clus = arma::unique(ci);
  
  for(int k = 0; k < active_clus.size(); ++k){
    int cc = active_clus[k];
    arma::uvec nk_vec = arma::find(ci == cc);
    result += std::lgamma(nk_vec.size() + xi0);
    result -= std::lgamma(xi0);
    xi_total += xi0;
  }
  
  result += std::lgamma(xi_total);
  result -= std::lgamma(ci.size() + xi_total);
  
  return result;
}

// [[Rcpp::export]]
double log_proposal(arma::vec y, arma::vec ci_after, arma::vec ci_before, 
                    arma::vec sm_clus, arma::vec mu_before, 
                    arma::vec s2_before, arma::uvec S){
  
  /* Description: NA
   */
  
  double result = 0.0;
  unsigned int K_pos = sm_clus.size();
  
  // Update the cluster assignment
  for(int i = 0; i < S.size(); ++i){
    
    int s = S[i];
    double yi = y[s];
    
    arma::vec c_not_i(ci_before);
    c_not_i.shed_row(s);
    
    arma::vec log_realloc(K_pos, arma::fill::value(0.0));
    for(int k = 0; k < K_pos; ++k){
      int cc = sm_clus[k];
      arma::uvec nk_vec = arma::find(c_not_i == cc);
      log_realloc.row(k).fill(std::log(nk_vec.size()) + R::dnorm4(yi, mu_before[cc], std::sqrt(s2_before[cc]), 1));
      }
    
    arma::vec alloc_prob = log_sum_exp(log_realloc);
    
    arma::uvec restricted_index = arma::find(sm_clus == ci_after[s]);
    result += std::log(alloc_prob[restricted_index[0]]);
    ci_before.row(s).fill(ci_after[s]);
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
      log_realloc.row(k).fill(std::log(nk_vec.size() + xi0) + R::dnorm4(yi, mu[cc], std::sqrt(s2[cc]), 1));
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
  
  Rcpp::List launch_product;
  
  for(int t = 0; t < launch_iter; ++t){
    launch_product = SFDMM_rGibbs(y, samp_clus, launch_assign, launch_mu, 
                                  launch_s2, S, a0, b0, mu0, s20);
    arma::vec l_assign = launch_product["assign"];
    arma::vec l_mu = launch_product["new_mu"];
    arma::vec l_s2 = launch_product["new_s2"];
    
    launch_assign = l_assign;
    launch_mu = l_mu;
    launch_s2 = l_s2;
  }
  
  // Split-Merge step
  arma::vec proposed_assign(launch_assign);
  arma::vec proposed_alpha(launch_alpha);
  arma::vec proposed_mu(launch_mu);
  arma::vec proposed_s2(launch_s2);
  Rcpp::List proposed_product;
  
  if(split_ind == 1){
    proposed_product = SFDMM_rGibbs(y, samp_clus, proposed_assign, launch_mu, 
                                    launch_s2, S, a0, b0, mu0, s20);
    arma::vec p_assign = proposed_product["assign"];
    arma::vec p_mu = proposed_product["new_mu"];
    arma::vec p_s2 = proposed_product["new_s2"];
    
    proposed_assign = p_assign;
    proposed_mu = p_mu;
    proposed_s2 = p_s2;
    
  } else {
    // Merge: All observations in S and {ci, cj} will be allocated to cj
    proposed_assign.rows(S).fill(samp_clus[1]);
    proposed_assign.rows(samp_obs).fill(samp_clus[1]);
  }
  
  // Proposal Probability
  // Data
  for(int i = 0; i < y.size(); ++i){
    log_A += R::dnorm4(y[i], proposed_mu[proposed_assign[i]], std::sqrt(proposed_s2[proposed_assign[i]]), 1);
    log_A -= R::dnorm4(y[i], mu_init[ci_init[i]], std::sqrt(s2_init[ci_init[i]]), 1);
  }
  
  arma::vec norm_init_alpha = arma::normalise(alpha_init, 1);
  arma::vec norm_launch_alpha = arma::normalise(launch_alpha, 1); 
  
  for(int k = 0; k < K_max; ++k){
    // mu
    log_A += R::dnorm4(proposed_mu[k], mu0, std::sqrt(s20), 1);
    log_A -= R::dnorm4(mu_init[k], mu0, std::sqrt(s20), 1);
    
    // s2
    log_A += log_inv_gamma(proposed_s2[k], a0, b0);
    log_A -= log_inv_gamma(s2_init[k], a0, b0);
  }
  
  log_A += log_cluster_prior(launch_assign, xi0);
  log_A -= log_cluster_prior(ci_init, xi0);
  
  log_A += (((2 * split_ind) - 1) * (std::log(a_theta) - std::log(b_theta)));
  
  // Proposal
  if(split_ind == 1){ // split
    log_A += log_proposal(y, launch_assign, proposed_assign, samp_clus, proposed_mu, proposed_s2, S);
    log_A -= log_proposal(y, proposed_assign, launch_assign, samp_clus, launch_mu, launch_s2, S);
  } else {
    log_A += log_proposal(y, launch_assign, ci_init, samp_clus, mu_init, s2_init, S);
  }
  
  // MH
  arma::vec new_assign(ci_init);
  arma::vec new_mu(mu_init);
  arma::vec new_s2(s2_init);
  arma::vec new_alpha(alpha_init); 
  
  if(std::log(R::runif(0.0, 1.0)) < std::min(0.0, log_A)){
    // Accept the proposed vector
    accept_proposed = 1;
    new_assign = proposed_assign;
    new_mu = proposed_mu;
    new_s2 = proposed_s2;
    new_alpha = proposed_alpha;
  }
  
  // Update alpha vector
  new_alpha = adjust_alpha(K_max, new_assign, new_alpha);
  
  result["split_ind"] = split_ind;
  result["accept_proposed"] = accept_proposed;
  result["log_A"] = log_A;
  result["new_mu"] = new_mu;
  result["new_s2"] = new_s2;
  result["new_alpha"] = new_alpha;
  result["new_ci"] = new_assign;
  
  return result;
  
}

// [[Rcpp::export]]
Rcpp::List SFDMM_param(arma::vec clus_assign, arma::vec y, arma::vec mu, arma::vec s2, 
                       arma::vec alpha_vec, double old_U, unsigned int K_max, 
                       double a0, double b0, double mu0, double s20, double xi0){
  
  /* Description: This function will perform update the parameters in the model, 
   *              the third step of the SFDMM.
   */
  
  Rcpp::List result;
  arma::vec new_alpha(alpha_vec); 
  
  // update alpha vector
  arma::vec active_clus = arma::unique(clus_assign);
  for(int k = 0; k < active_clus.size(); ++k){
    int current_c = active_clus[k];
    arma::uvec nk = arma::find(clus_assign == current_c);
    double scale_gamma = 1/(1 + old_U); // change the rate to scale parameter
    new_alpha.row(current_c).fill(R::rgamma(nk.size() + xi0, scale_gamma));
  }
  
  // update U
  int n = clus_assign.size();
  double scale_u = 1/arma::accu(new_alpha);
  double new_U = R::rgamma(n, scale_u);
  
  // update mu and s2
  for(int k = 0; k < K_max; ++k){
    arma::uvec nk_vec = arma::find(clus_assign == k);
    
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
      bn = (b0 + (0.5 * arma::accu(arma::pow(yk - mu[k], 2.0))));
      mun = (((s20 * arma::accu(yk)) + (mu0 * s2[k]))/((nk * s20) + s2[k]));
      s2n = ((s2[k] * s20)/((nk * s20) + s2[k]));
    }
    
    mu.row(k).fill(R::rnorm(mun, std::sqrt(s2n)));
    s2.row(k).fill(1/(R::rgamma(an, (1/bn))));
    
  }
  
  // Relabel: Label Switching protection
  arma::uvec mu_sort_order = arma::sort_index(mu);
  mu = mu.rows(mu_sort_order);
  s2 = s2.rows(mu_sort_order);
  alpha_vec = alpha_vec.rows(mu_sort_order);
  
  arma::vec clus_order(y.size(), arma::fill::value(-1));
  for(int k = 0; k < K_max; ++k){ // New index
    int old_c = mu_sort_order[k]; // get the old index
    arma::uvec old_ci = arma::find(clus_assign == old_c);
    clus_order.rows(old_ci).fill(k);
  }
  clus_assign = clus_order;
  
  result["new_U"] = new_U;
  result["new_alpha"] = new_alpha;
  result["new_mu"] = mu;
  result["new_s2"] = s2;
  result["new_ci"] = clus_assign;
  
  return result;
  
}

// Final Functions: ------------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List SFDMM_model(int iter, unsigned int K_max, arma::vec init_assign, 
                       arma::vec y, double a0, double b0, double mu0, double s20, 
                       double xi0, double a_theta, double b_theta, 
                       int launch_iter, int print_iter){
  
  /* Description: This function will perform update the parameters in the model, 
   *              the third step of the SFDMM.
   */
  
  Rcpp::List result;
  
  // Initial alpha
  arma::uvec init_unique = arma::conv_to<arma::uvec>::from(arma::unique(init_assign));
  arma::vec init_alpha(K_max, arma::fill::zeros);

  for(int k = 0; k < init_unique.size(); ++k){
    init_alpha.row(init_unique[k]).fill(R::rgamma(xi0, 1.0));
  }
  
  // Initial U (auxiliary variable)
  double init_u = R::rgamma(y.size(), 1/(arma::accu(init_alpha)));
  
  // Initial mu and s2
  arma::vec mu(K_max, arma::fill::value(0.0));
  arma::vec s2(K_max, arma::fill::value(0.0));
  mu = arma::randn(K_max, arma::distr_param(mu0, std::sqrt(s20)));
  s2 = 1/(arma::randg(K_max, arma::distr_param(a0, (1/b0))));
  
  // Create vectors/matrices for storing the final result
  arma::mat iter_assign(y.size(), iter, arma::fill::value(-1));
  arma::mat iter_alpha(K_max, iter, arma::fill::value(-1));
  arma::mat iter_mu(K_max, iter, arma::fill::value(-1));
  arma::mat iter_s2(K_max, iter, arma::fill::value(-1));
  arma::vec sm_status(iter, arma::fill::value(-1));
  arma::vec iter_log_A(iter, arma::fill::value(-1));
  arma::vec split_or_merge(iter, arma::fill::value(-1));
  
  for(int i = 0; i < iter; ++i){
    // Reallocation Step
    Rcpp::List realloc_List = SFDMM_realloc(y, K_max, a0, b0, mu0, s20, xi0, 
                                            init_assign, mu, s2, init_alpha);
    arma::vec realloc_assign = realloc_List["new_ci"];
    arma::vec realloc_alpha = realloc_List["new_alpha"];
    arma::vec realloc_mu = realloc_List["new_mu"];
    arma::vec realloc_s2 = realloc_List["new_s2"];
    
    // Split-Merge Step
    Rcpp::List sm_List = SFDMM_SM(y, K_max, a0, b0, mu0, s20, xi0, realloc_assign, 
                                  realloc_mu, realloc_s2, realloc_alpha,
                                  launch_iter, a_theta, b_theta);
    arma::vec sm_assign = sm_List["new_ci"];
    arma::vec sm_alpha = sm_List["new_alpha"];
    arma::vec sm_mu = sm_List["new_mu"];
    arma::vec sm_s2 = sm_List["new_s2"];
    
    // Update the parameters
    Rcpp::List param_update_List = SFDMM_param(sm_assign, y, sm_mu, sm_s2, sm_alpha, 
                                               init_u, K_max, a0, b0, mu0, s20, xi0);
    arma::vec alpha_updated = param_update_List["new_alpha"];
    double U_update = param_update_List["new_U"];
    arma::vec assign_updated = param_update_List["new_ci"];
    arma::vec mu_updated = param_update_List["new_mu"];
    arma::vec s2_updated = param_update_List["new_s2"];
    
    // Store the result
    iter_assign.col(i) = assign_updated;
    iter_alpha.col(i) = alpha_updated;
    iter_log_A.row(i).fill(sm_List["log_A"]);
    split_or_merge.row(i).fill(sm_List["split_ind"]);
    sm_status.row(i).fill(sm_List["accept_proposed"]);
    iter_mu.col(i) = mu_updated;
    iter_s2.col(i) = s2_updated;
    
    // Initialize for the next iteration
    init_assign = assign_updated;
    init_alpha = alpha_updated;
    init_u = U_update;
    mu = mu_updated;
    s2 = s2_updated;
    
    // Print the result
    if(((i + 1) - (floor((i + 1)/print_iter) * print_iter)) == 0){
      std::cout << "Iter: " << (i+1) << " - Done!" << std::endl;
    }
    
  } 
  
  // Return the result
  result["iter_assign"] = iter_assign.t();
  result["iter_alpha"] = iter_alpha.t();
  result["log_A"] = iter_log_A;
  result["sm_status"] = sm_status;
  result["split_or_merge"] = split_or_merge;
  result["iter_mu"] = iter_mu.t();
  result["iter_s2"] = iter_s2.t();
  
  return result;
  
}
// END: ------------------------------------------------------------------------