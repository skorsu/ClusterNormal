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
   *              by applying log-sum-exp trick.
   * Credit: https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/
   * Input: log of the unnormalized probability (log_unnorm_prob)
   * Output: normalized probability
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

arma::vec adjust_alpha(int K, arma::vec clus_assign, arma::vec alpha_vec){
  arma::vec a_alpha = arma::zeros(K);
  
  /* Description: To adjust the alpha vector. Keep only the element with at 
   *              least 1 observation is allocated to.
   * Input: maximum cluster (K), cluster assignment, cluster weight (alpha_vec)
   * Output: adjusted alpha vector
   */
  
  arma::vec new_active_clus = arma::unique(clus_assign);
  arma::uvec index_active = arma::conv_to<arma::uvec>::from(new_active_clus) - 1;
  a_alpha.elem(index_active) = alpha_vec.elem(index_active);
  
  return a_alpha;
}

// [[Rcpp::export]]
double log_marginal(arma::vec y, int ci, arma::vec mu0_cluster, 
                    arma::vec lambda_cluster, arma::vec a_sigma_cluster, 
                    arma::vec b_sigma_cluster){
  
  /* Description: This function will calculate the marginal of the data, 
   *              assuming that the data is Gaussian distributed.
   * Output: The marginal distribution in a log-scale of the data based on the 
   *         corresponding cluster.
   * Input: data (y), corresponding cluster (ci), hyperparameters of the data 
   *        (mu0_cluster, lambda_cluster, a_sigma_cluster, b_sigma_cluster)
   */
  
  double result = 0.0;
  
  // Hyperparameter for the corresponding cluster
  double mu0 = mu0_cluster[ci];
  double lb = lambda_cluster[ci];
  double as = a_sigma_cluster[ci];
  double bs = b_sigma_cluster[ci];
  
  // Calculate the intermediate quantities
  double nk = y.size();
  double lb_n = lb + nk;
  double an = as + (nk/2);
  double bn = bs + (0.5 * (nk - 1) * arma::var(y)) + 
    (0.5 * ((lb * nk)/lb_n) * std::pow(mu0 - arma::mean(y), 2.0));
  
  // Calculate the marginal
  result -= ((nk/2) * std::log(2 * pi));
  result += (std::lgamma(an) - std::lgamma(as));
  result += (0.5 * (std::log(lb_n) - std::log(lb)));
  result += ((as * std::log(bs)) - (an * std::log(bn)));
  
  return result;
}

// [[Rcpp::export]]
double log_posterior(arma::vec y_new, arma::vec data, int ci, 
                     arma::vec mu0_cluster, arma::vec lambda_cluster, 
                     arma::vec a_sigma_cluster, arma::vec b_sigma_cluster){
  
  /* Description: This function will calculate the posterior predictive of the 
   *              new observations given the data, assuming that the data is 
   *              Gaussian distributed.
   * Output: The posterior predictive distribution for the new observation given
   *         the data in a log-scale of the data based on the corresponding 
   *         cluster.
   * Input: new observation (y_new), current observations (data), 
   *        corresponding cluster (ci), hyperparameters of the data 
   *        (mu0_cluster, lambda_cluster, a_sigma_cluster, b_sigma_cluster)
   */
  
  double result = 0.0;
  double log_numer = log_marginal(arma::join_cols(y_new, data), ci, mu0_cluster, 
                                  lambda_cluster, a_sigma_cluster, 
                                  b_sigma_cluster);
  double log_denom = log_marginal(data, ci, mu0_cluster, lambda_cluster, 
                                  a_sigma_cluster, b_sigma_cluster);
  result = log_numer - log_denom; // P(y_new|Data) = P(y_new, Data)/P(Data)
  
  return result;
}

// [[Rcpp::export]]
int rmultinom_1(arma::vec unnorm_prob, unsigned int N){
  
  /* Credit: 
   * https://gallery.rcpp.org/articles/recreating-rmultinom-and-rpois-with-rcpp/
   */
  
  // Convert arma object to Rcpp object
  Rcpp::NumericVector prob_rcpp = Rcpp::wrap(unnorm_prob);
  
  // Run a multinomial distribution random function
  Rcpp::IntegerVector outcome(N);
  rmultinom(1, prob_rcpp.begin(), N, outcome.begin());
  
  // Convert back to the cluster index
  arma::vec outcome_arma = Rcpp::as<arma::vec>(Rcpp::wrap(outcome));
  int desire_cluster = arma::index_max(outcome_arma);
  
  return desire_cluster;
}

// Finite Mixture Model: -------------------------------------------------------
// [[Rcpp::export]]
arma::vec fmm_iter(int K, arma::vec old_assign, arma::vec y,
                   arma::vec mu0_cluster, arma::vec lambda_cluster,
                   arma::vec a_sigma_cluster, arma::vec b_sigma_cluster,
                   arma::vec xi_cluster){

  /* Description: -
   * Output: -
   * Input: -
   */

  arma::vec new_assign(old_assign);

  // Loop through the observations
  for(int i = 0; i < y.size(); ++i){
    
    // Select y_current (as a y_new)
    arma::vec y_current = y.row(i);
    // Create vector for y_not_i and cluster_not_i
    arma::vec y_not_i(y);
    arma::vec assign_not_i(new_assign);
    y_not_i.shed_row(i);
    assign_not_i.shed_row(i);

    // Create a matrix for collecting the allocation probability
    // (row: clusters, column: (cluster index, log_predictive, predictive))
    arma::mat alloc_prob(K, 3, arma::fill::value(-1000));

    // Loop through all possible cluster
    for(int k = 0; k < K; ++k)
    {
      alloc_prob.col(0).row(k).fill(k);
      arma::uvec obs_index = arma::find(assign_not_i == k);
      double log_pred = 0.0;
      if(obs_index.size() == 0){
        // If there are no observations in that cluster, the predictive is just
        // a marginal distribution.
        log_pred += log_marginal(y_current, k, mu0_cluster, lambda_cluster,
                                a_sigma_cluster, b_sigma_cluster);
      } else {
        log_pred += log_posterior(y_current, y_not_i.rows(obs_index), k,
                                 mu0_cluster, lambda_cluster, a_sigma_cluster,
                                 b_sigma_cluster);
      }

      log_pred += std::log(obs_index.size() + xi_cluster[k]);
      alloc_prob.col(1).row(k).fill(log_pred);
    }

    // Calculate the predictive probability and normalize it.
    alloc_prob.col(2) = log_sum_exp(alloc_prob.col(1));

    // Sample a new cluster based on the predictive probability
    int index_new_cluster = rmultinom_1(alloc_prob.col(2), K);
    new_assign.row(i).fill(alloc_prob.col(0)[index_new_cluster]);
  }

  return new_assign;
}

// [[Rcpp::export]]
arma::mat fmm(int iter, int K, arma::vec old_assign, arma::vec y,
              arma::vec mu0_cluster, arma::vec lambda_cluster,
              arma::vec a_sigma_cluster, arma::vec b_sigma_cluster,
              arma::vec xi_cluster){

  /* Description: -
   * Output: -
   * Input: -
   */

  arma::mat cluster_iter(y.size(), iter, arma::fill::value(-1));
  arma::vec inter_assign(old_assign);

  for(int j = 0; j < iter; ++j){

    cluster_iter.col(j) = fmm_iter(K, inter_assign, y, mu0_cluster,
                     lambda_cluster, a_sigma_cluster, b_sigma_cluster,
                     xi_cluster);
    inter_assign = cluster_iter.col(j);

  }

  return cluster_iter.t();

}

// Function for SFDM: ----------------------------------------------------------
// [[Rcpp::export]]
arma::vec adjust_alpha(arma::vec cluster_assign, arma::vec old_alpha){
  
  /* Description: This function will adjust the alpha vector. This function 
   *              should be used within the reallocation and split-merge step.
   * Output: The vector of adjusted alpha. If there is no observation in that 
   *         cluster, we consider it as non-active. The corresponding alpha must
   *         be 0.
   * Input: -
   */
  
  arma::vec new_alpha(old_alpha.size(), arma::fill::value(0.0));
  
  arma::uvec active_clus = arma::conv_to<arma::uvec>::from(arma::unique(cluster_assign));
  new_alpha.rows(active_clus) = old_alpha.rows(active_clus);
  
  return new_alpha;
}

// [[Rcpp::export]]
arma::vec split_launch(arma::vec old_assign, arma::vec y, arma::vec mu0_cluster, 
                       arma::vec lambda_cluster, arma::vec a_sigma_cluster, 
                       arma::vec b_sigma_cluster, arma::vec sm_cluster, 
                       arma::uvec S_index){
  
  /* Description: This function will perform a single launch step. This function
   *              will be used in the split-merge procedure.
   * Output: The vector of new cluster assignment after performing one iteration
   *         of launch step.
   * Input: -
   */
  
  arma::vec new_assign(old_assign);
  
  // Loop through the observations in the S
  for(int i = 0; i < S_index.size(); ++i){
    
    int j = S_index[i];
    
    // Select y_current (as a y_new)
    arma::vec y_current = y.row(j);
    // Create vector for y_not_i and cluster_not_i
    arma::vec y_not_i(y);
    arma::vec assign_not_i(new_assign);
    y_not_i.shed_row(j);
    assign_not_i.shed_row(j);
    
    // Create a matrix for collecting the allocation probability
    // (row: clusters, column: (cluster index, log_predictive, predictive))
    arma::mat alloc_prob(2, 3, arma::fill::value(-1000));
    
    // Loop through all possible cluster
    for(int k = 0; k < 2; ++k)
    {
      int cc = sm_cluster[k];
      alloc_prob.col(0).row(k).fill(cc);
      arma::uvec obs_index = arma::find(assign_not_i == cc);
      
      double nk = obs_index.size();
      double log_pred = 0.0;
      
      if(nk == 0){
        // If there are no observations in that cluster, the predictive is just
        // a marginal distribution.
        log_pred += log_marginal(y_current, cc, mu0_cluster, lambda_cluster,
                                 a_sigma_cluster, b_sigma_cluster);
        } else {
        log_pred += log_posterior(y_current, y_not_i.rows(obs_index), cc,
                                  mu0_cluster, lambda_cluster, a_sigma_cluster,
                                  b_sigma_cluster);
      }
    
      log_pred += std::log(nk);
      alloc_prob.col(1).row(k).fill(log_pred);
    }
  
    // Calculate the predictive probability and normalize it.
    alloc_prob.col(2) = log_sum_exp(alloc_prob.col(1));
    
    // Sample a new cluster based on the predictive probability
    int index_new_cluster = rmultinom_1(alloc_prob.col(2), 2);
    new_assign.row(j).fill(alloc_prob.col(0)[index_new_cluster]);
  }
  
  return new_assign;
}

// [[Rcpp::export]]
double log_proposal(arma::vec c1, arma::vec c2, arma::vec y, 
                    arma::vec xi_cluster, arma::vec mu0_cluster, 
                    arma::vec lambda_cluster, arma::vec a_sigma_cluster, 
                    arma::vec b_sigma_cluster, arma::vec sm_cluster, 
                    arma::uvec S_index){
  
  /* Description: This will calculate the proposal probability in the log scale.
   *              Based on the SM paper, we will consider only the observations
   *              which are in S.
   * Output: log proposal probability, log(q(c1|c2)).
   * Input: Two lists of the cluster assignment (c1, c2), the set of the
   *        observation that we are interested in SM procedure (S), the list of
   *        interested cluster (s_clus), data (y), cluster concentration (xi),
   *        data hyperparameters (a_sigma, b_sigma, lambda, mu0)
   */

  double result = 0.0;
  arma::vec inter_assign(c2);
  
  // Loop through the observations in the S
  for(int i = 0; i < S_index.size(); ++i){
    
    int j = S_index[i];
    
    // Select y_current (as a y_new)
    arma::vec y_current = y.row(j);
    // Create vector for y_not_i and cluster_not_i
    arma::vec y_not_i(y);
    arma::vec assign_not_i(inter_assign);
    y_not_i.shed_row(j);
    assign_not_i.shed_row(j);
    
    // Create a matrix for collecting the allocation probability
    // (row: clusters, column: (cluster index, log_predictive, predictive))
    arma::mat alloc_prob(2, 3, arma::fill::value(-1000));
    
    // Loop through all possible cluster
    for(int k = 0; k < 2; ++k){
      int cc = sm_cluster[k];
      alloc_prob.col(0).row(k).fill(cc);
      arma::uvec obs_index = arma::find(assign_not_i == cc);
      
      double nk = obs_index.size();
      double log_pred = 0.0;
      
      if(nk == 0){
        // If there are no observations in that cluster, the predictive is just
        // a marginal distribution.
        
        log_pred += log_marginal(y_current, cc, mu0_cluster, lambda_cluster, 
                                 a_sigma_cluster, b_sigma_cluster);
        } else {
        
        log_pred += log_posterior(y_current, y_not_i.rows(obs_index), cc,
                                  mu0_cluster, lambda_cluster, a_sigma_cluster,
                                  b_sigma_cluster);
     }
     
     log_pred += std::log(nk);
     alloc_prob.col(1).row(k).fill(log_pred);
   }
   
   // Calculate the predictive probability and normalize it.
   alloc_prob.col(2) = log_sum_exp(alloc_prob.col(1));
   
   // Calculate the log proposal for this observation
   arma::uvec c1_cc = arma::find(alloc_prob.col(0) == c1[j]);
   arma::vec prob_c = alloc_prob.submat(c1_cc, arma::uvec(1, arma::fill::value(2)));
   result += std::log(prob_c[0]);
   
   // Replace the current observation with cluster assignment in c1
   inter_assign.row(j).fill(c1[j]);
 }
  
  return result;
}

// [[Rcpp::export]]
double log_prior_cluster(arma::vec cluster_assign, arma::vec xi_cluster){
  
  /* Description: -
   * Output: -
   * Input: -
   */
  
  double result = 0.0; 
  
  // Get the list of active cluster
  arma::uvec active_clus = arma::conv_to<arma::uvec>::from(arma::unique(cluster_assign));
  
  // Get the required information
  arma::vec nk_active(active_clus.size(), arma::fill::value(-1));
  for(int k = 0; k < active_clus.size(); ++k){
    int clus = active_clus[k];
    arma::uvec nk_obs = arma::find(cluster_assign == clus);
    nk_active.row(k).fill(nk_obs.size());
  }
  arma::vec xi_active = xi_cluster.rows(active_clus);
  
  // Calculate p(c|phi)
  result += std::lgamma(arma::accu(nk_active) + 1);
  result -= arma::accu(arma::lgamma(nk_active + 1));
  result += std::lgamma(arma::accu(xi_active));
  result -= std::lgamma(arma::accu(xi_active + nk_active));
  result += arma::accu(arma::lgamma(xi_active + nk_active));
  result -= arma::accu(arma::lgamma(xi_active));
  
  return result;
}

// [[Rcpp::export]]
Rcpp::List SFDM_realloc(arma::vec old_assign, arma::vec y, arma::vec alpha_vec,
                        arma::vec mu0_cluster, arma::vec lambda_cluster, 
                        arma::vec a_sigma_cluster, arma::vec b_sigma_cluster, 
                        arma::vec xi_cluster){
  
  /* Description: This function will perform the reallocation step in SFDM. This
   *              is a first step of sparse finite discrete mixture model (SFDM). 
   * Output: The vector of new cluster assignment after performing reallocation 
   *         step and alpha vector.
   * Input: -
   */
  
  Rcpp::List result_realloc;
  
  arma::vec new_assign(old_assign);
  arma::vec active_cluster = arma::unique(old_assign);
  int K_pos = active_cluster.size();
  
  // Loop through the observations
  for(int i = 0; i < y.size(); ++i){
    
    // Select y_current (as a y_new)
    arma::vec y_current = y.row(i);
    // Create vector for y_not_i and cluster_not_i
    arma::vec y_not_i(y);
    arma::vec assign_not_i(new_assign);
    y_not_i.shed_row(i);
    assign_not_i.shed_row(i);
    
    // Create a matrix for collecting the allocation probability
    // (row: clusters, column: (cluster index, log_predictive, predictive))
    arma::mat alloc_prob(K_pos, 3, arma::fill::value(-1000));
    
    // Loop through all possible cluster
    for(int k = 0; k < K_pos; ++k)
    {
      int cc = active_cluster[k];
      
      alloc_prob.col(0).row(k).fill(cc);
      arma::uvec obs_index = arma::find(assign_not_i == cc);
      
      double nk = obs_index.size();
      double log_pred = 0.0;
      
      if(nk == 0){
        // If there are no observations in that cluster, the predictive is just
        // a marginal distribution.
        log_pred += log_marginal(y_current, cc, mu0_cluster, lambda_cluster,
                                 a_sigma_cluster, b_sigma_cluster);
      } else {
        log_pred += log_posterior(y_current, y_not_i.rows(obs_index), cc,
                                  mu0_cluster, lambda_cluster, a_sigma_cluster,
                                  b_sigma_cluster);
      }
      
      log_pred += std::log(nk + xi_cluster[cc]);
      alloc_prob.col(1).row(k).fill(log_pred);
    }

    // Calculate the predictive probability and normalize it.
    alloc_prob.col(2) = log_sum_exp(alloc_prob.col(1));

    // Sample a new cluster based on the predictive probability
    int index_new_cluster = rmultinom_1(alloc_prob.col(2), K_pos);
    new_assign.row(i).fill(alloc_prob.col(0)[index_new_cluster]);
  }
  
  // Adjusted the alpha vector
  arma::vec new_alpha = adjust_alpha(new_assign, alpha_vec); 
  
  // Record the result
  result_realloc["new_assign"] = new_assign;
  result_realloc["new_alpha"] = new_alpha;
  return result_realloc;
  
}

// [[Rcpp::export]]
Rcpp::List SFDM_SM(int K, arma::vec old_assign, arma::vec y, arma::vec alpha_vec,
                   arma::vec mu0_cluster, arma::vec lambda_cluster, 
                   arma::vec a_sigma_cluster, arma::vec b_sigma_cluster, 
                   arma::vec xi_cluster, int launch_iter, double a_theta, 
                   double b_theta){
  
  /* Description: -
   * Output: -
   * Input: -
   */
  
  Rcpp::List sm_result;
  
  // (0) Prepare for the algorithm
  int n = old_assign.size(); // Number of the observations
  arma::vec active_clus = arma::unique(old_assign);
  int K_active = active_clus.size();
  int split_ind = -1; // Indicate that we will perform split or merge
  arma::vec launch_assign(old_assign); // Assignment from the launch step
  arma::vec launch_alpha(alpha_vec); // Alpha from the launch step
  int accept_proposed = 0; // Indicator for accepting the proposed assignment.
  double log_A = 0.0; // log of acceptance probability
  
  // (1) Select two observations to determine whether we will split or merge
  arma::uvec samp_obs = arma::randperm(n, 2); // Select two observations
  // If all cluster is already active, we can perform only merge.
  while((K_active == K) and (old_assign[samp_obs[0]] == old_assign[samp_obs[1]])){
    samp_obs = arma::randperm(n, 2); // sample two observations again until we get merge.
  }
  arma::vec samp_clus = old_assign.rows(samp_obs);
  int new_clus = -1;
  // Create the split indicator. If we split, split_ind = 1. Otherwise, split_ind = 0.
  if(samp_clus[0] == samp_clus[1]){
    split_ind = 1; // Split
    
    arma::uvec inactive_index = arma::find(alpha_vec == 0);
    arma::uvec new_clus_index = arma::randperm(inactive_index.size(), 1);
    new_clus = inactive_index[new_clus_index[0]]; // new active cluster
    samp_clus.row(0).fill(new_clus);
    launch_assign.row(samp_obs[0]).fill(new_clus); // set ci_launch to be a new cluster
    launch_alpha.row(new_clus).fill(R::rgamma(xi_cluster[new_clus], 1.0));
  } else {
    split_ind = 0;
  }

  // (2) Create a set S := {same cluster as samp_obs, but not index_samp}
  arma::uvec S = arma::find(old_assign == old_assign[samp_obs[0]] or old_assign == old_assign[samp_obs[1]]);
  arma::uvec samp_obs_index = arma::find(S == samp_obs[0] or S == samp_obs[1]);
  S.shed_rows(samp_obs_index);
  
  // (3) Perform a launch step
  // Randomly assign observation in S to be ci_launch or cj_launch.
  arma::vec init_ind = arma::randu(S.size(), arma::distr_param(0, 1));
  launch_assign.rows(S) = samp_clus.rows(arma::conv_to<arma::uvec>::from(init_ind > 0.5));
  
  // Perform a launch step
  arma::vec dummy_launch(launch_assign.size(), arma::fill::value(-1));
  for(int t = 0; t < launch_iter; ++t){
    dummy_launch = split_launch(launch_assign, y, mu0_cluster, lambda_cluster, 
                                a_sigma_cluster, b_sigma_cluster, samp_clus, S);
    launch_assign = dummy_launch;
  }
  
  // (4) Split-Merge
  arma::vec proposed_assign(launch_assign);
  arma::vec proposed_alpha(launch_alpha);
  
  if(split_ind == 1){
    // Split: Perform another launch iteration
    proposed_assign = split_launch(launch_assign, y, mu0_cluster, lambda_cluster, 
                                   a_sigma_cluster, b_sigma_cluster, samp_clus, S);
    log_A += (std::log(a_theta) - std::log(b_theta)); // Spike-and-slab
    // Proposal Probability
    log_A += log_proposal(launch_assign, proposed_assign, y, xi_cluster, 
                          mu0_cluster, lambda_cluster, a_sigma_cluster, 
                          b_sigma_cluster, samp_clus, S);
    log_A -= log_proposal(proposed_assign, launch_assign, y, xi_cluster, 
                          mu0_cluster, lambda_cluster, a_sigma_cluster, 
                          b_sigma_cluster, samp_clus, S);
  } else {
    // Merge: All observations in S and {ci, cj} will be allocated to cj
    proposed_assign.rows(S).fill(samp_clus[1]);
    proposed_assign.rows(samp_obs).fill(samp_clus[1]);
    log_A += (std::log(b_theta) - std::log(a_theta)); // Spike-and-slab
    // Proposal Probability
    log_A += log_proposal(old_assign, launch_assign, y, xi_cluster, 
                          mu0_cluster, lambda_cluster, a_sigma_cluster, 
                          b_sigma_cluster, samp_clus, S);
  }
  
  // (5) Calculate the acceptance probability
  arma::vec proposed_active = arma::unique(proposed_assign);
  for(int j = 0; j < proposed_active.size(); ++j){
    int proposed_now = proposed_active[j];
    log_A += log_marginal(y.rows(arma::find(proposed_assign == proposed_now)), 
                          proposed_now, mu0_cluster, lambda_cluster, 
                          a_sigma_cluster, b_sigma_cluster);
  }
  
  arma::vec old_active = arma::unique(old_assign);
  for(int j = 0; j < old_active.size(); ++j){
    int old_now = old_active[j];
    log_A -= log_marginal(y.rows(arma::find(old_assign == old_now)), 
                          old_now, mu0_cluster, lambda_cluster, 
                          a_sigma_cluster, b_sigma_cluster);
  }

  log_A += log_prior_cluster(proposed_assign, xi_cluster);
  log_A -= log_prior_cluster(old_assign, xi_cluster);
  
  arma::vec new_assign(old_assign);
  
  if(std::log(R::runif(0.0, 1.0)) < std::min(0.0, log_A)){
    // Accept the proposed vector
    accept_proposed = 1;
    new_assign = proposed_assign;
  }
  
  // (6) Update alpha vector
  arma::vec new_alpha = adjust_alpha(new_assign, proposed_alpha);
  
  sm_result["split_ind"] = split_ind;
  sm_result["log_A"] = log_A;
  sm_result["accept_proposed"] = accept_proposed;
  sm_result["new_assign"] = new_assign;
  sm_result["new_alpha"] = new_alpha;
  
  return sm_result;
} 

// [[Rcpp::export]]
Rcpp::List SFDM_alpha(arma::vec clus_assign, arma::vec xi_cluster, 
                      arma::vec alpha_vec, double old_u){
  
  /* Description: This function will update the alpha vector, and the auxiliary 
   *              variable. The derivation of this function is based on Matt's 
   *              Dirichlet Data Augmentation Trick document.
   * Output: An updated auxiliary variable (new_u) and alpha vector (new_alpha).
   * Input: The cluster assignment (clus_assign), cluster concentration (xi), 
   *        previous alpha vector (old_alpha), previous auxiliary variable (old_u).
   */
  
  Rcpp::List result;
  arma::vec new_alpha(alpha_vec); 
  
  // Update alpha
  arma::vec active_clus = arma::unique(clus_assign);
  for(int k = 0; k < active_clus.size(); ++k){
    int current_c = active_clus[k];
    arma::uvec nk = arma::find(clus_assign == current_c);
    double scale_gamma = 1/(1 + old_u); // change the rate to scale parameter
    new_alpha.row(current_c).fill(R::rgamma(nk.size() + xi_cluster[current_c], scale_gamma));
  }
  
  // Update U
  int n = clus_assign.size();
  double scale_u = 1/arma::accu(new_alpha);
  double new_U = R::rgamma(n, scale_u);
  
  result["new_alpha"] = new_alpha;
  result["new_u"] = new_U;
  
  return result;
  
}

// Final Function: -------------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List SFDM_model(int iter, int K, arma::vec init_assign, arma::vec y, 
                      arma::vec mu0_cluster, arma::vec lambda_cluster, 
                      arma::vec a_sigma_cluster, arma::vec b_sigma_cluster, 
                      arma::vec xi_cluster, double a_theta, double b_theta, 
                      int launch_iter, int print_iter){

  /* Description: This function runs our model.
   * Output: A list of the result consisted of (1) the cluster assignment for
   *         each iterations (iter x size of data matrix) (2) split/merge
   *         decision for each iteration (split_or_merge) (3) accept or reject
   *         SM status (sm_status)0
   * Input: Number of iteration for runiing the model (iter) Maximum possible
   *        cluster (K), the initial assignment (init_assign),
   *        cluster concentration (xi), data (y),
   *        data hyperparameters (mu0, a_sigma, b_sigma, lambda),
   *        spike-and-slab hyperparameters (a_theta, b_theta),
   *        number of iteration for the launch step (sm_iter),
   *        progress report (print_iter)
   */

  Rcpp::List result;

  // Initial alpha
  arma::uvec init_unique = arma::conv_to<arma::uvec>::from(arma::unique(init_assign));
  arma::vec init_alpha(K, arma::fill::zeros);

  for(int k = 0; k < init_unique.size(); ++k){
    init_alpha.row(init_unique[k]).fill(R::rgamma(xi_cluster[init_unique[k]], 1.0));
  }

  // Initial u (auxiliary variable)
  double init_u = R::rgamma(y.size(), 1/(arma::accu(init_alpha)));

  // Objects for storing the intermediate result
  Rcpp::List alloc_List;
  Rcpp::List sm_List;
  Rcpp::List alpha_update_List;

  // Create vectors/matrices for storing the final result
  arma::mat iter_assign(y.size(), iter, arma::fill::value(-1));
  arma::mat iter_alpha(K, iter, arma::fill::value(-1));
  arma::vec sm_status(iter, arma::fill::value(-1));
  arma::vec iter_log_A(iter, arma::fill::value(-1));
  arma::vec split_or_merge(iter, arma::fill::value(-1));

  // Perform an algorithm
  for(int i = 0; i < iter; ++i){
    // Reallocation Step
    alloc_List = SFDM_realloc(init_assign, y, init_alpha, mu0_cluster, 
                              lambda_cluster, a_sigma_cluster, b_sigma_cluster, 
                              xi_cluster);
    arma::vec realloc_assign = alloc_List["new_assign"];
    arma::vec realloc_alpha = alloc_List["new_alpha"];

    // Split-Merge Step
    sm_List = SFDM_SM(K, realloc_assign, y, realloc_alpha, mu0_cluster, 
                      lambda_cluster, a_sigma_cluster, b_sigma_cluster, 
                      xi_cluster, launch_iter, a_theta, b_theta);
    arma::vec sm_assign = sm_List["new_assign"];
    arma::vec sm_alpha = sm_List["new_alpha"];

    // Update alpha vector (and u)
    alpha_update_List = SFDM_alpha(sm_assign, xi_cluster, sm_alpha, init_u);
    arma::vec alpha_updated = alpha_update_List["new_alpha"];
    double u_updated = alpha_update_List["new_u"];

    // Store the result
    iter_assign.col(i) = sm_assign;
    iter_alpha.col(i) = alpha_updated;
    iter_log_A.row(i).fill(sm_List["log_A"]);
    split_or_merge.row(i).fill(sm_List["split_ind"]);
    sm_status.row(i).fill(sm_List["accept_proposed"]);

    // Initialize for the next iteration
    init_assign = sm_assign;
    init_alpha = alpha_updated;
    init_u = u_updated;

    // Print the result
    if(((i + 1) - (floor((i + 1)/print_iter) * print_iter)) == 0){
      std::cout << "Iter: " << (i+1) << " - Done!" << std::endl;
    }

  }

  result["iter_assign"] = iter_assign.t();
  result["iter_alpha"] = iter_alpha.t();
  result["log_A"] = iter_log_A;
  result["sm_status"] = sm_status;
  result["split_or_merge"] = split_or_merge;

  return result;
}

// END: ------------------------------------------------------------------------