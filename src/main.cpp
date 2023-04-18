#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

#define pi 3.141592653589793238462643383280

// Note to self: ---------------------------------------------------------------
// * FMM: Complete (updated on 3/22/2023)
// * Our model: First draft is done!

// User-defined function: ------------------------------------------------------
// [[Rcpp::export]]
arma::vec init_seq(int n, int K){
  arma::vec test_vec(n);
  int K_c = 1;
  for(int i = 0; i != n; ++i){
    test_vec[i] = (i - floor(i/K) * K) + 1;
  }
  
  return test_vec;
}

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

// Finite Mixture Model: -------------------------------------------------------
// [[Rcpp::export]]
arma::vec fmm_log_alloc_prob(int K, int i, arma::vec old_assign, arma::vec xi, 
                         arma::vec y, arma::vec a_sigma, arma::vec b_sigma, 
                         arma::vec lambda, arma::vec mu0){
  
  /* Description: This function will calculate the log allocation probability
   *              of the particular observation for all possible clusters. 
   *              This function is designed for the finite mixture model.
   * Output: log of the allocation probability for each cluster.
   * Input: Maximum possible cluster (K), index of the current observation (i),
   *        current cluster assignment (old_assign), cluster concentration (xi),
   *        data (y), data hyperparameters (a_sigma, b_sigma, lambda, mu0)
   */
  
  // Get the list of the active clusters
  arma::uvec active_clus;
  active_clus = arma::conv_to<arma::uvec>::from(arma::unique(old_assign));
  
  // Error Handling
  if(active_clus.size() > K){
    Rcpp::stop("The active clusters is more than the possible maximum clusters.");
  }
  if(xi.size() != K or a_sigma.size() != K or b_sigma.size() != K or 
       lambda.size() != K or mu0.size() != K){
    Rcpp::stop("The size of the hyperparameter and K are not matched.");
  }
  
  arma::vec log_alloc = 100 * arma::ones(K);
  
  // Create the data vector which exclude the observation i.
  arma::vec y_not_i(y);
  y_not_i.shed_row(i);
  arma::vec c_not_i(old_assign);
  c_not_i.shed_row(i);
  
  // Calculate the log allocation probability for each cluster
  for(int c = 1; c <= K; ++c){ // Iterate on the cluster index
    arma::vec y_c(y_not_i);
    y_c.shed_rows(arma::find(c_not_i != c)); // data point in the current c
    int n_k = y_c.size(); // number of element in cluster c
    
    // Calculate the posterior parameters
    double a_n = a_sigma[(c-1)] + (n_k/2);
    double V_n = 1/(n_k + lambda[(c-1)]);
    double sum_y = 0.0;
    double b_n = b_sigma[(c-1)]; // if n_k = 0 then b_n = b_k;
    if(n_k != 0){
      sum_y += arma::accu(y_c);
      b_n += (0.5 * (n_k - 1) * arma::var(y_c)); // if n_k = 1, drop the second terms of b_k
      b_n += (0.5 * (n_k * lambda[(c-1)]) / (n_k + lambda[(c-1)])) * std::pow((sum_y/n_k) - mu0[(c-1)], 2.0);
    }
    double mu_n = (sum_y + ((lambda % mu0)[(c-1)]))/(n_k + lambda[(c-1)]);
    
    // The posterior predictive is scaled-t distribution.
    double sd_t = std::pow(b_n * (1 + V_n) / a_n, 0.5);
    double log_p = R::dt((y[i] - mu_n)/sd_t, (2 * a_n), 1);
    log_p -= std::log(sd_t);
    
    // The allocation probability needs to include log(n_k + xi_k).
    log_p += std::log(n_k + xi[(c-1)]);
    
    log_alloc.row((c-1)).fill(log_p);
  }
  
  return log_alloc;
}

// [[Rcpp::export]]
int fmm_samp_new(int K, arma::vec log_alloc){
  
  /* Description: This function will perform two things. The first is to 
   *              transform the log allocation probability back to the 
   *              probability by applying log-sum-exp trick. Secondly, it will 
   *              sample the new cluster based on the probability from the first
   *              step. This function is designed for the finite mixture model.
   * Output: new cluster assignment for the observation #i
   * Input: Maximum possible cluster (K), the allocation probability in the log 
   *        scale (log_alloc).
   */
  
  // Convert the log probability back to probability using log-sum-exp trick
  arma::vec prob = log_sum_exp(log_alloc);
  
  // Sample from the list of the active cluster using the aloocation probability 
  Rcpp::IntegerVector x_index = Rcpp::seq(1, K);
  Rcpp::NumericVector norm_prob = Rcpp::wrap(prob);
  Rcpp::IntegerVector x = Rcpp::sample(x_index, 1, false, norm_prob);
  return x[0];
}

// [[Rcpp::export]]
arma::mat fmm_mod(int t, int K, arma::vec old_assign, arma::vec xi, arma::vec y, 
                  arma::vec a_sigma, arma::vec b_sigma, arma::vec lambda, 
                  arma::vec mu0){
  
  /* Description: This function will perform a finite mixture model.
   * Output: a matrix of the cluster assignment. Each row represents each 
   *         iteration and each column represent each observation.
   * Input: Number of iteration (t), Maximum possible cluster (K), 
   *        the previous assignment (old_assign), cluster concentration (xi), 
   *        data (y), data hyperparameters (a_sigma, b_sigma, lambda, mu0)
   */
  
  arma::mat final_result = -1 * arma::ones(t, y.size());
  
  arma::vec new_assign(old_assign);
  for(int iter = 0; iter < t; ++iter){
    
    // Reassign the observation
    for(int i = 0; i < new_assign.size(); ++i){
      arma::vec obs_i_alloc = fmm_log_alloc_prob(K, i, new_assign, xi, y, 
                                                 a_sigma, b_sigma, lambda, mu0);
      new_assign.row(i).fill(fmm_samp_new(K, obs_i_alloc));
    }
    
    // Record the result for the iteration #iter
    final_result.row(iter) = new_assign.t();
  }
  
  return final_result;
}

// Updated Code: ---------------------------------------------------------------
// [[Rcpp::export]]
arma::mat log_alloc_prob(int i, arma::vec active_clus, arma::vec old_assign, 
                         arma::vec xi, arma::vec y, arma::vec a_sigma, 
                         arma::vec b_sigma, arma::vec lambda, arma::vec mu0){
  
  /* Description: This function is an adjusted `fmm_log_alloc_prob` function. 
   *              Instead of calculating the log allocation probability for all 
   *              possible cluster, we calculate only active clusters.
   * Output: A K by 2 matrix. Each row represents each active cluster. The 
   *         second column is the log of the allocation probability.
   * Input: Index of the current observation (i), 
   *        list of the active cluster (active_clus),
   *        current cluster assignment (old_assign), cluster concentration (xi),
   *        data (y), data hyperparameters (a_sigma, b_sigma, lambda, mu0)
   */
  
  // Get the list of the active clusters
  int K = active_clus.size();
  arma::mat log_alloc(K, 2, arma::fill::value(-1000));
  
  // Create the data vector which exclude the observation i.
  arma::vec y_not_i(y);
  y_not_i.shed_row(i);
  arma::vec c_not_i(old_assign);
  c_not_i.shed_row(i);
  
  // Calculate the log allocation probability for each cluster
  for(int k = 0; k < K; ++k){
    int c = active_clus[k]; // select the current cluster
    log_alloc.row(k).col(0).fill(c);
    arma::vec y_c(y_not_i);
    y_c.shed_rows(arma::find(c_not_i != c)); // data point in the current c
    int n_k = y_c.size(); // number of element in cluster c

    // Calculate the posterior parameters
    double a_n = a_sigma[(c-1)] + (n_k/2);
    double V_n = 1/(n_k + lambda[(c-1)]);
    double sum_y = 0.0;
    double b_n = b_sigma[(c-1)]; // if n_k = 0 then b_n = b_k;
    if(n_k != 0){
      sum_y += arma::accu(y_c);
      b_n += (0.5 * (n_k - 1) * arma::var(y_c)); // if n_k = 1, drop the second terms of b_k
      b_n += (0.5 * (n_k * lambda[(c-1)]) / (n_k + lambda[(c-1)])) * std::pow((sum_y/n_k) - mu0[(c-1)], 2.0);
    }
    double mu_n = (sum_y + ((lambda % mu0)[(c-1)]))/(n_k + lambda[(c-1)]);
    
    // The posterior predictive is scaled-t distribution.
    double sd_t = std::pow(b_n * (1 + V_n) / a_n, 0.5);
    double log_p = R::dt((y[i] - mu_n)/sd_t, (2 * a_n), 1);
    log_p -= std::log(sd_t);
    
    // The allocation probability needs to include log(n_k + xi_k).
    log_p += std::log(n_k + xi[(c-1)]);
    
    log_alloc.row(k).col(1).fill(log_p);
  }
  
  return log_alloc;
}

// [[Rcpp::export]]
int samp_new(arma::mat log_prob_mat){
  
  /* Description: This function is an adjusted `fmm_samp_new` function. Instead 
   *              of considering for all possible cluster, we consider only 
   *              active clusters.
   * Output: new cluster assignment for the observation #i
   * Input: a matrix resulted from the `log_alloc_prob` function. 
   *        (log_prob_mat)
   */
  
  // Convert the log probability back to probability using log-sum-exp trick
  arma::vec log_alloc = log_prob_mat.col(1);
  arma::vec prob = log_sum_exp(log_alloc);
  
  // Sample from the list of the active cluster using the aloocation probability 
  arma::vec ac = log_prob_mat.col(0);
  Rcpp::IntegerVector active_clus = Rcpp::wrap(ac);
  Rcpp::NumericVector norm_prob = Rcpp::wrap(prob);
  Rcpp::IntegerVector x = Rcpp::sample(active_clus, 1, false, norm_prob);
  
  return x[0];
}

// [[Rcpp::export]]
double log_marginal_y(arma::vec clus_assign, arma::vec y, arma::vec mu0, 
                      arma::vec a_sigma, arma::vec b_sigma, arma::vec lambda){
  
  /* Description: This will calculate the log marginal probability of the data.
   * Output: log marginal probability of the data.
   * Input: list of the active cluster (clus_assign), data (y),
   *        data hyperparameters (a_sigma, b_sigma, lambda, mu0)
   */
  
  double result = 0.0;
  
  // Hyperparameter for each observation
  arma::uvec ci = arma::conv_to<arma::uvec>::from(clus_assign);
  arma::vec a_s = a_sigma.rows(ci - 1);
  arma::vec b_s = b_sigma.rows(ci - 1);
  arma::vec l_s = lambda.rows(ci - 1);
  arma::vec mu0_s = mu0.rows(ci - 1);
  
  // Intermediate calculation for b_n
  arma::vec bn_s(b_s);
  bn_s += ((l_s % arma::pow(2 * (l_s + 1), -1)) % arma::pow(mu0_s - y, 2));
  
  // Calculate the log marginal probability for each observation
  arma::vec log_m(clus_assign.size(), arma::fill::value(-0.5 * std::log(2 * pi)));
  log_m += arma::lgamma(a_s + 0.5);
  log_m -= arma::lgamma(a_s);
  log_m += (a_s % arma::log(b_s));
  log_m -= ((a_s + 0.5) % arma::log(bn_s));
  log_m += (0.5 * arma::log(l_s));
  log_m -= (0.5 * arma::log(l_s + 1));
  
  // Calculate the log marginal probability for overall
  result += arma::accu(log_m);
  
  return result;
}

// [[Rcpp::export]]
double log_cluster_param(arma::vec clus_assign, arma::vec alpha){
  
  /* Description: This will calculate the log probability of the clusters.
   * Output: log of the cluster probability
   * Input: list of the active cluster (clus_assign), 
   *        cluster parameter (alpha)
   */
  
  double result = 0.0;
  arma::uvec active_clus = arma::conv_to<arma::uvec>::from(arma::unique(clus_assign));
  
  for(int k = 0; k < active_clus.size(); ++k){
    int current_clus = active_clus[k];
    arma::uvec n_k = arma::find(clus_assign == current_clus);
    result += (n_k.size() * std::log(alpha[current_clus - 1]));
  }
  
  return result;
}

// [[Rcpp::export]]
double log_gamma_cluster(arma::vec alpha, arma::vec xi, arma::vec clus_assign){
  
  /* Description: This will calculate the log probability of the clusters.
   * Output: log of the cluster probability
   * Input: cluster parameter (alpha), cluster concentration (xi), cluster 
   *        assignment (clus_assign)
   */
  
  double result = 0.0;
  arma::uvec active_clus = arma::conv_to<arma::uvec>::from(arma::unique(clus_assign));
  
  for(int k = 0; k < active_clus.size(); ++k){
    int current_clus = active_clus[k];
    result += R::dgamma(alpha[current_clus - 1], xi[current_clus - 1], 1.0, 1);
  }
  
  return result;
}

// Step 1: Update the cluster space: -------------------------------------------
// * Note: Omitted this step for now

// Rcpp::List uni_expand(int K, arma::vec old_assign, arma::vec alpha,
//                       arma::vec xi, arma::vec y, arma::mat ldata, 
//                       double a_theta, double b_theta){
//   
//   Rcpp::List result;
//   double accept_iter = 0.0;
//   
//   // Indicate the existed clusters and inactive clusters
//   Rcpp::List List_clusters = active_inactive(K, old_assign);
//   arma::uvec inactive_clus = List_clusters["inactive"];
//   arma::uvec active_clus = List_clusters["active"];
//   
//   arma::vec new_alpha(K);
//   arma::vec new_assign(y.size());
//   
//   if(active_clus.size() == K){
//     new_alpha = alpha;
//     new_assign = old_assign;
//   } else {
//     // Select a candidate cluster
//     arma::vec samp_prob = arma::ones(inactive_clus.size())/inactive_clus.size();
//     int candidate_clus = sample_clus(samp_prob, inactive_clus);
//     
//     // Sample alpha for new active cluster
//     double alpha_k = 
//       arma::as_scalar(arma::randg(1, arma::distr_param(xi.at(candidate_clus - 1), 1.0)));
//     double sum_alpha = arma::sum(alpha);
//     double sum_alpha_k = sum_alpha + alpha_k;
//     
//     for(int i = 0; i < y.size(); ++i){
//       int cc = old_assign.at(i);
//       arma::rowvec ldata_y = ldata.row(i);
//       double log_a = std::min(0.0, ldata_y.at(candidate_clus - 1) - ldata_y.at(cc - 1) +
//                               log(alpha_k) - log(alpha.at(cc - 1)) + log(sum_alpha) - log(sum_alpha_k) +
//                               log(a_theta) - log(b_theta));
//       double log_u = log(arma::randu());
//       if(log_u <= log_a){
//         accept_iter += 1.0;
//         old_assign.row(i).fill(candidate_clus);
//       }
//     }
//     
//     new_assign = old_assign;
//     alpha.row(candidate_clus - 1).fill(alpha_k);
//     new_alpha = adjust_alpha(K, new_assign, alpha);
//   }
//   
//   result["accept_prob"] = accept_iter/y.size();
//   result["new_alpha"] = new_alpha;
//   result["new_assign"] = new_assign;
//   
//   return result;
// }


// Step 2: Allocate the observation to the existing clusters: ------------------
// [[Rcpp::export]]
Rcpp::List our_allocate(arma::vec old_assign, arma::vec xi, arma::vec y,
                        arma::vec a_sigma, arma::vec b_sigma, arma::vec lambda,
                        arma::vec mu0, arma::vec old_alpha){

  /* Description: This function will perform an adjusted finite mixture model
   *              for our model. It will reallocate the observation, and adjust
   *              the cluster space (alpha) in the end.
   * Output: The list of 2 vectors: (1) vector of an updated cluster assignment
   *         and (2) vector of an updated cluster space (alpha).
   * Input: Previous assignment (old_assign), cluster concentration (xi),
   *        data (y), data hyperparameters (a_sigma, b_sigma, lambda, mu0),
   *        cluster space parameter (old_alpha)
   */

  Rcpp::List result;

  arma::vec new_assign(old_assign);
  arma::vec new_alpha(old_alpha);
  arma::vec old_unique = arma::unique(old_assign);

  // Reassign the observation
  for(int i = 0; i < new_assign.size(); ++i){
    arma::mat obs_i_alloc = log_alloc_prob(i, old_unique, new_assign, xi, y, 
                                           a_sigma, b_sigma, lambda, mu0);
    new_assign.row(i).fill(samp_new(obs_i_alloc));
  }

  // Update alpha vector
  for(int k = 0; k < old_unique.size(); ++k){
    int c = old_unique[k];
    arma::uvec n_c = arma::find(new_assign == c);
    if(n_c.size() == 0){
      new_alpha.row(k).fill(0.0);
    }
  }

  result["new_assign"] = new_assign;
  result["new_alpha"] = new_alpha;

  return result;
}

// Step 3: Split-Merge: --------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List our_SM(int K, arma::vec old_assign, arma::vec old_alpha,
                  arma::vec xi, arma::vec y, arma::vec mu0, arma::vec a_sigma,
                  arma::vec b_sigma, arma::vec lambda, double a_theta,
                  double b_theta, int sm_iter){

  /* Description: This function will perform a split-merge procedure in our model.
   * Output: The list of 4 objects: (1) vector of an updated cluster assignment
   *         (2) vector of an updated cluster space (alpha),
   *         (3) split-merge result (1 is split, 0 is merge.)
   *         (4) accept result (1 is accept the split-merge, 0 is not.)
   * Input: Maximum possible cluster (K), the previous assignment (old_assign), 
   *        cluster concentration (xi), data (y), 
   *        data hyperparameters (a_sigma, b_sigma, lambda, mu0),
   *        spike-and-slab hyperparameters (a_theta, b_theta), 
   *        number of iteration for the launch step (sm_iter)
   */

  Rcpp::List result;

  // (1) Perform a Launch step
  arma::vec launch_assign(old_assign);
  arma::vec all_alpha(old_alpha);
  Rcpp::IntegerVector active_clus = Rcpp::wrap((arma::unique(launch_assign)));
  Rcpp::IntegerVector all_clus = Rcpp::seq(1, K);
  Rcpp::IntegerVector inactive_clus = Rcpp::setdiff(all_clus, active_clus);
  int split_ind = -1; // If we split, split_ind = 1. If we merge, split_ind = 0.

  // (1.1) Choose two observations to decide between split and merge.
  Rcpp::IntegerVector n_vec = Rcpp::seq(0, launch_assign.size() - 1); // index of the all observations
  Rcpp::IntegerVector index_samp = Rcpp::sample(n_vec, 2);

  // (1.1 Extra) If our clusters are active, we cannot split further.
  while(active_clus.size() == K and // incorrect here.
          (launch_assign[index_samp[0]] == launch_assign[index_samp[1]])){
    index_samp = Rcpp::sample(n_vec, 2); // sampling until we have to merge
  }

  // (1.2) Create a set S := {same cluster as index_samp, but not index_samp}
  arma::uvec S_index = arma::find((launch_assign == launch_assign[index_samp[0]])
                                            or (launch_assign == launch_assign[index_samp[1]]));
  S_index.shed_rows(arma::find((S_index == index_samp[0]) or (S_index == index_samp[1])));

  // (1.3) Reindex followed by the SM paper
  arma::vec S_clus(2);
  if(launch_assign[index_samp[0]] == launch_assign[index_samp[1]]){
    // If they are from the same cluster, we will split.
    split_ind = 1;
    // Find the candidate cluster from the inactive list.
    int candi_clus = Rcpp::sample(inactive_clus, 1)[0];
    launch_assign[index_samp[0]] = candi_clus; // Set ci to a new cluster.
  } else{
    split_ind = 0;
  }

  S_clus[0] = launch_assign[index_samp[0]]; // This is ci
  S_clus[1] = launch_assign[index_samp[1]]; // This is cj
  
  // (1.4) Create an alpha vector for all active cluster
  arma::uvec launch_active = arma::conv_to<arma::uvec>::from(arma::unique(launch_assign));
  for(int j = 0; j < launch_active.size(); ++j){
    int cc = launch_active[j];
    if(all_alpha[cc - 1] == 0){
      all_alpha.row(cc - 1).fill(R::rgamma(xi[(cc - 1)], 1.0));
    }
  }

  // (1.5) Create an initial step
  launch_assign.row(index_samp[0]).fill(S_clus[0]);
  launch_assign.row(index_samp[1]).fill(S_clus[1]);
  for(int i = 0; i < S_index.size(); ++i){
    // Randomly assign the observation of S_index to one of the ci or cj.
    launch_assign.row(S_index[i]).fill(S_clus[round(arma::randu())]);
  }
  
  // (1.6) Perform a launch step
  for(int t = 0; t < sm_iter; ++t){
    for(int i = 0; i < S_index.size(); ++i){
      arma::mat obs_i_alloc = log_alloc_prob(S_index[i], S_clus, launch_assign, 
                                             xi, y, a_sigma, b_sigma, lambda, mu0);
      launch_assign.row(S_index[i]).fill(samp_new(obs_i_alloc));
    }
  }
  
  // (2) Split-Merge
  // (2.1) Perform a Split-Merge step
  arma::vec proposed_assign(launch_assign);

  if(split_ind == 0){
    // Merge
    proposed_assign.rows(S_index).fill(S_clus[1]);
    proposed_assign.row(index_samp[0]).fill(S_clus[1]);
    proposed_assign.row(index_samp[1]).fill(S_clus[1]);
  } else {
    // Split: Perform another round of allocation of S_index.
    for(int i = 0; i < S_index.size(); ++i){
      arma::mat obs_i_alloc = log_alloc_prob(S_index[i], S_clus, proposed_assign, 
                                             xi, y, a_sigma, b_sigma, lambda, mu0);
      proposed_assign.row(S_index[i]).fill(samp_new(obs_i_alloc));
    }
  }
  
  // (3) Acceptance Step (MH algorithm)
  // Note: Compare proposed with launch
  int accept_new = -1;

  // (3.1) Calculate the log accpetance probability
  double log_A = 0.0;
  
  // (3.1.1) The log marginal of the data
  log_A += log_marginal_y(proposed_assign, y, mu0, a_sigma, b_sigma, lambda);
  log_A -= log_marginal_y(launch_assign, y, mu0, a_sigma, b_sigma, lambda);
  
  // (3.1.2) The cluster assignment
  log_A += log_cluster_param(proposed_assign, all_alpha);
  log_A -= log_cluster_param(launch_assign, all_alpha);
  
  // (3.1.3) The cluster parameter
  log_A += log_gamma_cluster(all_alpha, xi, proposed_assign);
  log_A -= log_gamma_cluster(all_alpha, xi, launch_assign);
  
  // (3.1.4) Sparse-control from Beta-Binomial (spike-and-slab)
  log_A += ((2 * split_ind) - 1) * (std::log(a_theta) - std::log(b_theta));
  
  // (3.1.5) Proposal Density
  log_A -= (((2 * split_ind) - 1) * (S_index.size() * std::log(0.5)));
  
  // (3.2) Cluster Assignment
  arma::vec new_assign(y.size(), arma::fill::value(-1));
  
  if(log(R::runif(0.0, 1.0)) < log_A){
    new_assign = proposed_assign;
    accept_new = 1;
  } else {
    new_assign = old_assign;
    accept_new = 0;
  }
  
  // (3.3) Update alpha
  arma::vec new_alpha(K, arma::fill::zeros);
  arma::uvec new_unique = arma::conv_to<arma::uvec>::from(arma::unique(new_assign));
  for(int k = 0; k < new_unique.size(); ++k){
    int current_new = new_unique[k];
    new_alpha.row(current_new - 1).fill(all_alpha[current_new - 1]);
  }
  
  result["split_ind"] = split_ind;
  result["log_A"] = log_A;
  result["accept_new"] = accept_new;
  result["new_assign"] = new_assign;
  result["new_alpha"] = new_alpha;
  result["proposed_assign"] = proposed_assign;
  result["launch_assign"] = launch_assign;
  
  return result;
}

// Final Function: -------------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List our_model(int iter, int K, arma::vec init_assign, arma::vec xi, 
                     arma::vec y, arma::vec mu0, arma::vec a_sigma,
                     arma::vec b_sigma, arma::vec lambda, double a_theta,
                     double b_theta, int sm_iter, int print_iter){
  
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
    init_alpha.row(init_unique[k] - 1).fill(R::rgamma(xi[init_unique[k] - 1], 1.0));
  }
  
  // Objects for storing the intermediate result
  Rcpp::List alloc_List;
  Rcpp::List sm_List;
  
  // Create vectors/matrices for storing the final result 
  arma::mat iter_assign(y.size(), iter, arma::fill::value(-1));
  arma::mat all_alpha(K, iter, arma::fill::value(-1));
  arma::mat iter_launch(y.size(), iter, arma::fill::value(-1));
  arma::mat iter_proposed(y.size(), iter, arma::fill::value(-1));
  arma::vec sm_status(iter, arma::fill::value(7));
  arma::vec split_or_merge(iter, arma::fill::value(12));
  arma::vec log_A_vec(iter, arma::fill::value(1000));
  
  // Perform an algorithm
  for(int i = 0; i < iter; ++i){
    // Allocation Step
    alloc_List = our_allocate(init_assign, xi, y, a_sigma, b_sigma, lambda, mu0, 
                              init_alpha);
    arma::vec alloc_assign = alloc_List["new_assign"];
    arma::vec alloc_alpha = alloc_List["new_alpha"];
    
    // Split-Merge Step
    sm_List = our_SM(K, alloc_assign, alloc_alpha, xi, y, mu0, a_sigma, b_sigma, 
                     lambda, a_theta, b_theta, sm_iter);
    arma::vec sm_assign = sm_List["new_assign"];
    arma::vec sm_alpha = sm_List["new_alpha"];
    arma::vec sm_assign_launch = sm_List["launch_assign"];
    iter_launch.col(i) = sm_assign_launch;
    arma::vec sm_assign_proposed = sm_List["proposed_assign"];
    iter_proposed.col(i) = sm_assign_proposed;

    log_A_vec.row(i).fill(sm_List["log_A"]);
    
    iter_assign.col(i) = sm_assign;
    split_or_merge.row(i).fill(sm_List["split_ind"]);
    sm_status.row(i).fill(sm_List["accept_new"]);
    
    
    init_assign = sm_assign;
    init_alpha = sm_alpha;
    
    // Print the result
    if(((i + 1) - (floor((i + 1)/print_iter) * print_iter)) == 0){
      std::cout << "Iter: " << (i+1) << " - Done!" << std::endl;
    }
    
  }
  
  result["all_alpha"] = all_alpha;
  result["iter_assign"] = iter_assign.t();
  result["iter_launch"] = iter_launch.t();
  result["iter_proposed"] = iter_proposed.t();
  result["sm_status"] = sm_status;
  result["split_or_merge"] = split_or_merge;
  result["log_A_vec"] = log_A_vec;

  return result;
}

// END: ------------------------------------------------------------------------