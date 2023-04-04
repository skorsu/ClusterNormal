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
  
  // Select the parameter for each observation based on its cluster.
  arma::vec mu0_k = mu0.rows(arma::conv_to<arma::uvec>::from(clus_assign - 1));
  arma::vec a_sigma_k = a_sigma.rows(arma::conv_to<arma::uvec>::from(clus_assign - 1));
  arma::vec b_sigma_k = b_sigma.rows(arma::conv_to<arma::uvec>::from(clus_assign - 1));
  arma::vec lambda_k = lambda.rows(arma::conv_to<arma::uvec>::from(clus_assign - 1));
  
  // Intermediate Calculation
  arma::vec denom(b_sigma_k);
  denom += 0.5 * (lambda_k % arma::pow(y - mu0_k, 2) / (lambda_k + 1));
  
  // Calculate the log marginal for each observation
  arma::vec marginal_vec(y.size(), arma::fill::zeros);
  marginal_vec -= (0.5 * std::log(2 * pi));
  marginal_vec += (0.5 * arma::log(lambda_k));
  marginal_vec -= (0.5 * arma::log(lambda_k + 1));
  marginal_vec += arma::lgamma(0.5 + a_sigma_k);
  marginal_vec -= arma::lgamma(a_sigma_k);
  marginal_vec += (a_sigma_k % arma::log(b_sigma_k));
  marginal_vec -= (a_sigma_k + 0.5) % arma::log(denom);
  
  result = arma::accu(marginal_vec);
  return result;
}

// [[Rcpp::export]]
double log_cluster_param(arma::vec clus_assign, arma::vec alpha){
  
  /* Description: This will calculate the log probability of the clusters.
   * Output: log of the cluster probability
   * Input: list of the active cluster (clus_assign), 
   *        cluster parameter (alpha)
   */
  
  arma::uvec unique_clus = arma::conv_to<arma::uvec>::from(arma::unique(clus_assign)); 
  double result = -(clus_assign.size() * std::log(arma::accu(alpha.rows(unique_clus - 1))));
  result += R::lgammafn(clus_assign.size() + 1);
  
  for(int k = 0; k < unique_clus.size(); ++k){
    arma::uvec nk = arma::find(clus_assign == unique_clus[k]);
    result += (nk.size() * std::log(alpha[unique_clus[k] - 1]));
    result -= R::lgammafn(nk.size() + 1);
  }
  
  return result;
}

// [[Rcpp::export]]
double log_gamma_cluster(arma::vec alpha, arma::vec xi){
  
  /* Description: This will calculate the log probability of the clusters.
   * Output: log of the cluster probability
   * Input: cluster parameter (alpha), cluster concentration (xi)
   */
  
  double result = 0.0;
  for(int k = 0; k < xi.size(); ++k){
    if(alpha[k] != 0){
      result += R::dgamma(alpha[k], xi[k], 1.0, 1);
    }
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
  arma::vec launch_alpha(old_alpha);
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
    launch_alpha.row(candi_clus - 1).fill(R::rgamma(xi[(candi_clus - 1)], 1.0));
    launch_assign[index_samp[0]] = candi_clus; // Set ci to a new cluster.
  } else{
    split_ind = 0;
  }

  S_clus[0] = launch_assign[index_samp[0]]; // This is ci
  S_clus[1] = launch_assign[index_samp[1]]; // This is cj

  // (1.4) Create an initial step
  launch_assign.row(index_samp[0]).fill(S_clus[0]);
  launch_assign.row(index_samp[1]).fill(S_clus[1]);
  for(int i = 0; i < S_index.size(); ++i){
    // Randomly assign the observation of S_index to one of the ci or cj.
    launch_assign.row(S_index[i]).fill(S_clus[round(arma::randu())]);
  }
  
  // (1.5) Perform a launch step
  for(int t = 0; t < sm_iter; ++t){
    for(int i = 0; i < S_index.size(); ++i){
      arma::mat obs_i_alloc = log_alloc_prob(S_index[i], S_clus, launch_assign, 
                                             xi, y, a_sigma, b_sigma, lambda, mu0);
      launch_assign.row(S_index[i]).fill(samp_new(obs_i_alloc));
    }
  }
  
  // (1.6) Adjust the alpha vector for the launch step
  for(int k = 1; k <= K; ++k){
    arma::uvec n_c = arma::find(launch_assign == k);
    if(n_c.size() == 0){
      launch_alpha.row(k-1).fill(0.0);
    }
  }
  
  // (2) Split-Merge
  // (2.1) Perform a Split-Merge step
  arma::vec proposed_assign(launch_assign);
  arma::vec proposed_alpha(launch_alpha);

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
  
  // (2.2) Adjust the alpha vector for the launch step
  for(int k = 1; k <= K; ++k){
    arma::uvec n_c = arma::find(proposed_assign == k);
    if(n_c.size() == 0){
      proposed_alpha.row(k-1).fill(0.0);
    }
  }

  // (3) Acceptance Step (MH algorithm)
  // Note: Compare proposed with launch
  int accept_new = 0;

  // (3.1) Calculate the log accpetance probability
  double log_A = 0.0;
  log_A += log_marginal_y(proposed_assign, y, mu0, a_sigma, b_sigma, lambda);
  log_A -= log_marginal_y(launch_assign, y, mu0, a_sigma, b_sigma, lambda);
  log_A += log_cluster_param(proposed_assign, proposed_alpha);
  log_A += log_cluster_param(launch_assign, launch_alpha);
  log_A += log_gamma_cluster(proposed_alpha, xi);
  log_A -= log_gamma_cluster(launch_alpha, xi);
  log_A += ((2 * split_ind) - 1) * (std::log(a_theta) - std::log(b_theta));
  log_A -= (((2 * split_ind) - 1) * (S_index.size() * std::log(0.5)));
  
  // (3.2) Accept the proposed or still use the old assign
  arma::vec new_assign(old_assign);
  arma::vec new_alpha(old_alpha);
  
  if(log(R::runif(0.0, 1.0)) < log_A){
    new_assign = proposed_assign;
    new_alpha = proposed_alpha;
    accept_new = 1;
  }
  
  result["split_ind"] = split_ind;
  result["accept_new"] = accept_new;
  result["new_assign"] = new_assign;
  result["new_alpha"] = new_alpha;
  
  return result;
}

// Final Function: -------------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List our_model(int iter, int K, arma::vec init_assign, arma::vec xi, 
                     arma::vec y, arma::vec mu0, arma::vec a_sigma,
                     arma::vec b_sigma, arma::vec lambda, double a_theta,
                     double b_theta, int sm_iter){
  
  /* Description: -
   * Output: -
   * Input: -
   */
  
  Rcpp::List result;
  
  // Initial alpha
  arma::uvec init_unique = arma::conv_to<arma::uvec>::from(arma::unique(init_assign));
  arma::vec init_alpha(K, arma::fill::zeros);
  
  for(int k = 0; k < init_unique.size(); ++k){
    init_alpha.row(init_unique[k] - 1).fill(R::rgamma(xi[init_unique[k] - 1], 1.0));
  }
  
  // Create vectors/matrices for storing the final result 
  arma::mat iter_assign(y.size(), iter, arma::fill::value(-1));
  arma::vec sm_status(iter, arma::fill::value(7));
  arma::vec split_or_merge(iter, arma::fill::value(12));
  
  // Perform an algorithm
  for(int i = 0; i < iter; ++i){
    Rcpp::List alloc_List = our_allocate(init_assign, xi, y, a_sigma, b_sigma, 
                                         lambda, mu0, init_alpha);
    arma::vec alloc_assign = alloc_List["new_assign"];
    arma::vec alloc_alpha = alloc_List["new_alpha"];
    
    Rcpp::List sm_List = our_SM(K, alloc_assign, alloc_alpha, xi, y, mu0, a_sigma, 
                                b_sigma, lambda, a_theta, b_theta, sm_iter);
    arma::vec sm_assign = sm_List["new_assign"];
    arma::vec sm_alpha = sm_List["new_alpha"];
    
    iter_assign.col(i) = sm_assign;
    split_or_merge.row(i).fill(sm_List["split_ind"]);
    sm_status.row(i).fill(sm_List["accept_new"]);
    
    init_assign = sm_assign;
    init_alpha = sm_alpha;
  }
  
  result["iter_assign"] = iter_assign.t();
  result["sm_status"] = sm_status;
  result["split_or_merge"] = split_or_merge;

  return result;
}

// END: ------------------------------------------------------------------------