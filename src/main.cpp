#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

#define pi 3.141592653589793238462643383280

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
                         arma::vec b_sigma, arma::vec lambda, arma::vec mu0,
                         bool restricted){
  
  /* Description: This function is an adjusted `fmm_log_alloc_prob` function. 
   *              Instead of calculating the log allocation probability for all 
   *              possible cluster, we calculate only active clusters.
   * Output: A K by 2 matrix. Each row represents each active cluster. The 
   *         second column is the log of the allocation probability.
   * Input: Index of the current observation (i), 
   *        list of the active cluster (active_clus),
   *        current cluster assignment (old_assign), cluster concentration (xi),
   *        data (y), data hyperparameters (a_sigma, b_sigma, lambda, mu0),
   *        perform a restricted Gibbs or not (appeared in SM paper as an 
   *        equation 3.14)
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
    
    if(restricted == true){
      // For the restricted, we will use n instead.
      log_p += std::log(n_k);
    } else {
      // The allocation probability needs to include log(n_k + xi_k).
      log_p += std::log(n_k + xi[(c-1)]);
    }
    
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
double log_marginal(arma::vec clus_assign, arma::vec y, arma::vec a_sigma, 
                    arma::vec b_sigma, arma::vec lambda, arma::vec mu0){
  
  double result = 0.0;
  
  /* Description: This is the function for calculating the marginal of the data.  
   * Output: log marginal probability.
   * Input: vector of the cluster assignment (clus_assign), data (y), 
   *        data hyperparameters (a_sigma, b_sigma, lambda, mu0).
   */
  
  arma::uvec ci = arma::conv_to<arma::uvec>::from(clus_assign);
  
  // Select the hyperparameters for each observation based on its cluster.
  arma::vec a_ci = a_sigma.rows(ci - 1);
  arma::vec b_ci = b_sigma.rows(ci - 1);
  arma::vec lambda_ci = lambda.rows(ci - 1);
  arma::vec mu0_ci = mu0.rows(ci - 1);
  
  // Calculate the marginal parameters
  arma::vec an = a_ci + 0.5;
  arma::vec inv_Vn = lambda_ci + 1;
  arma::vec mu_n = (y + (mu0_ci % lambda_ci)) / inv_Vn;
  arma::vec bn = arma::pow(y, 2) + 
    (arma::pow(mu0_ci, 2) % lambda_ci) - (arma::pow(mu_n, 2) % inv_Vn);
  bn *= 0.5;
  bn += b_ci;
  
  // Calculate the log marginal probability for each observation
  arma::vec lmar(clus_assign.size(), arma::fill::value(-0.5 * std::log(2 * pi)));
  lmar += arma::lgamma(an);
  lmar -= arma::lgamma(a_ci);
  lmar += (0.5 * arma::log(lambda_ci));
  lmar -= (0.5 * arma::log(inv_Vn));
  lmar += (a_ci % arma::log(b_ci));
  lmar -= (an % arma::log(bn));
  
  // Calculate the sum of the log marginal
  result += arma::accu(lmar);
  
  return result;
}


// [[Rcpp::export]]
double log_likelihood(arma::vec clus_assign, arma::vec y, arma::vec a_sigma, 
                      arma::vec b_sigma, arma::vec lambda, arma::vec mu0){
  
  /* Description: This is the function for calculating the likelihood. This is 
   *              based on the split-merge paper (Equation 3.7) 
   * Output: log likelihood
   * Input: vector of the cluster assignment (clus_assign), data (y), 
   *        data hyperparameters (a_sigma, b_sigma, lambda, mu0).
   */
  
  double result = 0.0;
  
  // Select only the active cluster
  arma::uvec active_clus = arma::conv_to<arma::uvec>::from(arma::unique(clus_assign));
  int K_p = active_clus.size();
  arma::vec an_pos(K_p, arma::fill::value(-1000));
  arma::vec mu_pos(K_p, arma::fill::value(-1000));
  arma::vec scale_pos(K_p, arma::fill::value(-1000));
  
  // Calculate the posterior parameters
  for(int k = 0; k < K_p; ++k){
    int c = active_clus[k]; // select the current cluster
    arma::vec y_c = y.rows(arma::find(clus_assign == c)); // data point in the current c
    int n_k = y_c.size(); // number of element in cluster c
    
    double a_n = a_sigma[(c-1)] + (n_k/2);
    double V_n = 1/(n_k + lambda[(c-1)]);
    double sum_y = 0.0;
    double b_n = b_sigma[(c-1)]; // if n_k = 0 then b_n = b_k;
    if(n_k != 0){
      sum_y += arma::accu(y_c);
      b_n += (0.5 * (n_k - 1) * arma::var(y_c)); // if n_k = 1, drop the second terms of b_k
      b_n += (0.5 * (n_k * lambda[(c-1)]) / (n_k + lambda[(c-1)])) * std::pow((sum_y/n_k) - mu0[(c-1)], 2.0);
    }
    
    // The posterior predictive is scaled-t distribution.
    double mu_n = (sum_y + ((lambda % mu0)[(c-1)]))/(n_k + lambda[(c-1)]);
    double sd_t = std::pow(b_n * (1 + V_n) / a_n, 0.5);
    
    an_pos.row(k).fill(a_n);
    mu_pos.row(k).fill(mu_n);
    scale_pos.row(k).fill(sd_t);
  }
  
  // Calculate the log-likelihood
  for(int i = 0; i < y.size(); ++i){
    arma::uvec ci = arma::find(clus_assign[i] == active_clus);
    result += R::dt((y[i] - mu_pos[ci[0]])/scale_pos[ci[0]], (2 * an_pos[ci[0]]), 1);
    result -= std::log(scale_pos[ci[0]]);
  }

  return result;
}

// [[Rcpp::export]]
double log_cluster_param(arma::vec clus_assign, arma::vec alpha){
  
  /* Description: This will calculate the log probability of the clusters.
   * Output: log of the cluster probability
   * Input: vector of the cluster assignment (clus_assign), 
   *        cluster parameter (alpha)
   */
  
  double result = 0.0;
  arma::uvec active_clus = arma::conv_to<arma::uvec>::from(arma::unique(clus_assign));
  arma::vec nk(active_clus.size(), arma::fill::value(-1));
  arma::vec alp(active_clus.size(), arma::fill::value(-1));
  
  for(int k = 0; k < active_clus.size(); ++k){
    int current_clus = active_clus[k];
    arma::uvec n_k = arma::find(clus_assign == current_clus);
    nk.row(k).fill(n_k.size());
    alp.row(k).fill(alpha[current_clus - 1]);
  }
  
  result += arma::accu(nk % arma::log(arma::normalise(alp)));
  
  return result;
}

// [[Rcpp::export]]
double log_gamma_cluster(arma::vec alpha, arma::vec xi, arma::vec clus_assign){
  
  /* Description: This will calculate the log probability of the clusters.
   * Output: log of the cluster probability
   * Input: cluster parameter (alpha), cluster concentration (xi),
   *        vector of the cluster assignment (clus_assign).
   */
  
  double result = 0.0;
  arma::vec active_clus = arma::unique(clus_assign);
  
  for(int k = 0; k < active_clus.size(); ++k){
    int c = active_clus[k];
    result += R::dgamma(alpha[(c - 1)], xi[(c - 1)], 1.0, 1);
  }
  
  return result;
}

// [[Rcpp::export]]
double log_proposal(arma::vec c1, arma::vec c2, arma::uvec S, arma::vec s_clus,
                    arma::vec y, arma::vec xi, arma::vec mu0, arma::vec a_sigma,
                    arma::vec b_sigma, arma::vec lambda){

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

  arma::vec init_assign(c2);
  arma::vec c2_unique = arma::unique(init_assign);

  for(int i = 0; i < S.size(); ++i){
    int c = S[i];
    arma::mat obs_i_alloc = log_alloc_prob(c, s_clus, init_assign, xi, y,
                                           a_sigma, b_sigma, lambda, mu0, true);
    arma::vec prob = log_sum_exp(obs_i_alloc.col(1));
    arma::mat info = arma::join_rows(obs_i_alloc, prob);
    arma::vec clus = info.col(0);
    arma::uvec index = arma::find(clus == c1[c]);
    arma::vec prob_c = info.submat(index, arma::uvec(1, arma::fill::value(2)));
    result += std::log(prob_c[0]);

    init_assign.row(c).fill(c1[c]);
  }

  return result;
}

// [[Rcpp::export]]
double log_prior(arma::vec clus_assign, arma::vec xi){
  
  /* Description: This is a function for calculating a log of the prior 
   *              distribution for the cluster assignment based on the 
   *              Split-Merge paper.
   * Output: log of prior.
   * Input: Cluster assignment (clus_assign), cluster concentration (xi)
   */
  
  double result = 0.0;
  
  arma::vec clus_vec_current(clus_assign.size(), arma::fill::value(-1));
  for(int i = 0; i < clus_assign.size(); ++i){
    int cc = clus_assign[i];
    clus_vec_current.row(i).fill(cc);
    
    arma::uvec nk = arma::find(clus_vec_current == cc);
    result += std::log(nk.size() + xi[(cc - 1)]);
  }
  
  arma::uvec active_clus = arma::conv_to<arma::uvec>::from(arma::unique(clus_assign));
  arma::vec ni = arma::regspace(1, clus_assign.size());
  result -= arma::accu(arma::log((ni - 1) + arma::accu(xi.rows(arma::find(active_clus - 1)))));

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
Rcpp::List SFDM_allocate(int K, arma::vec old_assign, arma::vec xi, arma::vec y,
                         arma::vec a_sigma, arma::vec b_sigma, arma::vec lambda,
                         arma::vec mu0, arma::vec old_alpha){

  /* Description: This function will perform an adjusted finite mixture model
   *              for our model. It will reallocate the observation, and adjust
   *              the cluster space (alpha) in the end.
   * Output: The list of 2 vectors: (1) vector of an updated cluster assignment
   *         and (2) vector of an updated cluster space (alpha).
   * Input: Maximum possible cluster (K), previous assignment (old_assign), 
   *        cluster concentration (xi), data (y), data hyperparameters (a_sigma,
   *        b_sigma, lambda, mu0), cluster space parameter (old_alpha)
   */

  Rcpp::List result;

  arma::vec new_assign(old_assign);
  arma::vec new_alpha(old_alpha);
  arma::vec old_unique = arma::unique(old_assign);

  // Reassign the observation
  for(int i = 0; i < new_assign.size(); ++i){
    arma::mat obs_i_alloc = log_alloc_prob(i, old_unique, new_assign, xi, y, 
                                           a_sigma, b_sigma, lambda, mu0, false);
    new_assign.row(i).fill(samp_new(obs_i_alloc));
  }

  // Update alpha vector
  for(int k = 1; k <= K; ++k){
    arma::uvec n_c = arma::find(new_assign == k);
    if(n_c.size() == 0){
      new_alpha.row((k-1)).fill(0.0);
    }
  }

  result["new_assign"] = new_assign;
  result["new_alpha"] = new_alpha;

  return result;
}

// Step 3: Split-Merge: --------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List SFDM_SM(int K, arma::vec old_assign, arma::vec old_alpha,
                   arma::vec xi, arma::vec y, arma::vec mu0, arma::vec a_sigma,
                   arma::vec b_sigma, arma::vec lambda, double a_theta,
                   double b_theta, int launch_iter){

  /* Description: This function will perform a split-merge procedure in our model.
   * Output: The list of 4 objects: (1) vector of an updated cluster assignment
   *         (2) vector of an updated cluster space (alpha),
   *         (3) split-merge result (1 is split, 0 is merge.)
   *         (4) accept result (1 is accept the split-merge, 0 is not.)
   * Input: Maximum possible cluster (K), the previous assignment (old_assign), 
   *        cluster concentration (xi), data (y), 
   *        data hyperparameters (a_sigma, b_sigma, lambda, mu0),
   *        spike-and-slab hyperparameters (a_theta, b_theta), 
   *        number of iteration for the launch step (launch_iter)
   */

  Rcpp::List result;
  
  // (0) Create variables
  int n = old_assign.size(); // Number of the observations
  arma::uvec active_clus = arma::conv_to<arma::uvec>::from(arma::unique(old_assign));
  int K_active = active_clus.size();
  int split_ind = -1; // Indicate that we will perform split or merge
  arma::vec launch_assign(old_assign); // Assignment from the launch step
  arma::vec all_alpha(old_alpha); // Alpha from the launch step
  int accept_proposed = 0; // Indicator for accepting the proposed assignment.
  double log_A = 0.0; // log of acceptance probability
  
  // (1) Select the observation and determine whether performing split or merge.
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
    new_clus = (arma::min(arma::find(old_alpha == 0)) + 1); // new active cluster
    samp_clus.row(0).fill(new_clus);
    launch_assign.row(samp_obs[0]).fill(new_clus); // set ci_launch to be a new cluster
    all_alpha.row((new_clus - 1)).fill(R::rgamma(xi[(new_clus - 1)], 1.0));
  } else {
    split_ind = 0;
  }
  
  // (2) Create a set S := {same cluster as samp_obs, but not index_samp}
  arma::uvec S = arma::find(old_assign == old_assign[samp_obs[0]] or old_assign == old_assign[samp_obs[1]]);
  arma::uvec samp_obs_index = arma::find(S == samp_obs[0] or S == samp_obs[1]);
  S.shed_rows(samp_obs_index);
  
  // (3) Perform a launch step
  // Randomly assign observation in S to be ci_launch or cj_launch.
  arma::uvec init_assign = arma::conv_to<arma::uvec>::from(arma::randu(S.size(), arma::distr_param(0, 1)) > 0.5);
  launch_assign.rows(S) = samp_clus.rows(init_assign);
  // Perform a launch step
  for(int t = 0; t < launch_iter; ++t){
    for(int j = 0; j < S.size(); ++j){
      arma::mat obs_i_alloc = log_alloc_prob(S[j], samp_clus, launch_assign, xi, 
                                             y, a_sigma, b_sigma, lambda, mu0, true);
      launch_assign.row(S[j]).fill(samp_new(obs_i_alloc));
    }
  }
  
  // (4) Split-Merge
  arma::vec proposed_assign(launch_assign);
  if(split_ind == 1){
    // Split: Perform another launch iteration
    for(int j = 0; j < S.size(); ++j){
      arma::mat obs_i_alloc = log_alloc_prob(S[j], samp_clus, proposed_assign, 
                                             xi, y, a_sigma, b_sigma, lambda, 
                                             mu0, true);
      proposed_assign.row(S[j]).fill(samp_new(obs_i_alloc));
    }
    log_A += (std::log(a_theta) - std::log(b_theta)); // Spike-and-slab
    // Proposal Probability
    log_A += log_proposal(launch_assign, proposed_assign, S, samp_clus, y, xi,                       
                          mu0, a_sigma, b_sigma, lambda); 
    log_A -= log_proposal(proposed_assign, launch_assign, S, samp_clus, y, xi,                       
                          mu0, a_sigma, b_sigma, lambda); 
  } else {
    // Merge: All observations in S and {ci, cj} will be allocated to cj
    proposed_assign.rows(S).fill(samp_clus[1]);
    proposed_assign.rows(samp_obs).fill(samp_clus[1]);
    log_A += (std::log(b_theta) - std::log(a_theta)); // Spike-and-slab
    // Proposal Probability
    log_A += log_proposal(old_assign, launch_assign, S, samp_clus, y, xi, 
                          mu0, a_sigma, b_sigma, lambda); 
  }
  
  // (5) Evaluate the proposal by using MH
  log_A += log_marginal(proposed_assign, y, a_sigma, b_sigma, lambda, mu0);
  log_A -= log_marginal(old_assign, y, a_sigma, b_sigma, lambda, mu0);

  log_A += log_gamma_cluster(all_alpha, xi, proposed_assign);
  log_A -= log_gamma_cluster(all_alpha, xi, old_assign);
  
  log_A += log_prior(proposed_assign, xi);
  log_A -= log_prior(proposed_assign, xi);
  
  arma::vec new_assign(old_assign);
  arma::vec new_alpha(all_alpha);
  
  if(std::log(R::runif(0.0, 1.0)) < std::min(0.0, log_A)){
    // Accept the proposed vector
    accept_proposed = 1;
    new_assign = proposed_assign;
  }
  
  arma::vec new_unique = arma::unique(new_assign);
  
  for(int k = 1; k <= K; ++k){
    arma::uvec n_c = arma::find(new_assign == k);
    if(n_c.size() == 0){
      new_alpha.row((k-1)).fill(0.0);
    }
  }
  
  result["new_assign"] = new_assign;
  result["new_alpha"] = new_alpha;
  result["log_A"] = log_A;
  result["split_index"] = split_ind;
  result["accept_proposed"] = accept_proposed;
  
  return result;
}

// Step 4: Update alpha: -------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List SFDM_alpha(arma::vec clus_assign, arma::vec xi, arma::vec old_alpha,
                      double old_u){
  
  /* Description: This function will update the alpha vector, and the auxiliary 
   *              variable. The derivation of this function is based on Matt's 
   *              Dirichlet Data Augmentation Trick document.
   * Output: An updated auxiliary variable (new_u) and alpha vector (new_alpha).
   * Input: The cluster assignment (clus_assign), cluster concentration (xi), 
   *        previous alpha vector (old_alpha), previous auxiliary variable (old_u).
   */
  
  Rcpp::List result;
  arma::vec new_alpha(old_alpha); 
  
  // Update alpha
  arma::vec active_clus = arma::unique(clus_assign);
  for(int k = 0; k < active_clus.size(); ++k){
    int current_c = active_clus[k];
    arma::uvec nk = arma::find(clus_assign == current_c);
    double scale_gamma = 1/(1 + old_u); // change the rate to scale parameter
    new_alpha.row(current_c - 1).fill(R::rgamma(nk.size() + xi[current_c - 1],
                                      scale_gamma));
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
Rcpp::List SFDM_model(int iter, int K, arma::vec init_assign, arma::vec xi,
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
    // Allocation Step
    alloc_List = SFDM_allocate(K, init_assign, xi, y, a_sigma, b_sigma, lambda, 
                               mu0, init_alpha);
    arma::vec alloc_assign = alloc_List["new_assign"];
    arma::vec alloc_alpha = alloc_List["new_alpha"];

    // Split-Merge Step
    sm_List = SFDM_SM(K, alloc_assign, alloc_alpha, xi, y, mu0, a_sigma, b_sigma,
                      lambda, a_theta, b_theta, sm_iter);

    arma::vec sm_assign = sm_List["new_assign"];
    arma::vec sm_alpha = sm_List["new_alpha"];
    
    // Update alpha vector (and u)
    alpha_update_List = SFDM_alpha(sm_assign, xi, sm_alpha, init_u);
    arma::vec alpha_updated = alpha_update_List["new_alpha"];
    double u_updated = alpha_update_List["new_u"];

    // Store the result
    iter_assign.col(i) = sm_assign;
    iter_alpha.col(i) = alpha_updated;
    iter_log_A.row(i).fill(sm_List["log_A"]);
    split_or_merge.row(i).fill(sm_List["split_index"]);
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