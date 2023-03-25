#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

#define pi 3.141592653589793238462643383280

// Note to self: ---------------------------------------------------------------
// * Debugging the whole process
// * Step 1: Omitted 
// * Step 2 : Reallocation 
// ** Updated on 3/22/2023: FMM works! (Better than the previous version)

// * Step 3: Split-Merge

// * Step 4: Update alpha vector

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

// [[Rcpp::export]]
arma::mat rdirichlet_cpp(int num_samples, arma::vec alpha_m){
  int distribution_size = alpha_m.n_elem;
  // each row will be a draw from a Dirichlet
  arma::mat distribution = arma::zeros(num_samples, distribution_size);
  
  /* Description: Sample from dirichlet distribution.
   * Credit: https://www.mjdenny.com/blog.html
   */
  
  for (int i = 0; i < num_samples; ++i){
    double sum_term = 0;
    // loop through the distribution and draw Gamma variables
    for (int j = 0; j < distribution_size; ++j){
      double cur = R::rgamma(alpha_m[j],1.0);
      distribution(i,j) = cur;
      sum_term += cur;
    }
    // now normalize
    for (int j = 0; j < distribution_size; ++j) {
      distribution(i,j) = distribution(i,j)/sum_term;
    }
  }
  return(distribution);
}

// Updated Code: ---------------------------------------------------------------
// [[Rcpp::export]]
arma::vec log_alloc_prob(int K, int i, arma::vec old_assign, arma::vec xi, 
                         arma::vec y, arma::vec a_sigma, arma::vec b_sigma, 
                         arma::vec lambda, arma::vec mu0){
  
  /* Description: This function will calculate the log allocation probability
   *              of the particular observation for all possible clusters.
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
int samp_new(int K, arma::vec log_alloc){
  
  /* Description: This function will perform two things. The first is to 
   *              transform the log allocation probability back to the 
   *              probability by applying log-sum-exp trick. Secondly, it will 
   *              sample the new cluster based on the probability from the first
   *              step.
   * Output: new cluster
   * Input: Maximum possible cluster (K), the allocation probability in the log 
   *        scale (log_alloc).
   */
  
  // Convert the log probability back to probability using log-sum-exp trick
  arma::vec prob = log_sum_exp(log_alloc);
  
  // Sample from the list of the active cluster using the aloocation probability 
  // arma::vec active_prob = alloc_list["active_clus"];
  // int K = prob.size();
  
  Rcpp::IntegerVector x_index = Rcpp::seq(1, K);
  Rcpp::NumericVector norm_prob = Rcpp::wrap(prob);
  Rcpp::IntegerVector x = Rcpp::sample(x_index, 1, false, norm_prob);
  return x[0];
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
// * Univariate (New)
// [[Rcpp::export]]
arma::vec clus_alloc(int K, arma::vec old_assign, arma::vec xi, arma::vec y, 
                     arma::vec alpha, arma::vec mu0, arma::vec a_sigma, 
                     arma::vec b_sigma, arma::vec lambda){
  
  arma::vec new_assign(old_assign);

  // Reassign the observation
  for(int i = 0; i < new_assign.size(); ++i){
    arma::vec obs_i_alloc = log_alloc_prob(K, i, old_assign, xi, y, a_sigma, 
                                           b_sigma, lambda, mu0);
    new_assign.row(i).fill(samp_new(K, obs_i_alloc));
  }
  
  return new_assign;
}

// Step 3: Split-Merge: --------------------------------------------------------
// * Univariate 
// Rcpp::List uni_split_merge(int K, arma::vec old_assign, arma::vec alpha,
//                            arma::vec xi, arma::vec y, arma::vec mu_0, 
//                            arma::vec a_sigma, arma::vec b_sigma, 
//                            arma::vec lambda, double a_theta, double b_theta, 
//                            int sm_iter){
//   Rcpp::List result;
//   int accept_iter = 1;
//   int split_i = 0;
//   int split_k = 0;
//   int merge_i = 0;
//   
//   // Initial the alpha vector and assignment vector
//   arma::vec launch_assign = old_assign;
//   arma::vec launch_alpha = alpha;
//   
//   // Create the set of active and inactive cluster
//   Rcpp::List List_clusters = active_inactive(K, old_assign);
//   arma::uvec active_clus = List_clusters["active"];
//   arma::uvec inactive_clus = List_clusters["inactive"];
//   
//   // Sample two observations from the data.
//   Rcpp::IntegerVector obs_index = Rcpp::seq(0, old_assign.size() - 1);
//   Rcpp::IntegerVector samp_obs = Rcpp::sample(obs_index, 2);
//   
//   int obs_i = samp_obs[0];
//   int obs_j = samp_obs[1];
//   int c_i = old_assign.at(obs_i); // ci_launch
//   int c_j = old_assign.at(obs_j); // cj_launch
//   
//   if(active_clus.size() == K){
//     while(c_i == c_j){
//       samp_obs = Rcpp::sample(obs_index, 2);
//       obs_i = samp_obs[0];
//       obs_j = samp_obs[1];
//       c_i = old_assign.at(obs_i);
//       c_j = old_assign.at(obs_j);
//     }
//   }
//   
//   // Select only the observations that in the same cluster as obs_i and obs_j
//   arma::uvec s_index = find((old_assign == c_i) or (old_assign == c_j));
//   
//   if(c_i == c_j){
//     arma::vec prob_inactive = arma::ones(inactive_clus.size())/
//       inactive_clus.size();
//     c_i = sample_clus(prob_inactive, inactive_clus);
//     launch_assign.row(obs_i).fill(c_i);
//     launch_alpha.row(c_i - 1).fill(R::rgamma(xi.at(c_i - 1), 1.0));
//   }
//   
//   // Randomly assign the observation in s_index to either c_i or c_j
//   arma::uvec cluster_launch(2);
//   cluster_launch.row(0).fill(c_i);
//   cluster_launch.row(1).fill(c_j);
//   
//   for(int i = 0; i < s_index.size(); ++i){
//     int current_obs = s_index.at(i);
//     arma::vec random_prob = 0.5 * arma::ones(2);
//     if((current_obs != obs_i) and (current_obs != obs_j)){
//       launch_assign.row(current_obs).
//       fill(sample_clus(random_prob, cluster_launch));
//     }
//   }
//   
//   // Perform a Launch Step
//   for(int t = 0; t < sm_iter; ++t){
//     for(int i = 0; i < s_index.size(); ++i){
//       int current_obs = s_index.at(i);
//       int launch_c = uni_alloc(current_obs, launch_assign, xi, y, a_sigma, 
//                                b_sigma, lambda, mu_0, cluster_launch);
//       launch_assign.row(current_obs).fill(launch_c);
//     }
//   }
//   
//   // Prepare for the split-merge process
//   double sm_indicator = 0.0;
//   arma::vec new_assign = launch_assign;
//   arma::vec launch_alpha_vec = launch_alpha; 
//   // We will use launch_assign and launch_alpha_vec for MH algorithm.
//   List_clusters = active_inactive(K, launch_assign);
//   arma::uvec active_sm = List_clusters["active"];
//   arma::uvec inactive_sm = List_clusters["inactive"];
//   
//   c_i = launch_assign.at(obs_i);
//   c_j = launch_assign.at(obs_j);
//   arma::uvec cluster_sm(2);
//   cluster_sm.row(0).fill(-1);
//   cluster_sm.row(1).fill(c_j);
//   
//   // Split-Merge Process
//   if(c_i != c_j){ 
//     // merge these two clusters into c_j cluster
//     sm_indicator = 1.0;
//     new_assign.elem(s_index).fill(c_j);
//     merge_i += 1;
//   } else if((c_i == c_j) and (active_sm.size() != K)) { 
//     // split in case that at least one cluster is inactive.
//     sm_indicator = -1.0;
//     split_i += 1;
//     // sample a new inactive cluster
//     arma::vec prob_inactive = arma::ones(inactive_sm.size())/inactive_sm.size();
//     c_i = sample_clus(prob_inactive, inactive_sm);
//     new_assign.row(obs_i).fill(c_i);
//     launch_alpha.row(c_i - 1).fill(R::rgamma(xi.at(c_i - 1), 1.0));
//     cluster_sm.row(0).fill(c_i);
//     
//     for(int i = 0; i < s_index.size(); ++i){
//       int current_obs = s_index.at(i);
//       if((current_obs != obs_i) and (current_obs != obs_j)){
//         int sm_clus = uni_alloc(current_obs, new_assign, xi, y, a_sigma, 
//                                 b_sigma, lambda, mu_0, cluster_sm);
//         new_assign.row(current_obs).fill(sm_clus);
//       }
//     }
//   } else {
//     // Rcpp::Rcout << "final: split (none inactive)" << std::endl;
//     split_k += 1;
//   }
//   
//   arma::vec new_alpha = adjust_alpha(K, new_assign, launch_alpha);
//   
//   // MH Update (log form)
//   // Elements
//   double launch_elem = 0.0;
//   double final_elem = 0.0;
//   double alpha_log = 0.0;
//   double proposal = sm_indicator * std::log(0.5) * s_index.size();
//   
//   for(int k = 1; k <= K; ++k){
//     // Calculate alpha
//     if(launch_alpha_vec.at(k - 1) != new_alpha.at(k - 1)){
//       if(new_alpha.at(k - 1) != 0){
//         alpha_log += R::dgamma(new_alpha.at(k - 1), xi.at(k - 1), 1.0, 1);
//         alpha_log += std::log(a_theta);
//         alpha_log -= std::log(b_theta);
//       } else {
//         alpha_log -= R::dgamma(launch_alpha_vec.at(k - 1), 
//                                xi.at(k - 1), 1.0, 1);
//         alpha_log -= std::log(a_theta);
//         alpha_log += std::log(b_theta);
//       }
//     }
//     // Calculate Multinomial
//     arma::uvec launch_elem_vec = arma::find(launch_assign == k);
//     arma::uvec final_elem_vec = arma::find(new_assign == k);
//     if(launch_elem_vec.size() > 0){
//       launch_elem += launch_elem_vec.size() * 
//         std::log(launch_alpha_vec.at(k - 1));
//     }
//     if(final_elem_vec.size() > 0){
//       final_elem += final_elem_vec.size() * std::log(new_alpha.at(k - 1));
//     }
//   }
//   
//   double log_A = std::min(std::log(1), 
//                           alpha_log + final_elem - launch_elem + proposal);
//   double log_u = std::log(R::runif(0.0, 1.0));
//   
//   if(log_u >= log_A){
//     new_assign = launch_assign;
//     new_alpha = launch_alpha_vec;
//     accept_iter -= 1;
//   }
//   
//   result["accept_prob"] = accept_iter;
//   result["split"] = split_i;
//   result["split_k"] = split_k;
//   result["merge"] = merge_i;
//   result["new_assign"] = new_assign;
//   result["new_alpha"] = new_alpha;
//   
//   return result;
// }

// Step 4: Update alpha: -------------------------------------------------------
// * Both Univariate and Multivariate
// arma::vec update_alpha(int K, arma::vec alpha, arma::vec xi, 
//                        arma::vec old_assign){
//   
//   arma::vec new_alpha = alpha;
//   
//   /* Input: maximum cluster (K),previous cluster weight (alpha), 
//    *        hyperparameter for cluster (xi), 
//    *        previous cluster assignment (old_assign).
//    * Output: new cluster weight.
//    */
//   
//   Rcpp::List List_active = active_inactive(K, old_assign);
//   arma::uvec active_clus = List_active["active"];
//   
//   arma::vec n_xi_elem = -1.0 * arma::ones(active_clus.size());
//   
//   for(int k = 0; k < active_clus.size(); ++k){
//     int clus_current = active_clus.at(k);
//     arma::uvec obs_current_index = old_assign == clus_current;
//     n_xi_elem.at(k) = sum(obs_current_index) + xi.at(clus_current - 1);
//   }
//   
//   arma::mat psi_new = rdirichlet_cpp(1, n_xi_elem);
//   
//   for(int k = 0; k < active_clus.size(); ++k){
//     int clus_current = active_clus.at(k);
//     new_alpha.at(clus_current - 1) = sum(alpha) * psi_new(0, k);
//   }
//   
//   return new_alpha;
// }

// Final Function: -------------------------------------------------------------


// END: ------------------------------------------------------------------------