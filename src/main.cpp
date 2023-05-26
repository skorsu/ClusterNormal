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
                   arma::vec a_sigma_cluster, arma::vec b_sigma_cluster){
  
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
        log_pred = log_marginal(y_current, k, mu0_cluster, lambda_cluster, 
                                a_sigma_cluster, b_sigma_cluster);
      } else {
        log_pred = log_posterior(y_current, y_not_i.rows(obs_index), k, 
                                 mu0_cluster, lambda_cluster, a_sigma_cluster, 
                                 b_sigma_cluster);
      }
      alloc_prob.col(1).row(k).fill(log_pred);
    }
    
    // Calculate the predictive probability and normalize it.
    alloc_prob.col(2) = log_sum_exp(alloc_prob.col(1));
    
    // Sample a new cluster based on the predictive probability
    new_assign.row(i).fill(rmultinom_1(alloc_prob.col(2), K)); 
  }
  
  return new_assign;
}

// [[Rcpp::export]]
arma::mat fmm(int iter, int K, arma::vec old_assign, arma::vec y, 
              arma::vec mu0_cluster, arma::vec lambda_cluster, 
              arma::vec a_sigma_cluster, arma::vec b_sigma_cluster){
  
  /* Description: -
   * Output: -
   * Input: -
   */
  
  arma::mat cluster_iter(y.size(), iter, arma::fill::value(-1));
  arma::vec inter_assign(old_assign);
  
  for(int j = 0; j < iter; ++j){
    
    inter_assign = fmm_iter(K, inter_assign, y, mu0_cluster, 
                            lambda_cluster, a_sigma_cluster, b_sigma_cluster);
    cluster_iter.col(j) = inter_assign;
    
  }
  
  return cluster_iter.t();
  
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





// Step 1: Allocate the observation to the existing clusters: ------------------
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

// Step 2: Split-Merge: --------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List SFDM_SM(int K, arma::vec old_assign, arma::vec old_alpha,
                   arma::vec xi, arma::vec y, arma::vec mu0, arma::vec a_sigma,
                   arma::vec b_sigma, arma::vec lambda, double a_theta,
                   double b_theta, int launch_iter){

  Rcpp::List result; 
  
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

  // (0) Prepare for the algorithm
  int n = old_assign.size(); // Number of the observations
  arma::vec active_clus = arma::unique(old_assign);
  int K_active = active_clus.size();
  int split_ind = -1; // Indicate that we will perform split or merge
  arma::vec launch_assign(old_assign); // Assignment from the launch step
  arma::vec launch_alpha(old_alpha); // Alpha from the launch step
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
    
    arma::uvec inactive_index = arma::find(old_alpha == 0);
    arma::uvec new_clus_index = arma::randperm(inactive_index.size(), 1);
    new_clus = inactive_index[new_clus_index[0]] + 1; // new active cluster
    samp_clus.row(0).fill(new_clus);
    launch_assign.row(samp_obs[0]).fill(new_clus); // set ci_launch to be a new cluster
    launch_alpha.row((new_clus - 1)).fill(R::rgamma(xi[(new_clus - 1)], 1.0));
  } else {
    split_ind = 0;
  }
  
  // (2) Create a set S := {same cluster as samp_obs, but not index_samp}
  arma::uvec S = arma::find(old_assign == old_assign[samp_obs[0]] or old_assign == old_assign[samp_obs[1]]);
  arma::uvec samp_obs_index = arma::find(S == samp_obs[0] or S == samp_obs[1]);
  S.shed_rows(samp_obs_index);
  
  std::cout << "S: " << S << std::endl;
  std::cout << " ------------------ " << std::endl;
  std::cout << "samp_clus: " << samp_clus << std::endl;
  std::cout << " ------------------ " << std::endl;
  std::cout << "launch_alpha: " << launch_alpha << std::endl;
  
  return result;
  
}

// Step 3: Update alpha: -------------------------------------------------------
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
// Rcpp::List SFDM_model(int iter, int K, arma::vec init_assign, arma::vec xi,
//                       arma::vec y, arma::vec mu0, arma::vec a_sigma,
//                       arma::vec b_sigma, arma::vec lambda, double a_theta,
//                       double b_theta, int sm_iter, int print_iter){
// 
//   /* Description: This function runs our model.
//    * Output: A list of the result consisted of (1) the cluster assignment for
//    *         each iterations (iter x size of data matrix) (2) split/merge
//    *         decision for each iteration (split_or_merge) (3) accept or reject
//    *         SM status (sm_status)0
//    * Input: Number of iteration for runiing the model (iter) Maximum possible
//    *        cluster (K), the initial assignment (init_assign),
//    *        cluster concentration (xi), data (y),
//    *        data hyperparameters (mu0, a_sigma, b_sigma, lambda),
//    *        spike-and-slab hyperparameters (a_theta, b_theta),
//    *        number of iteration for the launch step (sm_iter),
//    *        progress report (print_iter)
//    */
// 
//   Rcpp::List result;
// 
//   // Initial alpha
//   arma::uvec init_unique = arma::conv_to<arma::uvec>::from(arma::unique(init_assign));
//   arma::vec init_alpha(K, arma::fill::zeros);
// 
//   for(int k = 0; k < init_unique.size(); ++k){
//     init_alpha.row(init_unique[k] - 1).fill(R::rgamma(xi[init_unique[k] - 1], 1.0));
//   }
//   
//   // Initial u (auxiliary variable)
//   double init_u = R::rgamma(y.size(), 1/(arma::accu(init_alpha)));
// 
//   // Objects for storing the intermediate result
//   Rcpp::List alloc_List;
//   Rcpp::List sm_List;
//   Rcpp::List alpha_update_List;
//   
//   // Create vectors/matrices for storing the final result
//   arma::mat iter_assign(y.size(), iter, arma::fill::value(-1));
//   arma::mat iter_alpha(K, iter, arma::fill::value(-1));
//   arma::vec sm_status(iter, arma::fill::value(-1));
//   arma::vec iter_log_A(iter, arma::fill::value(-1));
//   arma::vec split_or_merge(iter, arma::fill::value(-1));
// 
//   // Perform an algorithm
//   for(int i = 0; i < iter; ++i){
//     // Allocation Step
//     alloc_List = SFDM_allocate(K, init_assign, xi, y, a_sigma, b_sigma, lambda, 
//                                mu0, init_alpha);
//     arma::vec alloc_assign = alloc_List["new_assign"];
//     arma::vec alloc_alpha = alloc_List["new_alpha"];
// 
//     // Split-Merge Step
//     sm_List = SFDM_SM(K, alloc_assign, alloc_alpha, xi, y, mu0, a_sigma, b_sigma,
//                       lambda, a_theta, b_theta, sm_iter);
// 
//     arma::vec sm_assign = sm_List["new_assign"];
//     arma::vec sm_alpha = sm_List["new_alpha"];
//     
//     // Update alpha vector (and u)
//     alpha_update_List = SFDM_alpha(sm_assign, xi, sm_alpha, init_u);
//     arma::vec alpha_updated = alpha_update_List["new_alpha"];
//     double u_updated = alpha_update_List["new_u"];
// 
//     // Store the result
//     iter_assign.col(i) = sm_assign;
//     iter_alpha.col(i) = alpha_updated;
//     iter_log_A.row(i).fill(sm_List["log_A"]);
//     split_or_merge.row(i).fill(sm_List["split_index"]);
//     sm_status.row(i).fill(sm_List["accept_proposed"]);
//     
//     // Initialize for the next iteration
//     init_assign = sm_assign;
//     init_alpha = alpha_updated;
//     init_u = u_updated;
// 
//     // Print the result
//     if(((i + 1) - (floor((i + 1)/print_iter) * print_iter)) == 0){
//       std::cout << "Iter: " << (i+1) << " - Done!" << std::endl;
//     }
// 
//   }
//   
//   result["iter_assign"] = iter_assign.t();
//   result["iter_alpha"] = iter_alpha.t();
//   result["log_A"] = iter_log_A;
//   result["sm_status"] = sm_status;
//   result["split_or_merge"] = split_or_merge;
// 
//   return result;
// }

// END: ------------------------------------------------------------------------