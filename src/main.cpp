#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

#define pi 3.141592653589793238462643383280

// Note to self: ---------------------------------------------------------------
// * Start with univariate case.
// * Then, multivariate case.

// User-defined function: ------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List active_inactive(int K, arma::vec clus_assign){
  Rcpp::List result;
  
  /* Description: This function will return the list consisted of two vectors
   *              (1) active clusters and (2) inactive cluster from the cluster
   *              assignment vector.
   * Input: maximum cluster (K), cluster assignment vector (clus_assign)
   * Output: A list of two vectors (active & inactive clusters.) 
   */
  
  Rcpp::IntegerVector all_possible = Rcpp::seq(1, K);
  Rcpp::IntegerVector active_clus = 
    Rcpp::as<Rcpp::IntegerVector>(Rcpp::wrap(arma::unique(clus_assign)));
  Rcpp::IntegerVector inactive_clus = Rcpp::setdiff(all_possible, active_clus);
  
  result["active"] = active_clus;
  result["inactive"] = inactive_clus;
  
  return result;
}

// [[Rcpp::export]]
int sample_clus(arma::vec norm_probs, arma::uvec active_clus){
  
  /* Description: To run a multinomial distribution and get the reallocated 
   *              cluster.
   * Input: Normalized probability (norm_prob), active cluster
   * Output: New assigned cluster
   */
  
  int k = active_clus.size();
  arma::imat C = arma::imat(k, 1);
  rmultinom(1, norm_probs.begin(), k, C.colptr(0));
  
  arma::mat mat_index = arma::conv_to<arma::mat>::from(arma::index_max(C));
  int new_clus = active_clus.at(mat_index(0, 0));
  
  return new_clus;
}

// [[Rcpp::export]]
double log_multi_lgamma(double a, double d){
  double result = 0.0;
  
  result += (0.25 * d * (d-1)) * log(pi);
  arma::vec d_seq = Rcpp::as<arma::vec>(Rcpp::wrap(Rcpp::seq(1, d)));
  result += sum(arma::lgamma(a + ((1 - d_seq)/2)));
  
  return result;
}

// [[Rcpp::export]]
double log_marginal_univariate(double y, double a_sigma_K, double b_sigma_K,
                               double lambda_K, double mu_0_K){
  
  /* Description: This is the function for calculating the log of the 
   *              marginal likelihood of the data (Univariate Case).
   * Input: Data point (y) and hyperparameter for that cluster 
   *        (a_sigma, b_sigma, lambda, mu_0)
   * Output: log(p(yi|a_sigma, b_sigma, lambda, mu_0))
   */
  
  double log_marginal = 0.0;
  double denom_base = b_sigma_K + 
    ((lambda_K/(2 * (lambda_K + 1))) * std::pow(y - mu_0_K, 2.0));
  
  log_marginal += (0.5 * log(lambda_K)) - (0.5 * log(lambda_K + 1));
  log_marginal += lgamma(a_sigma_K + 0.5) - lgamma(a_sigma_K);
  log_marginal += (a_sigma_K * log(b_sigma_K));
  log_marginal -= (a_sigma_K + 0.5) * log(denom_base);

  return log_marginal;
}

// [[Rcpp::export]]
double log_marginal_multi(arma::vec y, arma::vec mu_0, double lambda_0, 
                          double nu_0, arma::mat L_0){
  
  /* Description: This is the function for calculating the log of the 
   *              marginal likelihood of the data (Univariate Case).
   * Input: Data point (y) and hyperparameter for that cluster 
   *        (a_sigma, b_sigma, lambda, mu_0)
   * Output: log(p(yi|a_sigma, b_sigma, lambda, mu_0))
   */
  

  double log_marginal = 0.0;
  
  // Posterior parameters for calculating marginal distribution
  double d = y.size();
  double nu_n = nu_0 + 1;
  double lambda_n = lambda_0 + 1;
  
  arma::vec diff_y_mu = y - mu_0;

  arma::mat L_n = L_0 + 
    ((lambda_0/(lambda_0 + 1)) * (diff_y_mu * arma::trans(diff_y_mu)));

  // Calculate the log marginal density 
  log_marginal -= (d/2)*log(pi);
  log_marginal += log_multi_lgamma(nu_n/2, d) - log_multi_lgamma(nu_0/2, d); 
  
  double val_L0;
  double sign_L0;
  bool log_det_L0 = arma::log_det(val_L0, sign_L0, L_0);
  
  log_marginal += ((nu_0/2) * val_L0 * sign_L0);
  
  double val_Ln;
  double sign_Ln;
  bool log_det_Ln = arma::log_det(val_Ln, sign_Ln, L_n);
  
  log_marginal -= ((nu_n/2) * val_Ln * sign_Ln);
  log_marginal += ((d/2) * log(lambda_0));
  log_marginal -= ((d/2) * log(lambda_n));
  
  return log_marginal;
}

// [[Rcpp::export]]
arma::vec log_allocate_prob_univariate(int i, arma::vec current_assign, 
                                       arma::vec xi, arma::vec y, 
                                       arma::vec a_sigma, arma::vec b_sigma,
                                       arma::vec lambda, arma::vec mu_0, 
                                       arma::uvec active_clus){
  arma::vec log_unnorm = -1 * arma::ones(active_clus.size());
  
  // Split the data into two sets: (1) observation i (2) without observation i.
  double y_i = y.at(i);
  arma::vec y_not_i = y; 
  y_not_i.shed_row(i);
  arma::vec clus_not_i = current_assign; 
  clus_not_i.shed_row(i);
  
  // Calculate the log of unnormalized allocation for each active cluster
  for(int k = 0; k < active_clus.size(); ++k){
    int current_c = active_clus[k];
    arma::uvec current_ci = arma::find(clus_not_i == current_c);
    
    // Filter only the observation from cluster i
    arma::vec y_current = y_not_i.elem(current_ci);
    
    // Select the hyperparameter that corresponding to the cluster k
    double a_k = a_sigma.at(current_c - 1);
    double b_k = b_sigma.at(current_c - 1);
    double lambda_k = lambda.at(current_c - 1); 
    double mu_0k = mu_0.at(current_c - 1);
    double xi_k = xi.at(current_c - 1);
    double n_k = current_ci.size();
    
    // Calculate the parameter for the posterior predictive distribution
    double mean_k = 0.0;
    double var_k = 0.0;
    
    if(n_k > 0){
      mean_k = arma::mean(arma::mean(y_current));
      var_k = arma::var(y_current);
    }

    double nu = (2*a_k) + n_k;
    
    double mu_star = ((n_k * mean_k) + (lambda_k + mu_0k))/(lambda_k + n_k);
    double sigma2_star = (2 * (lambda_k + n_k + 1))/((lambda_k + n_k) * nu);
    double b_star = b_k + (0.5 * var_k * (n_k - 1)) + 
      (0.5 * ((n_k * lambda_k)/(n_k + lambda_k)) * std::pow(mean_k - mu_0k, 2.0));
    
    // t-distribution component
    double base_term = 1 + 
      ((1/nu) * (1/(sigma2_star * b_star)) * std::pow(y_i - mu_star, 2.0));
    double t = lgamma((nu + 1)/2) - lgamma(nu/2) - (0.5 * log(nu)) - 
      (0.5 * log(sigma2_star * b_star)) - (((nu + 1)/2) * log(base_term));
  
    log_unnorm.row(k).fill(t + log(n_k + xi_k));
    
  }
  
  return log_unnorm;
}

// [[Rcpp::export]]
arma::vec log_allocate_prob(int i, arma::vec current_assign, arma::vec xi, 
                            arma::mat y, arma::mat gamma_hyper_mat, 
                            arma::uvec active_clus){
  
  arma::vec log_unnorm_prob = -1 * arma::ones(active_clus.size());
  
  /* Description: Calculate the log unnormalized probability for each cluster 
   *              for observation i.
   * Input: current index (i), current cluster assignment, 
   *        hyperparameter for cluster (xi), data matrix (y), 
   *        hyperparameter for the data (gamma_hyper_mat), active cluster.
   * Output: log unnormalized allocation probability.
   */
  
  // Split the data into two sets: (1) observation i (2) without observation i.
  arma::rowvec y_i = y.row(i);
  arma::mat y_not_i = y; 
  y_not_i.shed_row(i);
  arma::vec clus_not_i = current_assign; 
  clus_not_i.shed_row(i);
  
  // Calculate the unnormalized allocation probability for each active cluster
  for(int k = 0; k < active_clus.size(); ++k){
    int current_c = active_clus[k];
    arma::uvec current_ci = arma::find(clus_not_i == current_c);
    
    // Select the hyperparameter that corresponding to the cluster k
    arma::rowvec gamma_hyper = arma::conv_to<arma::rowvec>::
      from(gamma_hyper_mat.row(current_c - 1));
    double xi_k = xi.at(current_c - 1);
    
    // Filter only the observation from cluster i
    arma::mat y_current = y_not_i.rows(current_ci);
    
    // t-distribution component
    
    // Calculate required vectors
    arma::rowvec y_hyper = gamma_hyper + arma::sum(y_current, 0);
    arma::rowvec y_hyper_yi = y_hyper + y_i;
    double n_xi = xi_k + current_ci.size();
    
    // Calculate log(gamma) component.
    arma::mat lg_y_hyper = arma::lgamma(y_hyper);
    arma::mat lg_y_hyper_yi = arma::lgamma(y_hyper_yi);
    
    double calculate_lg = lgamma(sum(y_hyper)) + arma::accu(lg_y_hyper_yi) +
      std::log(n_xi) - arma::accu(lg_y_hyper) - lgamma(sum(y_hyper_yi));
    
    log_unnorm_prob.row(k).fill(calculate_lg);
  }
  
  return log_unnorm_prob;
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

// Step 1: Update the cluster space: -------------------------------------------
// * Univariate
// [[Rcpp::export]]
Rcpp::List expand_step_univariate(int K, arma::vec old_assign, arma::vec alpha,
                                  arma::vec xi, arma::vec y, arma::vec mu_0,
                                  arma::vec a_sigma, arma::vec b_sigma, 
                                  arma::vec lambda, double a_theta, 
                                  double b_theta){
  Rcpp::List result;
  
  // Indicate the existed clusters and inactive clusters
  Rcpp::List List_clusters = active_inactive(K, old_assign);
  arma::uvec inactive_clus = List_clusters["inactive"];
  arma::uvec active_clus = List_clusters["active"];
  
  if(active_clus.size() < K){
    // Select a candidate cluster
    arma::vec samp_prob = arma::ones(inactive_clus.size())/inactive_clus.size();
    int candidate_clus = sample_clus(samp_prob, inactive_clus);
    
    // Sample alpha for new active cluster
    alpha.row(candidate_clus - 1).fill(R::rgamma(xi.at(candidate_clus - 1), 1));
    
    // Calculate the log of acceptance probability and assign a new cluster
    arma::vec log_accept_prob = std::log(alpha.at(candidate_clus - 1)) + 
      std::log((sum(alpha) - alpha.at(candidate_clus - 1))) + 
      std::log(a_theta) - arma::log(alpha) - std::log(sum(alpha)) -
      std::log(b_theta);
    
    arma::vec new_assign = old_assign;
    for(int i = 0; i < old_assign.size(); ++i){
      double log_prob = log_accept_prob.at(old_assign.at(i) - 1) +
        log_marginal_univariate(y.at(i), a_sigma.at(candidate_clus - 1), 
                                b_sigma.at(candidate_clus - 1),
                                lambda.at(candidate_clus - 1), 
                                mu_0.at(candidate_clus - 1)) -
        log_marginal_univariate(y.at(i), a_sigma.at(old_assign.at(i) - 1), 
                                b_sigma.at(old_assign.at(i) - 1),
                                lambda.at(old_assign.at(i) - 1), 
                                mu_0.at(old_assign.at(i) - 1));
      double log_A = std::min(log_prob, 0.0);
      double log_U = std::log(arma::randu());
      if(log_U <= log_A){
        new_assign.row(i).fill(candidate_clus);
      }
    }
    
    // Adjust an alpha vector
    arma::vec new_alpha = adjust_alpha(K, new_assign, alpha);
    
    result["new_alpha"] = new_alpha;
    result["new_assign"] = new_assign;
  } else {
    result["new_alpha"] = alpha;
    result["new_assign"] = old_assign;
  }
  
  return result;
}

// * Multivariate
// [[Rcpp::export]]
Rcpp::List expand_step_multi(int K, arma::vec old_assign, arma::vec alpha,
                             arma::vec xi, arma::mat y, arma::mat mu_0,
                             arma::vec lambda_0, arma::vec nu_0, arma::cube L_0,
                             double a_theta, double b_theta){
  
  Rcpp::List result;
  
  // Indicate the existed clusters and inactive clusters
  Rcpp::List List_clusters = active_inactive(K, old_assign);
  arma::uvec inactive_clus = List_clusters["inactive"];
  arma::uvec active_clus = List_clusters["active"];
  
  if(active_clus.size() < K){
    // Select a candidate cluster
    arma::vec samp_prob = arma::ones(inactive_clus.size())/inactive_clus.size();
    int candidate_clus = sample_clus(samp_prob, inactive_clus);
    
    // Sample alpha for new active cluster
    alpha.row(candidate_clus - 1).fill(R::rgamma(xi.at(candidate_clus - 1), 1));
    
    // Calculate the log of acceptance probability and assign a new cluster
    arma::vec log_accept_prob = std::log(alpha.at(candidate_clus - 1)) + 
      std::log((sum(alpha) - alpha.at(candidate_clus - 1))) + 
      std::log(a_theta) - arma::log(alpha) - std::log(sum(alpha)) -
      std::log(b_theta);
    
    arma::vec new_assign = old_assign;
    for(int i = 0; i < old_assign.size(); ++i){
      Rcpp::Rcout << "1" << std::endl;
      double log_prob = log_accept_prob.at(old_assign.at(i) - 1);
      
      // Prepare for log_marginal_multi function
      arma::vec yi = arma::conv_to<arma::vec>::from(y.row(i));
      arma::vec mu_0_new = arma::conv_to<arma::vec>::from(mu_0.row(candidate_clus - 1));
      arma::vec mu_0_old = arma::conv_to<arma::vec>::from(mu_0.row(old_assign.at(i) - 1));
      Rcpp::Rcout << L_0.slice(2) << std::endl;
      
      log_prob += log_marginal_multi(yi, mu_0_new, lambda_0.at(candidate_clus - 1), 
                                     nu_0.at(candidate_clus - 1), L_0.slice(candidate_clus - 1));
      
      log_prob -= log_marginal_multi(yi, mu_0_old, lambda_0.at(old_assign.at(i) - 1), 
                                     nu_0.at(old_assign.at(i) - 1), 
                                     L_0.slice(old_assign.at(i) - 1));

      double log_A = std::min(log_prob, 0.0);
      double log_U = std::log(arma::randu());
      if(log_U <= log_A){
        new_assign.row(i).fill(candidate_clus);
      }
    }
    
    // Adjust an alpha vector
    arma::vec new_alpha = adjust_alpha(K, new_assign, alpha);
    
    result["new_alpha"] = new_alpha;
    result["new_assign"] = new_assign;
  } else {
    result["new_alpha"] = alpha;
    result["new_assign"] = old_assign;
  }
  
  return result;
}

// Step 2: Allocate the observation to the existing clusters: ------------------
// * Univariate
// [[Rcpp::export]]
Rcpp::List cluster_assign_univariate(int K, arma::vec old_assign, arma::vec xi, 
                                     arma::vec y, arma::vec alpha, 
                                     arma::vec mu_0, arma::vec a_sigma, 
                                     arma::vec b_sigma, arma::vec lambda){
  Rcpp::List result;
  arma::vec new_assign = old_assign;
  
  // Assign a new assignment
  for(int i = 0; i < new_assign.size(); ++i){
    // Create the vector of the active cluster
    Rcpp::List active_List = active_inactive(K, new_assign);
    arma::uvec active_clus = active_List["active"];
    
    // Calculate the unnormalized probability
    arma::vec log_unnorm_prob = log_allocate_prob_univariate(i, new_assign, xi, 
                                                             y, a_sigma, 
                                                             b_sigma, lambda, 
                                                             mu_0, active_clus);
    // Calculate the normalized probability
    arma::vec norm_prob = log_sum_exp(log_unnorm_prob);
    
    // Reassign a new cluster
    int new_clus = sample_clus(norm_prob, active_clus);
    new_assign.row(i).fill(new_clus);
  }
  
  // Adjust an alpha vector
  Rcpp::List active_List = active_inactive(K, new_assign);
  arma::uvec active_clus = active_List["active"];
  arma::vec new_alpha = adjust_alpha(K, new_assign, alpha);
  
  result["new_assign"] = new_assign;
  result["new_alpha"] = new_alpha;
  
  return result;
}

// Step 3: Split-Merge: --------------------------------------------------------
// * Univariate 
// [[Rcpp::export]]
Rcpp::List split_merge_univariate(int K, arma::vec old_assign, arma::vec alpha,
                                  arma::vec xi, arma::vec y, arma::vec mu_0, 
                                  arma::vec a_sigma, arma::vec b_sigma, 
                                  arma::vec lambda, double a_theta, 
                                  double b_theta, int sm_iter){
  Rcpp::List result;
  
  // Initial the alpha vector and assignment vector
  arma::vec launch_assign = old_assign;
  arma::vec launch_alpha = alpha;
  
  // Create the set of active and inactive cluster
  Rcpp::List List_clusters = active_inactive(K, old_assign);
  arma::uvec active_clus = List_clusters["active"];
  arma::uvec inactive_clus = List_clusters["inactive"];
  
  // Sample two observations from the data.
  Rcpp::IntegerVector obs_index = Rcpp::seq(0, old_assign.size() - 1);
  Rcpp::IntegerVector samp_obs = Rcpp::sample(obs_index, 2);
  
  int obs_i = samp_obs[0];
  int obs_j = samp_obs[1];
  int c_i = old_assign.at(obs_i); // ci_launch
  int c_j = old_assign.at(obs_j); // cj_launch
  
  if(active_clus.size() == K){
    while(c_i == c_j){
      samp_obs = Rcpp::sample(obs_index, 2);
      obs_i = samp_obs[0];
      obs_j = samp_obs[1];
      c_i = old_assign.at(obs_i);
      c_j = old_assign.at(obs_j);
    }
  }
  
  // Select only the observations that in the same cluster as obs_i and obs_j
  arma::uvec s_index = find((old_assign == c_i) or (old_assign == c_j));
  
  if(c_i == c_j){
    arma::vec prob_inactive = arma::ones(inactive_clus.size())/
      inactive_clus.size();
    c_i = sample_clus(prob_inactive, inactive_clus);
    launch_assign.row(obs_i).fill(c_i);
    launch_alpha.row(c_i - 1).fill(R::rgamma(xi.at(c_i - 1), 1.0));
  }
  
  // Randomly assign the observation in s_index to either c_i or c_j
  arma::uvec cluster_launch(2);
  cluster_launch.row(0).fill(c_i);
  cluster_launch.row(1).fill(c_j);
  
  for(int i = 0; i < s_index.size(); ++i){
    int current_obs = s_index.at(i);
    arma::vec random_prob = 0.5 * arma::ones(2);
    if((current_obs != obs_i) and (current_obs != obs_j)){
      launch_assign.row(current_obs).
      fill(sample_clus(random_prob, cluster_launch));
    }
  }
  
  // Perform a Launch Step
  for(int t = 0; t < sm_iter; ++t){
    for(int i = 0; i < s_index.size(); ++i){
      int current_obs = s_index.at(i);
      arma::vec log_unnorm_prob = log_allocate_prob_univariate(current_obs, launch_assign, 
                                                               xi, y, a_sigma, 
                                                               b_sigma, lambda, 
                                                               mu_0, cluster_launch);
      arma::vec norm_prob = log_sum_exp(log_unnorm_prob);
      launch_assign.row(current_obs).
      fill(sample_clus(norm_prob, cluster_launch));
    }
  }
  
  // Prepare for the split-merge process
  double sm_indicator = 0.0;
  arma::vec new_assign = launch_assign;
  arma::vec launch_alpha_vec = launch_alpha; 
  // We will use launch_assign and launch_alpha_vec for MH algorithm.
  List_clusters = active_inactive(K, launch_assign);
  arma::uvec active_sm = List_clusters["active"];
  arma::uvec inactive_sm = List_clusters["inactive"];
  
  c_i = launch_assign.at(obs_i);
  c_j = launch_assign.at(obs_j);
  arma::uvec cluster_sm(2);
  cluster_sm.row(0).fill(-1);
  cluster_sm.row(1).fill(c_j);
  
  // Split-Merge Process
  if(c_i != c_j){ 
    // merge these two clusters into c_j cluster
    // Rcpp::Rcout << "final: merge" << std::endl;
    sm_indicator = 1.0;
    new_assign.elem(s_index).fill(c_j);
  } else if((c_i == c_j) and (active_sm.size() != K)) { 
    // split in case that at least one cluster is inactive.
    // Rcpp::Rcout << "final: split (some inactive)" << std::endl;
    sm_indicator = -1.0;
    
    // sample a new inactive cluster
    arma::vec prob_inactive = arma::ones(inactive_sm.size())/inactive_sm.size();
    c_i = sample_clus(prob_inactive, inactive_sm);
    new_assign.row(obs_i).fill(c_i);
    launch_alpha.row(c_i - 1).fill(R::rgamma(xi.at(c_i - 1), 1.0));
    cluster_sm.row(0).fill(c_i);
    
    for(int i = 0; i < s_index.size(); ++i){
      int current_obs = s_index.at(i);
      if((current_obs != obs_i) and (current_obs != obs_j)){
        arma::vec log_unnorm_prob = log_allocate_prob_univariate(current_obs, new_assign, 
                                                                 xi, y, a_sigma, 
                                                                 b_sigma, lambda, 
                                                                 mu_0, cluster_sm);
        arma::vec norm_prob = log_sum_exp(log_unnorm_prob);
        // Rcpp::Rcout << norm_prob << std::endl;
        new_assign.row(current_obs).fill(sample_clus(norm_prob, cluster_sm));
      }
    }
  } else {
    // Rcpp::Rcout << "final: split (none inactive)" << std::endl;
  }
  
  arma::vec new_alpha = adjust_alpha(K, new_assign, launch_alpha);
  
  // MH Update (log form)
  // Elements
  double launch_elem = 0.0;
  double final_elem = 0.0;
  double alpha_log = 0.0;
  double proposal = sm_indicator * std::log(0.5) * s_index.size();
  
  for(int k = 1; k <= K; ++k){
    // Calculate alpha
    if(launch_alpha_vec.at(k - 1) != new_alpha.at(k - 1)){
      if(new_alpha.at(k - 1) != 0){
        alpha_log += R::dgamma(new_alpha.at(k - 1), xi.at(k - 1), 1.0, 1);
        alpha_log += std::log(a_theta);
        alpha_log -= std::log(b_theta);
      } else {
        alpha_log -= R::dgamma(launch_alpha_vec.at(k - 1), 
                               xi.at(k - 1), 1.0, 1);
        alpha_log -= std::log(a_theta);
        alpha_log += std::log(b_theta);
      }
    }
    // Calculate Multinomial
    arma::uvec launch_elem_vec = arma::find(launch_assign == k);
    arma::uvec final_elem_vec = arma::find(new_assign == k);
    if(launch_elem_vec.size() > 0){
      launch_elem += launch_elem_vec.size() * 
        std::log(launch_alpha_vec.at(k - 1));
    }
    if(final_elem_vec.size() > 0){
      final_elem += final_elem_vec.size() * std::log(new_alpha.at(k - 1));
    }
  }
  
  double log_A = std::min(std::log(1), 
                          alpha_log + final_elem - launch_elem + proposal);
  double log_u = std::log(R::runif(0.0, 1.0));
  
  if(log_u >= log_A){
    new_assign = launch_assign;
    new_alpha = launch_alpha_vec;
  }
  
  result["new_assign"] = new_assign;
  result["new_alpha"] = new_alpha;
  
  return result;
}

// Step 4: Update alpha: -------------------------------------------------------
// * Both Univariate and Multivariate
// [[Rcpp::export]]
arma::vec update_alpha(int K, arma::vec alpha, arma::vec xi, 
                       arma::vec old_assign){

  arma::vec new_alpha = alpha;
  
  /* Input: maximum cluster (K),previous cluster weight (alpha), 
   *        hyperparameter for cluster (xi), 
   *        previous cluster assignment (old_assign).
   * Output: new cluster weight.
   */
  
  Rcpp::List List_active = active_inactive(K, old_assign);
  arma::uvec active_clus = List_active["active"];
  
  arma::vec n_xi_elem = -1.0 * arma::ones(active_clus.size());
  
  for(int k = 0; k < active_clus.size(); ++k){
    int clus_current = active_clus.at(k);
    arma::uvec obs_current_index = old_assign == clus_current;
    n_xi_elem.at(k) = sum(obs_current_index) + xi.at(clus_current - 1);
  }
  
  arma::mat psi_new = rdirichlet_cpp(1, n_xi_elem);
  
  for(int k = 0; k < active_clus.size(); ++k){
    int clus_current = active_clus.at(k);
    new_alpha.at(clus_current - 1) = sum(alpha) * psi_new(0, k);
  }
  
  return new_alpha;
}

// Final Function: -------------------------------------------------------------
// * Univariate
// [[Rcpp::export]]
arma::mat normal_uni(int K, int K_init, arma::vec y, arma::vec xi, 
                     arma::vec mu_0, arma::vec a_sigma, arma::vec b_sigma, 
                     arma::vec lambda, double a_theta, double b_theta, 
                     int sm_iter, int all_iter, int iter_print){
  
  // K_init should less than or equal to K.
  if(K_init > K){
    Rcpp::Rcout << "K must greater than or equal to K_init." << std::endl;
    Rcpp::Rcout << "Fixed: Let K_init equals to K." << std::endl;
    K_init = K;
  }
  
  // Initial the alpha and cluster assignment
  arma::vec old_assign = arma::conv_to<arma::vec>::
    from(arma::randi(y.size(), arma::distr_param(1, K_init)));
  
  arma::vec alpha_vec = arma::zeros(K);
  for(int k = 0; k < K_init; ++k){
    if(xi.at(k) != 0){
      alpha_vec.row(k).fill(R::rgamma(xi.at(k), 1.0));
    }
  }
  
  // Storing the cluster assignment for each iteration
  arma::mat clus_assign = -1 * arma::ones(old_assign.size(), all_iter);
  
  int i = 0;
  while(i < all_iter){
    
    if((i+1) % iter_print == 0){
      Rcpp::Rcout << (i+1) << std::endl;
    }
    
    // Step 1: Expand
    Rcpp::List step1 = expand_step_univariate(K, old_assign, alpha_vec, xi, 
                                              y, mu_0, a_sigma, b_sigma, 
                                              lambda, a_theta, b_theta);
    arma::vec step1_assign = step1["new_assign"];
    arma::vec step1_alpha = step1["new_alpha"];
    
    // Step 2: Reassign
    Rcpp::List step2 = cluster_assign_univariate(K, step1_assign, xi, y, 
                                                 step1_alpha, mu_0, a_sigma, 
                                                 b_sigma, lambda);
    arma::vec step2_assign = step2["new_assign"];
    arma::vec step2_alpha = step2["new_alpha"];
    
    // Step 3: Split-Merge
    Rcpp::List step3 = split_merge_univariate(K, step2_assign, step2_alpha, xi, 
                                              y, mu_0, a_sigma, b_sigma, lambda,
                                              a_theta, b_theta, sm_iter);
    arma::vec step3_assign = step3["new_assign"];
    arma::vec step3_alpha = step3["new_alpha"];
    
    // Step 4: Update alpha
    arma::vec step4_alpha = update_alpha(K, step3_alpha, xi, step3_assign);
    
    // Record the cluster assignment
    clus_assign.col(i) = step3_assign;
    
    // Prepare for the next iteration
    i += 1;
    old_assign = step3_assign;
    alpha_vec = step4_alpha;
  }
  
  return clus_assign.t();
}
