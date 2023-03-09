#include "RcppArmadillo.h"
// [[Rcpp::depends(RcppArmadillo)]]

#define pi 3.141592653589793238462643383280

// Note to self: ---------------------------------------------------------------


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
arma::mat uni_lmar(int K, arma::vec y, arma::vec a_sigma, arma::vec b_sigma,
                   arma::vec lambda_k, arma::vec mu_0){
  arma::mat log_mar(y.size(), K);
  
  // Calculate the posterior parameters
  arma::vec V_n = 1/(lambda_k + 1);
  arma::vec a_n = a_sigma + 0.5;
  
  arma::vec lmar_y = arma::lgamma(a_n) - arma::lgamma(a_sigma) +
    (0.5 * arma::log(V_n)) - (0.5 * arma::log(1/lambda_k)) -
    (0.5 * log(2 * pi) * arma::ones(K)) + (a_sigma % arma::log(b_sigma));
  
  for(int i = 0; i < y.size(); ++i){
    arma::vec yi_mu0 = arma::pow((y.at(i) * arma::ones(K)) - mu_0, 2.0);
    arma::vec b_n = b_sigma + (0.5 * (lambda_k % (1/(lambda_k + 1)) % yi_mu0));
    log_mar.row(i) = (lmar_y - (a_n % arma::log(b_n))).t();
  }
  
  return log_mar;
}

// [[Rcpp::export]]
double uni_log_marginal(double y, double a_sigma_K, double b_sigma_K,
                        double lambda_K, double mu_0_K){
  
  /* Description: This is the function for calculating the log of the 
   *              marginal likelihood of the data (Univariate Case).
   * Input: Data point (y) and hyperparameter for that cluster 
   *        (a_sigma, b_sigma, lambda, mu_0)
   * Output: log(p(yi|a_sigma, b_sigma, lambda, mu_0))
   */
  
  // Calculate the posterior parameters
  double a_n = a_sigma_K + 0.5;
  double V_n = 1/(lambda_K + 1);
  double b_n = b_sigma_K + 
    ((0.5) * (lambda_K/(lambda_K + 1)) * (std::pow(mu_0_K - y, 2.0)));
  
  // Calculate the log marginal
  double log_marginal = 0.0;
  log_marginal += (0.5 * log(V_n));
  log_marginal -= (0.5 * log(1/lambda_K));
  log_marginal += (a_sigma_K * log(b_sigma_K));
  log_marginal -= (a_n * log(b_n));
  log_marginal += lgamma(a_n);
  log_marginal -= lgamma(a_sigma_K);
  log_marginal -= (0.5 * log(pi));
  log_marginal -= log(2);
  
  return log_marginal;
}

// [[Rcpp::export]]
double multi_log_marginal(arma::vec y, arma::vec mu_0, double lambda_0, 
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
int uni_alloc(int i, arma::vec old_assign, arma::vec xi, arma::vec y, 
              arma::vec a_sigma, arma::vec b_sigma, arma::vec lambda, 
              arma::vec mu_0, arma::uvec active_clus){
  
  int new_assign = 0;
  
  // Retrieve the active cluster
  int K_active = active_clus.size();
  
  if(K_active == 1){
    // If we have only one active cluster, it should be allocated to this cluster.
    new_assign = active_clus.at(0);
  } else {
    // If not, we will reallocate observation i into one of them.
    // First, create two sets of the data: yi and not yi.
    double yi = y.at(i);
    arma::vec y_not_i = y;
    y.shed_row(i);
    arma::vec c_not_yi = old_assign;
    c_not_yi.shed_row(i);
    
    // Create a vector for storing a log unnormaolized probability
    arma::vec log_unnorm(K_active);
    
    for(int k = 0; k < K_active; ++k){
      int current_clus = active_clus.at(k);
      arma::uvec S_index = arma::find(c_not_yi == current_clus);
      arma::vec y_k = y_not_i.rows(S_index);
      
      // Select the hyperparameter that corresponding to the cluster k
      double a_k = a_sigma.at(current_clus - 1);
      double b_k = b_sigma.at(current_clus - 1);
      double lambda_k = lambda.at(current_clus - 1); 
      double mu_0k = mu_0.at(current_clus - 1);
      double xi_k = xi.at(current_clus- 1);
      double n_k = S_index.size();
      
      // Calculate the parameter for the posterior predictive distribution
      double a_n = a_k;
      double i_lambda_n = lambda_k;
      double mu_0n = mu_0k;
      double b_n = b_k;
      
      if(n_k > 0){
        a_n += (n_k/2);
        i_lambda_n += n_k;
        mu_0n = ((lambda_k * mu_0k) + (arma::accu(y_k)))/(lambda_k + n_k);
        b_n += (0.5 * n_k * arma::var(y_k));
        double diff_mean_y = mu_0k - arma::mean(arma::mean(y_k));
        b_n += (0.5 * std::pow(diff_mean_y, 2.0) * ((n_k * lambda_k)/(n_k + lambda_k)));
      }
      
      // Calculate the log unnormalized allocation probability
      double nu = 2 * a_n;
      double scale_coef = (b_n * (1 + (1/i_lambda_n)))/a_n;
      double log_t = R::dt((yi - mu_0n)/scale_coef, nu, 1) - log(scale_coef);
      
      log_unnorm.row(k).fill(log_t + log(n_k + xi_k));
    }
    
    // Normalized the log_unnorm by applying the log_sum_exp trick
    new_assign = sample_clus(log_sum_exp(log_unnorm), active_clus);
    arma::vec test_prob = log_sum_exp(log_unnorm);
  }
  
  return new_assign;
}

// [[Rcpp::export]]
int multi_alloc(int i, arma::vec old_assign, arma::vec xi, arma::mat y, 
                arma::mat mu_0, arma::vec lambda_0, arma::vec nu_0, 
                arma::cube L_0, arma::uvec active_clus){
  
  int new_assign;
  
  // Retrieve the active cluster
  int K_active = active_clus.size();
  
  if(K_active == 1){
    // If we have only one active cluster, it should be allocated to this cluster.
    new_assign = active_clus.at(0);
  } else {
    // If not, we will reallocate observation i into one of them.
    // First, create two sets of the data: yi and not yi.
    arma::vec yi = arma::conv_to<arma::vec>::from(y.row(i));
    arma::mat y_not_i = y;
    y.shed_row(i);
    arma::vec c_not_yi = old_assign;
    c_not_yi.shed_row(i);
    
    // Create a vector for storing a log unnormaolized probability
    arma::vec log_unnorm(K_active);
    
    for(int k = 0; k < K_active; ++k){
      int cc = active_clus.at(k);
      arma::uvec S_index = arma::find(c_not_yi == cc);
      
      // Select the hyperparameter that corresponding to the cluster k
      double d = yi.size();
      arma::vec mu_0k = arma::conv_to<arma::vec>::from(mu_0.row(cc - 1));
      arma::mat L_0k = L_0.slice(cc - 1);
      double lamb_0k = lambda_0.at(cc - 1);
      double nu_0k = nu_0.at(cc - 1);
      double xi_k = xi.at(cc - 1);
      double n_k = S_index.size();
      
      // Calculate the parameter for the posterior predictive distribution
      double nu_nk = nu_0k + n_k;
      double lamb_nk = lamb_0k + n_k;
      arma::vec ybar = arma::zeros(d);
      arma::mat S = arma::zeros(arma::size(L_0k));
      
      if(n_k > 0){
        ybar = arma::conv_to<arma::vec>::from(arma::mean(y_not_i, 0));
      }
      
      if(n_k > 0){
        for(int i = 0; i < n_k; ++i){
          arma::vec y_c = arma::conv_to<arma::vec>::from(y_not_i.row(i));
          arma::vec y_ybar = y_c - ybar;
          S += (y_ybar * arma::trans(y_ybar));
        }
      }
      
      arma::vec mu_nk = (((lamb_0k)/(lamb_0k + n_k)) * mu_0k) + 
        (((n_k)/(lamb_0k + n_k)) * ybar);
      
      arma::vec ybar_mu0 = arma::conv_to<arma::vec>::from(ybar - mu_0k);
      arma::mat L_nk = ((lamb_0k * n_k)/(lamb_0k + n_k)) * (ybar_mu0 * arma::trans(ybar_mu0));
      L_nk += (L_0k + S);
      
      arma::mat s_star = L_nk * ((lamb_nk + 1)/((lamb_nk) * (nu_nk - d + 1)));
      double val_Lnk;
      double sign_Lnk;
      bool log_det_Lnk = arma::log_det(val_Lnk, sign_Lnk, L_nk);
      
      arma::vec y_mu_n = arma::conv_to<arma::vec>::from(yi - mu_nk);
      
      double base_p = arma::as_scalar(arma::trans(y_mu_n) * arma::inv(s_star) * y_mu_n);
      base_p *= (1/(nu_nk - d + 1));
      base_p += 1.0;
      
      // Calculate log_allocate prob
      double log_val = 0.0;
      log_val += lgamma(0.5 * (nu_nk + 1));
      log_val -= lgamma(0.5 * (nu_nk - d + 1));
      log_val -= ((d/2) * log(nu_nk - d + 1));
      log_val -= ((d/2) * log(pi));
      log_val -= (0.5 * val_Lnk * sign_Lnk);
      log_val -= ((0.5 * (nu_nk + 1)) * log(base_p));
      
      log_unnorm.row(k).fill(log_val + log(n_k + xi_k));
    }
    
    // Normalized the log_unnorm by applying the log_sum_exp trick
    new_assign = sample_clus(log_sum_exp(log_unnorm), active_clus);
    
  }
  
  return new_assign;
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

// Finite Mixture Model: -------------------------------------------------------
// [[Rcpp::export]]
arma::mat uni_fmm(int iter, arma::vec assign_init, arma::vec y, double xi, 
                  double mu0, double a_sigma, double b_sigma, double lambda){
  
  /* Description: This function will run the finite mixture model. This function
   *              is adapted from the second step of SFMM (Allocation step).
   * Input: Iteration (iter), The initial assignment (assign_init), 
   *        data (y), cluster allocation hyperparameter (xi),
   *        data hyperparameter (mu_0, a_sigma, b_sigma and lambda)
   * Output: The cluster allocation for each iteration.
   * Note: We assume that the hyperparameter for every cluster are same.
   */
  
  // Create a matrix for storing a final result
  arma::mat clus_assign = arma::zeros(y.size(), iter);
  
  // Create the vector storing the hyperparameter for each cluster.
  arma::vec uniq_clus = arma::unique(assign_init);
  int K_max = uniq_clus.size() + 1;
  arma::vec xi_vec = xi * arma::ones(K_max); // xi
  arma::vec mu0_vec = mu0 * arma::ones(K_max); // mu0
  arma::vec a_sigma_vec = a_sigma * arma::ones(K_max); // a_sigma
  arma::vec b_sigma_vec = b_sigma * arma::ones(K_max); // b_sigma
  arma::vec lambda_vec = lambda * arma::ones(K_max); // lambda
  
  // Indicate the possible number of clusters.
  arma::uvec all_clus = arma::regspace<arma::uvec>(1, 1, K_max); 
  
  // Perform a FMM
  for(int t = 0; t < iter; ++t){ // Loop through the iteration
    for(int i = 0; i < y.size(); ++i){ // Loop through the observation
      int new_c = -1;
      new_c = uni_alloc(i, assign_init, xi_vec, y, a_sigma_vec, b_sigma_vec, 
                        lambda_vec, mu0_vec, all_clus);
      assign_init.row(i).fill(new_c);
    }
    // Record the cluster assignment
    clus_assign.col(t) = assign_init;
  }
  
  return clus_assign.t();
}

// Step 1: Update the cluster space: -------------------------------------------
// * Univariate
// [[Rcpp::export]]
Rcpp::List uni_expand(int K, arma::vec old_assign, arma::vec alpha,
                      arma::vec xi, arma::vec y, arma::mat ldata, 
                      double a_theta, double b_theta){
  
  Rcpp::List result;
  double accept_iter = 0.0;
  
  // Indicate the existed clusters and inactive clusters
  Rcpp::List List_clusters = active_inactive(K, old_assign);
  arma::uvec inactive_clus = List_clusters["inactive"];
  arma::uvec active_clus = List_clusters["active"];
  
  arma::vec new_alpha(K);
  arma::vec new_assign(y.size());
  
  if(active_clus.size() == K){
    new_alpha = alpha;
    new_assign = old_assign;
  } else {
    // Select a candidate cluster
    arma::vec samp_prob = arma::ones(inactive_clus.size())/inactive_clus.size();
    int candidate_clus = sample_clus(samp_prob, inactive_clus);
    
    // Sample alpha for new active cluster
    double alpha_k = 
      arma::as_scalar(arma::randg(1, arma::distr_param(xi.at(candidate_clus - 1), 1.0)));
    double sum_alpha = arma::sum(alpha);
    double sum_alpha_k = sum_alpha + alpha_k;
    
    for(int i = 0; i < y.size(); ++i){
      int cc = old_assign.at(i);
      arma::rowvec ldata_y = ldata.row(i);
      double log_a = std::min(0.0, ldata_y.at(candidate_clus - 1) - ldata_y.at(cc - 1) +
                              log(alpha_k) - log(alpha.at(cc - 1)) + log(sum_alpha) - log(sum_alpha_k) +
                              log(a_theta) - log(b_theta));
      double log_u = log(arma::randu());
      if(log_u <= log_a){
        accept_iter += 1.0;
        old_assign.row(i).fill(candidate_clus);
      }
    }
    
    new_assign = old_assign;
    alpha.row(candidate_clus - 1).fill(alpha_k);
    new_alpha = adjust_alpha(K, new_assign, alpha);
  }
  
  result["accept_prob"] = accept_iter/y.size();
  result["new_alpha"] = new_alpha;
  result["new_assign"] = new_assign;
  
  return result;
}

// * Multivariate
// [[Rcpp::export]]
Rcpp::List multi_expand_step(int K, arma::vec old_assign, arma::vec alpha,
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
      double log_prob = log_accept_prob.at(old_assign.at(i) - 1);
      
      // Prepare for log_marginal_multi function
      arma::vec yi = arma::conv_to<arma::vec>::from(y.row(i));
      arma::vec mu_0_new = arma::conv_to<arma::vec>::from(mu_0.row(candidate_clus - 1));
      arma::vec mu_0_old = arma::conv_to<arma::vec>::from(mu_0.row(old_assign.at(i) - 1));
      
      log_prob += multi_log_marginal(yi, mu_0_new, lambda_0.at(candidate_clus - 1), 
                                     nu_0.at(candidate_clus - 1), L_0.slice(candidate_clus - 1));
      
      log_prob -= multi_log_marginal(yi, mu_0_old, lambda_0.at(old_assign.at(i) - 1), 
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
Rcpp::List uni_cluster_assign(int K, arma::vec old_assign, arma::vec xi, 
                              arma::vec y, arma::vec alpha, arma::vec mu_0, 
                              arma::vec a_sigma, arma::vec b_sigma, arma::vec lambda){
  Rcpp::List result;
  
  // Indicate the active clusters.
  Rcpp::List List_clusters = active_inactive(K, old_assign);
  arma::uvec active_clus = List_clusters["active"];
  
  // Assign a new assignment
  for(int i = 0; i < old_assign.size(); ++i){
    int new_c = uni_alloc(i, old_assign, xi, y, a_sigma, b_sigma, lambda, 
                          mu_0, active_clus);
    old_assign.row(i).fill(new_c);
  }
  
  // Adjust an alpha vector
  arma::vec new_alpha = adjust_alpha(K, old_assign, alpha);
  
  result["new_assign"] = old_assign;
  result["new_alpha"] = new_alpha;
  
  return result;
}

// * Multivariate
// [[Rcpp::export]]
Rcpp::List multi_cluster_assign(int K, arma::vec old_assign, arma::vec xi, 
                                arma::mat y, arma::vec alpha, 
                                arma::mat mu_0, arma::vec lambda_0, 
                                arma::vec nu_0, arma::cube L_0){
  Rcpp::List result;
  
  Rcpp::List active_List = active_inactive(K, old_assign);
  arma::uvec active_clus = active_List["active"];
  
  // Assign a new assignment
  for(int i = 0; i < old_assign.size(); ++i){
    // Create the vector of the active cluster
    int new_c = multi_alloc(i, old_assign, xi, y, mu_0, lambda_0, nu_0, L_0, 
                            active_clus);
    old_assign.row(i).fill(new_c);
  }
  
  // Adjust an alpha vector
  arma::vec new_alpha = adjust_alpha(K, old_assign, alpha);
  
  result["new_assign"] = old_assign;
  result["new_alpha"] = new_alpha;
  
  return result;
}

// Step 3: Split-Merge: --------------------------------------------------------
// * Univariate 
// [[Rcpp::export]]
Rcpp::List uni_split_merge(int K, arma::vec old_assign, arma::vec alpha,
                           arma::vec xi, arma::vec y, arma::vec mu_0, 
                           arma::vec a_sigma, arma::vec b_sigma, 
                           arma::vec lambda, double a_theta, double b_theta, 
                           int sm_iter){
  Rcpp::List result;
  int accept_iter = 1;
  int split_i = 0;
  int split_k = 0;
  int merge_i = 0;
  
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
      int launch_c = uni_alloc(current_obs, launch_assign, xi, y, a_sigma, 
                               b_sigma, lambda, mu_0, cluster_launch);
      launch_assign.row(current_obs).fill(launch_c);
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
    sm_indicator = 1.0;
    new_assign.elem(s_index).fill(c_j);
    merge_i += 1;
  } else if((c_i == c_j) and (active_sm.size() != K)) { 
    // split in case that at least one cluster is inactive.
    sm_indicator = -1.0;
    split_i += 1;
    // sample a new inactive cluster
    arma::vec prob_inactive = arma::ones(inactive_sm.size())/inactive_sm.size();
    c_i = sample_clus(prob_inactive, inactive_sm);
    new_assign.row(obs_i).fill(c_i);
    launch_alpha.row(c_i - 1).fill(R::rgamma(xi.at(c_i - 1), 1.0));
    cluster_sm.row(0).fill(c_i);
    
    for(int i = 0; i < s_index.size(); ++i){
      int current_obs = s_index.at(i);
      if((current_obs != obs_i) and (current_obs != obs_j)){
        int sm_clus = uni_alloc(current_obs, new_assign, xi, y, a_sigma, 
                                b_sigma, lambda, mu_0, cluster_sm);
        new_assign.row(current_obs).fill(sm_clus);
      }
    }
  } else {
    // Rcpp::Rcout << "final: split (none inactive)" << std::endl;
    split_k += 1;
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
    accept_iter -= 1;
  }
  
  result["accept_prob"] = accept_iter;
  result["split"] = split_i;
  result["split_k"] = split_k;
  result["merge"] = merge_i;
  result["new_assign"] = new_assign;
  result["new_alpha"] = new_alpha;
  
  return result;
}

// * Multivariate
// [[Rcpp::export]]
Rcpp::List multi_split_merge(int K, arma::vec old_assign, arma::vec alpha,
                             arma::vec xi, arma::mat y, arma::mat mu_0, 
                             arma::vec lambda_0, arma::vec nu_0, 
                             arma::cube L_0, double a_theta, double b_theta, 
                             int sm_iter){
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
      int launch_c = multi_alloc(current_obs, launch_assign, xi, y, mu_0, 
                                 lambda_0, nu_0, L_0, cluster_launch);
      launch_assign.row(current_obs).fill(launch_c);
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
    sm_indicator = 1.0;
    new_assign.elem(s_index).fill(c_j);
  } else if((c_i == c_j) and (active_sm.size() != K)) { 
    // split in case that at least one cluster is inactive.
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
        int new_c = multi_alloc(current_obs, new_assign, xi, y, mu_0, lambda_0, 
                                nu_0, L_0, cluster_sm);
        new_assign.row(current_obs).fill(new_c);
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
Rcpp::List normal_uni(int K, int K_init, arma::vec y, arma::vec xi, 
                      arma::vec mu_0, arma::vec a_sigma, arma::vec b_sigma, 
                      arma::vec lambda, double a_theta, double b_theta, 
                      int sm_iter, int all_iter, int iter_print){
  
  /* This is the function for the univariate case. We assume that our data is 
   * followed the normal distribution. The user requires to specified 
   * (1) The total possible number of clusters. (K)
   * (2) The number of clusters for the initialization of the MCMC. (K_init)
   * (3) Hyperparameter for the cluster assignment. (xi)
   * (4) Hyperparameter for each of K clusters. (mu_0, a_sigma, b_sigma, lambda)
   * (5) a_theta, b_theta
   * (6) number of iterations. (sm_iter, all_iter)
   */
  
  Rcpp::List result;
  
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
  arma::mat alpha_val(alpha_vec.size(), all_iter);
  
  // Calculate the log marginal of the data for each observations and each clusters
  arma::mat log_data = uni_lmar(K, y, a_sigma, b_sigma, lambda, mu_0);
  
  // Accept prob for SM step
  arma::vec SM_split(all_iter);
  arma::vec SM_split_k(all_iter);
  arma::vec SM_merge(all_iter);
  arma::vec SM_accept(all_iter);
  
  int i = 0;
  while(i < all_iter){
    
    if((i+1) % iter_print == 0){
      Rcpp::Rcout << (i+1) << std::endl;
    }
    
    // Step 2: Reassign
    Rcpp::List step2 = uni_cluster_assign(K, old_assign, xi, y, alpha_vec, 
                                          mu_0, a_sigma, b_sigma, lambda);
    arma::vec step2_assign = step2["new_assign"];
    arma::vec step2_alpha = step2["new_alpha"];
    
    // Step 3: Split-Merge
    Rcpp::List step3 = uni_split_merge(K, step2_assign, step2_alpha, xi, y, 
                                       mu_0, a_sigma, b_sigma, lambda, a_theta, 
                                       b_theta, sm_iter);
    arma::vec step3_assign = step3["new_assign"];
    arma::vec step3_alpha = step3["new_alpha"];
    SM_accept.row(i).fill(step3["accept_prob"]);
    SM_split.row(i).fill(step3["split"]);
    SM_split_k.row(i).fill(step3["split_k"]);
    SM_merge.row(i).fill(step3["merge"]);
    
    // Step 4: Update alpha
    arma::vec step4_alpha = update_alpha(K, step3_alpha, xi, step3_assign);
    
    // Record the cluster assignment
    clus_assign.col(i) = step3_assign;
    alpha_val.col(i) = step4_alpha;
    
    // Prepare for the next iteration
    i += 1;
    old_assign = step3_assign;
    alpha_vec = step4_alpha;
  }
  
  result["assign"] = clus_assign.t();
  result["alpha"] = alpha_val.t();
  result["SM_accept"] = SM_accept;
  result["Split"] = SM_split;
  result["Split_k"] = SM_split_k;
  result["Merge"] = SM_merge;
  
  return result;
}

// SPMM: -----------------------------------------------------------------------
// [[Rcpp::export]]
int SPMM_uni_alloc(int i, arma::vec old_assign, arma::vec xi, arma::vec y, 
                   arma::vec a_sigma, arma::vec b_sigma, arma::vec lambda, 
                   arma::vec mu_0, arma::uvec active_clus){
  
  int new_assign;
  
  // Retrieve the active cluster
  int K_active = active_clus.size();
  
  double yi = y.at(i);
  arma::vec y_not_i = y;
  y.shed_row(i);
  arma::vec c_not_yi = old_assign;
  c_not_yi.shed_row(i);
  
  // Create a vector for storing a log unnormaolized probability
  arma::vec log_unnorm(K_active);
  
  for(int k = 0; k < K_active; ++k){
    
    int current_clus = active_clus.at(k);
    arma::uvec S_index = arma::find(c_not_yi == current_clus);
    arma::vec y_k = y_not_i.rows(S_index);
    
    // Select the hyperparameter that corresponding to the cluster k
    double a_k = a_sigma.at(current_clus - 1);
    double b_k = b_sigma.at(current_clus - 1);
    double lambda_k = lambda.at(current_clus - 1); 
    double mu_0k = mu_0.at(current_clus - 1);
    double xi_k = xi.at(current_clus- 1);
    double n_k = S_index.size();
    
    // Calculate the parameter for the posterior predictive distribution
    double a_n = a_k;
    double i_lambda_n = lambda_k;
    double mu_0n = mu_0k;
    double b_n = b_k;
    
    if(n_k > 0){
      a_n += (n_k/2);
      i_lambda_n += n_k;
      mu_0n = ((lambda_k * mu_0k) + (arma::accu(y_k)))/(lambda_k + n_k);
      b_n += (0.5 * n_k * arma::var(y_k));
      double diff_mean_y = mu_0k - arma::mean(arma::mean(y_k));
      b_n += (0.5 * std::pow(diff_mean_y, 2.0) * ((n_k * lambda_k)/(n_k + lambda_k)));
    }
    
    // Calculate the log unnormalized allocation probability
    double nu = 2 * a_n;
    double scale_coef = (b_n * (1 + (1/i_lambda_n)))/a_n;
    double log_t = R::dt((yi - mu_0n)/scale_coef, nu, 1) - log(scale_coef);
    
    log_unnorm.row(k).fill(log_t + log(n_k + xi_k));
  }
  
  // Normalized the log_unnorm by applying the log_sum_exp trick
  new_assign = sample_clus(log_sum_exp(log_unnorm), active_clus);
  arma::vec test_prob = log_sum_exp(log_unnorm);
  
  return new_assign;
}

// * Univariate
// [[Rcpp::export]]
arma::vec SPMM_uni_cluster_assign(int K, arma::vec old_assign, arma::vec xi, 
                                  arma::vec y, arma::vec mu_0, arma::vec a_sigma, 
                                  arma::vec b_sigma, arma::vec lambda){
  
  // Indicate the active clusters.
  arma::uvec active_clus = arma::conv_to<arma::uvec>::from(arma::linspace(1, K, K));
  
  // Assign a new assignment
  for(int i = 0; i < old_assign.size(); ++i){
    int new_c = SPMM_uni_alloc(i, old_assign, xi, y, a_sigma, b_sigma, lambda, 
                               mu_0, active_clus);
    old_assign.row(i).fill(new_c);
  }
  
  return old_assign;
}

// * Univariate
// [[Rcpp::export]]
Rcpp::List normal_SPMM_uni(int K_init, arma::vec y, arma::vec xi, 
                           arma::vec mu_0, arma::vec a_sigma, arma::vec b_sigma, 
                           arma::vec lambda, int all_iter, int iter_print){
  Rcpp::List result;
  
  int K = xi.size(); // maximum possible cluster
  
  // Initialize the cluster weight
  arma::vec alpha_vec(K);
  for(int i = 0; i < K; ++i){
    alpha_vec.row(i).fill(R::rgamma(xi.at(i), 1.0));
  }
  
  // Initial the cluster assignment
  arma::vec old_assign = init_seq(y.size(), K_init);
  
  // Storing the cluster assignment for each iteration
  arma::mat clus_assign = -1 * arma::ones(old_assign.size(), all_iter);
  arma::mat alpha_val(alpha_vec.size(), all_iter);
  
  // Calculate the log marginal of the data for each observations and each clusters
  arma::mat log_data = uni_lmar(K, y, a_sigma, b_sigma, lambda, mu_0);
  
  int i = 0;
  while(i < all_iter){
    
    if((i+1) % iter_print == 0){
      Rcpp::Rcout << (i+1) << std::endl;
    }
    
    // Step 1: Reassign
    arma::vec step1_assign = SPMM_uni_cluster_assign(K, old_assign, xi, y, 
                                                     mu_0, a_sigma, b_sigma, lambda);
    
    // Step 2: Update alpha
    arma::vec step2_alpha = update_alpha(K, alpha_vec, xi, step1_assign);
    
    // Record the cluster assignment
    clus_assign.col(i) = step1_assign;
    alpha_val.col(i) = step2_alpha;
    
    // Prepare for the next iteration
    i += 1;
    old_assign = step1_assign;
    alpha_vec = step2_alpha;
  }
  
  result["log_data"] = log_data;
  result["assign"] = clus_assign.t();
  result["alpha"] = alpha_val.t();
  
  return result;
}

// END: ------------------------------------------------------------------------