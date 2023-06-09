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

Rcpp::IntegerVector rmultinom_1(Rcpp::NumericVector &probs, unsigned int &N){
  
  /* Description: NA
   * Credit: https://gallery.rcpp.org/articles/recreating-rmultinom-and-rpois-with-rcpp/
   */
  
  Rcpp::IntegerVector outcome(N);
  rmultinom(1, probs.begin(), N, outcome.begin());
  return outcome;
}

// Finite Mixture Model: -------------------------------------------------------
// [[Rcpp::export]]
Rcpp::List fmm_rcpp(int iter, arma::vec y, unsigned int K_max, double a0, 
                    double b0, double mu0, double s20, double xi0, 
                    arma::vec ci_init){
  
  /* Description: NA 
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
        double l_re = std::log(nk_vec.size() + xi0);
        
        if(nk_vec.size() == 0){
          l_re += log_marginal(yi, mu0, s20, a0, b0, mu[k], s2[k]);
        } else {
          l_re += R::dnorm4(yi, mu[k], std::sqrt(s2[k]), 1);
        }
        
        log_realloc.row(k).fill(l_re);

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


// END: ------------------------------------------------------------------------