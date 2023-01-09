// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// active_inactive
Rcpp::List active_inactive(int K, arma::vec clus_assign);
RcppExport SEXP _ClusterNormal_active_inactive(SEXP KSEXP, SEXP clus_assignSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type clus_assign(clus_assignSEXP);
    rcpp_result_gen = Rcpp::wrap(active_inactive(K, clus_assign));
    return rcpp_result_gen;
END_RCPP
}
// sample_clus
int sample_clus(arma::vec norm_probs, arma::uvec active_clus);
RcppExport SEXP _ClusterNormal_sample_clus(SEXP norm_probsSEXP, SEXP active_clusSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type norm_probs(norm_probsSEXP);
    Rcpp::traits::input_parameter< arma::uvec >::type active_clus(active_clusSEXP);
    rcpp_result_gen = Rcpp::wrap(sample_clus(norm_probs, active_clus));
    return rcpp_result_gen;
END_RCPP
}
// log_multi_lgamma
double log_multi_lgamma(double a, double d);
RcppExport SEXP _ClusterNormal_log_multi_lgamma(SEXP aSEXP, SEXP dSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    Rcpp::traits::input_parameter< double >::type d(dSEXP);
    rcpp_result_gen = Rcpp::wrap(log_multi_lgamma(a, d));
    return rcpp_result_gen;
END_RCPP
}
// log_marginal_univariate
double log_marginal_univariate(double y, double a_sigma_K, double b_sigma_K, double lambda_K, double mu_0_K);
RcppExport SEXP _ClusterNormal_log_marginal_univariate(SEXP ySEXP, SEXP a_sigma_KSEXP, SEXP b_sigma_KSEXP, SEXP lambda_KSEXP, SEXP mu_0_KSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< double >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type a_sigma_K(a_sigma_KSEXP);
    Rcpp::traits::input_parameter< double >::type b_sigma_K(b_sigma_KSEXP);
    Rcpp::traits::input_parameter< double >::type lambda_K(lambda_KSEXP);
    Rcpp::traits::input_parameter< double >::type mu_0_K(mu_0_KSEXP);
    rcpp_result_gen = Rcpp::wrap(log_marginal_univariate(y, a_sigma_K, b_sigma_K, lambda_K, mu_0_K));
    return rcpp_result_gen;
END_RCPP
}
// log_marginal_multi
double log_marginal_multi(arma::vec y, arma::vec mu_0, double lambda_0, double nu_0, arma::mat L_0);
RcppExport SEXP _ClusterNormal_log_marginal_multi(SEXP ySEXP, SEXP mu_0SEXP, SEXP lambda_0SEXP, SEXP nu_0SEXP, SEXP L_0SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu_0(mu_0SEXP);
    Rcpp::traits::input_parameter< double >::type lambda_0(lambda_0SEXP);
    Rcpp::traits::input_parameter< double >::type nu_0(nu_0SEXP);
    Rcpp::traits::input_parameter< arma::mat >::type L_0(L_0SEXP);
    rcpp_result_gen = Rcpp::wrap(log_marginal_multi(y, mu_0, lambda_0, nu_0, L_0));
    return rcpp_result_gen;
END_RCPP
}
// log_allocate_prob_univariate
arma::vec log_allocate_prob_univariate(int i, arma::vec current_assign, arma::vec xi, arma::vec y, arma::vec a_sigma, arma::vec b_sigma, arma::vec lambda, arma::vec mu_0, arma::uvec active_clus);
RcppExport SEXP _ClusterNormal_log_allocate_prob_univariate(SEXP iSEXP, SEXP current_assignSEXP, SEXP xiSEXP, SEXP ySEXP, SEXP a_sigmaSEXP, SEXP b_sigmaSEXP, SEXP lambdaSEXP, SEXP mu_0SEXP, SEXP active_clusSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type i(iSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type current_assign(current_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi(xiSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type a_sigma(a_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type b_sigma(b_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu_0(mu_0SEXP);
    Rcpp::traits::input_parameter< arma::uvec >::type active_clus(active_clusSEXP);
    rcpp_result_gen = Rcpp::wrap(log_allocate_prob_univariate(i, current_assign, xi, y, a_sigma, b_sigma, lambda, mu_0, active_clus));
    return rcpp_result_gen;
END_RCPP
}
// log_allocate_prob
arma::vec log_allocate_prob(int i, arma::vec current_assign, arma::vec xi, arma::mat y, arma::mat gamma_hyper_mat, arma::uvec active_clus);
RcppExport SEXP _ClusterNormal_log_allocate_prob(SEXP iSEXP, SEXP current_assignSEXP, SEXP xiSEXP, SEXP ySEXP, SEXP gamma_hyper_matSEXP, SEXP active_clusSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type i(iSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type current_assign(current_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi(xiSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type gamma_hyper_mat(gamma_hyper_matSEXP);
    Rcpp::traits::input_parameter< arma::uvec >::type active_clus(active_clusSEXP);
    rcpp_result_gen = Rcpp::wrap(log_allocate_prob(i, current_assign, xi, y, gamma_hyper_mat, active_clus));
    return rcpp_result_gen;
END_RCPP
}
// log_sum_exp
arma::vec log_sum_exp(arma::vec log_unnorm_prob);
RcppExport SEXP _ClusterNormal_log_sum_exp(SEXP log_unnorm_probSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type log_unnorm_prob(log_unnorm_probSEXP);
    rcpp_result_gen = Rcpp::wrap(log_sum_exp(log_unnorm_prob));
    return rcpp_result_gen;
END_RCPP
}
// rdirichlet_cpp
arma::mat rdirichlet_cpp(int num_samples, arma::vec alpha_m);
RcppExport SEXP _ClusterNormal_rdirichlet_cpp(SEXP num_samplesSEXP, SEXP alpha_mSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type num_samples(num_samplesSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type alpha_m(alpha_mSEXP);
    rcpp_result_gen = Rcpp::wrap(rdirichlet_cpp(num_samples, alpha_m));
    return rcpp_result_gen;
END_RCPP
}
// expand_step_univariate
Rcpp::List expand_step_univariate(int K, arma::vec old_assign, arma::vec alpha, arma::vec xi, arma::vec y, arma::vec mu_0, arma::vec a_sigma, arma::vec b_sigma, arma::vec lambda, double a_theta, double b_theta);
RcppExport SEXP _ClusterNormal_expand_step_univariate(SEXP KSEXP, SEXP old_assignSEXP, SEXP alphaSEXP, SEXP xiSEXP, SEXP ySEXP, SEXP mu_0SEXP, SEXP a_sigmaSEXP, SEXP b_sigmaSEXP, SEXP lambdaSEXP, SEXP a_thetaSEXP, SEXP b_thetaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type old_assign(old_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi(xiSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu_0(mu_0SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type a_sigma(a_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type b_sigma(b_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type a_theta(a_thetaSEXP);
    Rcpp::traits::input_parameter< double >::type b_theta(b_thetaSEXP);
    rcpp_result_gen = Rcpp::wrap(expand_step_univariate(K, old_assign, alpha, xi, y, mu_0, a_sigma, b_sigma, lambda, a_theta, b_theta));
    return rcpp_result_gen;
END_RCPP
}
// expand_step_multi
Rcpp::List expand_step_multi(int K, arma::vec old_assign, arma::vec alpha, arma::vec xi, arma::mat y, arma::mat mu_0, arma::vec lambda_0, arma::vec nu_0, arma::cube L_0, double a_theta, double b_theta);
RcppExport SEXP _ClusterNormal_expand_step_multi(SEXP KSEXP, SEXP old_assignSEXP, SEXP alphaSEXP, SEXP xiSEXP, SEXP ySEXP, SEXP mu_0SEXP, SEXP lambda_0SEXP, SEXP nu_0SEXP, SEXP L_0SEXP, SEXP a_thetaSEXP, SEXP b_thetaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type old_assign(old_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi(xiSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::mat >::type mu_0(mu_0SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda_0(lambda_0SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type nu_0(nu_0SEXP);
    Rcpp::traits::input_parameter< arma::cube >::type L_0(L_0SEXP);
    Rcpp::traits::input_parameter< double >::type a_theta(a_thetaSEXP);
    Rcpp::traits::input_parameter< double >::type b_theta(b_thetaSEXP);
    rcpp_result_gen = Rcpp::wrap(expand_step_multi(K, old_assign, alpha, xi, y, mu_0, lambda_0, nu_0, L_0, a_theta, b_theta));
    return rcpp_result_gen;
END_RCPP
}
// cluster_assign_univariate
Rcpp::List cluster_assign_univariate(int K, arma::vec old_assign, arma::vec xi, arma::vec y, arma::vec alpha, arma::vec mu_0, arma::vec a_sigma, arma::vec b_sigma, arma::vec lambda);
RcppExport SEXP _ClusterNormal_cluster_assign_univariate(SEXP KSEXP, SEXP old_assignSEXP, SEXP xiSEXP, SEXP ySEXP, SEXP alphaSEXP, SEXP mu_0SEXP, SEXP a_sigmaSEXP, SEXP b_sigmaSEXP, SEXP lambdaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type old_assign(old_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi(xiSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu_0(mu_0SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type a_sigma(a_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type b_sigma(b_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda(lambdaSEXP);
    rcpp_result_gen = Rcpp::wrap(cluster_assign_univariate(K, old_assign, xi, y, alpha, mu_0, a_sigma, b_sigma, lambda));
    return rcpp_result_gen;
END_RCPP
}
// split_merge_univariate
Rcpp::List split_merge_univariate(int K, arma::vec old_assign, arma::vec alpha, arma::vec xi, arma::vec y, arma::vec mu_0, arma::vec a_sigma, arma::vec b_sigma, arma::vec lambda, double a_theta, double b_theta, int sm_iter);
RcppExport SEXP _ClusterNormal_split_merge_univariate(SEXP KSEXP, SEXP old_assignSEXP, SEXP alphaSEXP, SEXP xiSEXP, SEXP ySEXP, SEXP mu_0SEXP, SEXP a_sigmaSEXP, SEXP b_sigmaSEXP, SEXP lambdaSEXP, SEXP a_thetaSEXP, SEXP b_thetaSEXP, SEXP sm_iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type old_assign(old_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi(xiSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu_0(mu_0SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type a_sigma(a_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type b_sigma(b_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type a_theta(a_thetaSEXP);
    Rcpp::traits::input_parameter< double >::type b_theta(b_thetaSEXP);
    Rcpp::traits::input_parameter< int >::type sm_iter(sm_iterSEXP);
    rcpp_result_gen = Rcpp::wrap(split_merge_univariate(K, old_assign, alpha, xi, y, mu_0, a_sigma, b_sigma, lambda, a_theta, b_theta, sm_iter));
    return rcpp_result_gen;
END_RCPP
}
// update_alpha
arma::vec update_alpha(int K, arma::vec alpha, arma::vec xi, arma::vec old_assign);
RcppExport SEXP _ClusterNormal_update_alpha(SEXP KSEXP, SEXP alphaSEXP, SEXP xiSEXP, SEXP old_assignSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi(xiSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type old_assign(old_assignSEXP);
    rcpp_result_gen = Rcpp::wrap(update_alpha(K, alpha, xi, old_assign));
    return rcpp_result_gen;
END_RCPP
}
// normal_uni
arma::mat normal_uni(int K, int K_init, arma::vec y, arma::vec xi, arma::vec mu_0, arma::vec a_sigma, arma::vec b_sigma, arma::vec lambda, double a_theta, double b_theta, int sm_iter, int all_iter, int iter_print);
RcppExport SEXP _ClusterNormal_normal_uni(SEXP KSEXP, SEXP K_initSEXP, SEXP ySEXP, SEXP xiSEXP, SEXP mu_0SEXP, SEXP a_sigmaSEXP, SEXP b_sigmaSEXP, SEXP lambdaSEXP, SEXP a_thetaSEXP, SEXP b_thetaSEXP, SEXP sm_iterSEXP, SEXP all_iterSEXP, SEXP iter_printSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type K_init(K_initSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi(xiSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu_0(mu_0SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type a_sigma(a_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type b_sigma(b_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type a_theta(a_thetaSEXP);
    Rcpp::traits::input_parameter< double >::type b_theta(b_thetaSEXP);
    Rcpp::traits::input_parameter< int >::type sm_iter(sm_iterSEXP);
    Rcpp::traits::input_parameter< int >::type all_iter(all_iterSEXP);
    Rcpp::traits::input_parameter< int >::type iter_print(iter_printSEXP);
    rcpp_result_gen = Rcpp::wrap(normal_uni(K, K_init, y, xi, mu_0, a_sigma, b_sigma, lambda, a_theta, b_theta, sm_iter, all_iter, iter_print));
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_hello_world
arma::mat rcpparma_hello_world();
RcppExport SEXP _ClusterNormal_rcpparma_hello_world() {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    rcpp_result_gen = Rcpp::wrap(rcpparma_hello_world());
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_outerproduct
arma::mat rcpparma_outerproduct(const arma::colvec& x);
RcppExport SEXP _ClusterNormal_rcpparma_outerproduct(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpparma_outerproduct(x));
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_innerproduct
double rcpparma_innerproduct(const arma::colvec& x);
RcppExport SEXP _ClusterNormal_rcpparma_innerproduct(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpparma_innerproduct(x));
    return rcpp_result_gen;
END_RCPP
}
// rcpparma_bothproducts
Rcpp::List rcpparma_bothproducts(const arma::colvec& x);
RcppExport SEXP _ClusterNormal_rcpparma_bothproducts(SEXP xSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::colvec& >::type x(xSEXP);
    rcpp_result_gen = Rcpp::wrap(rcpparma_bothproducts(x));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_ClusterNormal_active_inactive", (DL_FUNC) &_ClusterNormal_active_inactive, 2},
    {"_ClusterNormal_sample_clus", (DL_FUNC) &_ClusterNormal_sample_clus, 2},
    {"_ClusterNormal_log_multi_lgamma", (DL_FUNC) &_ClusterNormal_log_multi_lgamma, 2},
    {"_ClusterNormal_log_marginal_univariate", (DL_FUNC) &_ClusterNormal_log_marginal_univariate, 5},
    {"_ClusterNormal_log_marginal_multi", (DL_FUNC) &_ClusterNormal_log_marginal_multi, 5},
    {"_ClusterNormal_log_allocate_prob_univariate", (DL_FUNC) &_ClusterNormal_log_allocate_prob_univariate, 9},
    {"_ClusterNormal_log_allocate_prob", (DL_FUNC) &_ClusterNormal_log_allocate_prob, 6},
    {"_ClusterNormal_log_sum_exp", (DL_FUNC) &_ClusterNormal_log_sum_exp, 1},
    {"_ClusterNormal_rdirichlet_cpp", (DL_FUNC) &_ClusterNormal_rdirichlet_cpp, 2},
    {"_ClusterNormal_expand_step_univariate", (DL_FUNC) &_ClusterNormal_expand_step_univariate, 11},
    {"_ClusterNormal_expand_step_multi", (DL_FUNC) &_ClusterNormal_expand_step_multi, 11},
    {"_ClusterNormal_cluster_assign_univariate", (DL_FUNC) &_ClusterNormal_cluster_assign_univariate, 9},
    {"_ClusterNormal_split_merge_univariate", (DL_FUNC) &_ClusterNormal_split_merge_univariate, 12},
    {"_ClusterNormal_update_alpha", (DL_FUNC) &_ClusterNormal_update_alpha, 4},
    {"_ClusterNormal_normal_uni", (DL_FUNC) &_ClusterNormal_normal_uni, 13},
    {"_ClusterNormal_rcpparma_hello_world", (DL_FUNC) &_ClusterNormal_rcpparma_hello_world, 0},
    {"_ClusterNormal_rcpparma_outerproduct", (DL_FUNC) &_ClusterNormal_rcpparma_outerproduct, 1},
    {"_ClusterNormal_rcpparma_innerproduct", (DL_FUNC) &_ClusterNormal_rcpparma_innerproduct, 1},
    {"_ClusterNormal_rcpparma_bothproducts", (DL_FUNC) &_ClusterNormal_rcpparma_bothproducts, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_ClusterNormal(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
