// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

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
// log_marginal
double log_marginal(arma::vec y, int ci, arma::vec mu0_cluster, arma::vec lambda_cluster, arma::vec a_sigma_cluster, arma::vec b_sigma_cluster);
RcppExport SEXP _ClusterNormal_log_marginal(SEXP ySEXP, SEXP ciSEXP, SEXP mu0_clusterSEXP, SEXP lambda_clusterSEXP, SEXP a_sigma_clusterSEXP, SEXP b_sigma_clusterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< int >::type ci(ciSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu0_cluster(mu0_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda_cluster(lambda_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type a_sigma_cluster(a_sigma_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type b_sigma_cluster(b_sigma_clusterSEXP);
    rcpp_result_gen = Rcpp::wrap(log_marginal(y, ci, mu0_cluster, lambda_cluster, a_sigma_cluster, b_sigma_cluster));
    return rcpp_result_gen;
END_RCPP
}
// log_posterior
double log_posterior(arma::vec y_new, arma::vec data, int ci, arma::vec mu0_cluster, arma::vec lambda_cluster, arma::vec a_sigma_cluster, arma::vec b_sigma_cluster);
RcppExport SEXP _ClusterNormal_log_posterior(SEXP y_newSEXP, SEXP dataSEXP, SEXP ciSEXP, SEXP mu0_clusterSEXP, SEXP lambda_clusterSEXP, SEXP a_sigma_clusterSEXP, SEXP b_sigma_clusterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type y_new(y_newSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type data(dataSEXP);
    Rcpp::traits::input_parameter< int >::type ci(ciSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu0_cluster(mu0_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda_cluster(lambda_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type a_sigma_cluster(a_sigma_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type b_sigma_cluster(b_sigma_clusterSEXP);
    rcpp_result_gen = Rcpp::wrap(log_posterior(y_new, data, ci, mu0_cluster, lambda_cluster, a_sigma_cluster, b_sigma_cluster));
    return rcpp_result_gen;
END_RCPP
}
// rmultinom_1
int rmultinom_1(arma::vec unnorm_prob, unsigned int N);
RcppExport SEXP _ClusterNormal_rmultinom_1(SEXP unnorm_probSEXP, SEXP NSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type unnorm_prob(unnorm_probSEXP);
    Rcpp::traits::input_parameter< unsigned int >::type N(NSEXP);
    rcpp_result_gen = Rcpp::wrap(rmultinom_1(unnorm_prob, N));
    return rcpp_result_gen;
END_RCPP
}
// fmm_iter
arma::vec fmm_iter(int K, arma::vec old_assign, arma::vec y, arma::vec mu0_cluster, arma::vec lambda_cluster, arma::vec a_sigma_cluster, arma::vec b_sigma_cluster, arma::vec xi_cluster);
RcppExport SEXP _ClusterNormal_fmm_iter(SEXP KSEXP, SEXP old_assignSEXP, SEXP ySEXP, SEXP mu0_clusterSEXP, SEXP lambda_clusterSEXP, SEXP a_sigma_clusterSEXP, SEXP b_sigma_clusterSEXP, SEXP xi_clusterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type old_assign(old_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu0_cluster(mu0_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda_cluster(lambda_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type a_sigma_cluster(a_sigma_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type b_sigma_cluster(b_sigma_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi_cluster(xi_clusterSEXP);
    rcpp_result_gen = Rcpp::wrap(fmm_iter(K, old_assign, y, mu0_cluster, lambda_cluster, a_sigma_cluster, b_sigma_cluster, xi_cluster));
    return rcpp_result_gen;
END_RCPP
}
// fmm
arma::mat fmm(int iter, int K, arma::vec old_assign, arma::vec y, arma::vec mu0_cluster, arma::vec lambda_cluster, arma::vec a_sigma_cluster, arma::vec b_sigma_cluster, arma::vec xi_cluster);
RcppExport SEXP _ClusterNormal_fmm(SEXP iterSEXP, SEXP KSEXP, SEXP old_assignSEXP, SEXP ySEXP, SEXP mu0_clusterSEXP, SEXP lambda_clusterSEXP, SEXP a_sigma_clusterSEXP, SEXP b_sigma_clusterSEXP, SEXP xi_clusterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type iter(iterSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type old_assign(old_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu0_cluster(mu0_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda_cluster(lambda_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type a_sigma_cluster(a_sigma_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type b_sigma_cluster(b_sigma_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi_cluster(xi_clusterSEXP);
    rcpp_result_gen = Rcpp::wrap(fmm(iter, K, old_assign, y, mu0_cluster, lambda_cluster, a_sigma_cluster, b_sigma_cluster, xi_cluster));
    return rcpp_result_gen;
END_RCPP
}
// adjust_alpha
arma::vec adjust_alpha(arma::vec cluster_assign, arma::vec old_alpha);
RcppExport SEXP _ClusterNormal_adjust_alpha(SEXP cluster_assignSEXP, SEXP old_alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type cluster_assign(cluster_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type old_alpha(old_alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(adjust_alpha(cluster_assign, old_alpha));
    return rcpp_result_gen;
END_RCPP
}
// split_launch
arma::vec split_launch(arma::vec old_assign, arma::vec y, arma::vec mu0_cluster, arma::vec lambda_cluster, arma::vec a_sigma_cluster, arma::vec b_sigma_cluster, arma::vec sm_cluster, arma::uvec S_index);
RcppExport SEXP _ClusterNormal_split_launch(SEXP old_assignSEXP, SEXP ySEXP, SEXP mu0_clusterSEXP, SEXP lambda_clusterSEXP, SEXP a_sigma_clusterSEXP, SEXP b_sigma_clusterSEXP, SEXP sm_clusterSEXP, SEXP S_indexSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type old_assign(old_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu0_cluster(mu0_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda_cluster(lambda_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type a_sigma_cluster(a_sigma_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type b_sigma_cluster(b_sigma_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type sm_cluster(sm_clusterSEXP);
    Rcpp::traits::input_parameter< arma::uvec >::type S_index(S_indexSEXP);
    rcpp_result_gen = Rcpp::wrap(split_launch(old_assign, y, mu0_cluster, lambda_cluster, a_sigma_cluster, b_sigma_cluster, sm_cluster, S_index));
    return rcpp_result_gen;
END_RCPP
}
// log_proposal
double log_proposal(arma::vec c1, arma::vec c2, arma::vec y, arma::vec xi_cluster, arma::vec mu0_cluster, arma::vec lambda_cluster, arma::vec a_sigma_cluster, arma::vec b_sigma_cluster, arma::vec sm_cluster, arma::uvec S_index);
RcppExport SEXP _ClusterNormal_log_proposal(SEXP c1SEXP, SEXP c2SEXP, SEXP ySEXP, SEXP xi_clusterSEXP, SEXP mu0_clusterSEXP, SEXP lambda_clusterSEXP, SEXP a_sigma_clusterSEXP, SEXP b_sigma_clusterSEXP, SEXP sm_clusterSEXP, SEXP S_indexSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type c1(c1SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type c2(c2SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi_cluster(xi_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu0_cluster(mu0_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda_cluster(lambda_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type a_sigma_cluster(a_sigma_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type b_sigma_cluster(b_sigma_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type sm_cluster(sm_clusterSEXP);
    Rcpp::traits::input_parameter< arma::uvec >::type S_index(S_indexSEXP);
    rcpp_result_gen = Rcpp::wrap(log_proposal(c1, c2, y, xi_cluster, mu0_cluster, lambda_cluster, a_sigma_cluster, b_sigma_cluster, sm_cluster, S_index));
    return rcpp_result_gen;
END_RCPP
}
// log_prior_cluster
double log_prior_cluster(arma::vec cluster_assign, arma::vec xi_cluster);
RcppExport SEXP _ClusterNormal_log_prior_cluster(SEXP cluster_assignSEXP, SEXP xi_clusterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type cluster_assign(cluster_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi_cluster(xi_clusterSEXP);
    rcpp_result_gen = Rcpp::wrap(log_prior_cluster(cluster_assign, xi_cluster));
    return rcpp_result_gen;
END_RCPP
}
// SFDM_realloc
Rcpp::List SFDM_realloc(arma::vec old_assign, arma::vec y, arma::vec alpha_vec, arma::vec mu0_cluster, arma::vec lambda_cluster, arma::vec a_sigma_cluster, arma::vec b_sigma_cluster, arma::vec xi_cluster);
RcppExport SEXP _ClusterNormal_SFDM_realloc(SEXP old_assignSEXP, SEXP ySEXP, SEXP alpha_vecSEXP, SEXP mu0_clusterSEXP, SEXP lambda_clusterSEXP, SEXP a_sigma_clusterSEXP, SEXP b_sigma_clusterSEXP, SEXP xi_clusterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type old_assign(old_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type alpha_vec(alpha_vecSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu0_cluster(mu0_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda_cluster(lambda_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type a_sigma_cluster(a_sigma_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type b_sigma_cluster(b_sigma_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi_cluster(xi_clusterSEXP);
    rcpp_result_gen = Rcpp::wrap(SFDM_realloc(old_assign, y, alpha_vec, mu0_cluster, lambda_cluster, a_sigma_cluster, b_sigma_cluster, xi_cluster));
    return rcpp_result_gen;
END_RCPP
}
// SFDM_SM
Rcpp::List SFDM_SM(int K, arma::vec old_assign, arma::vec y, arma::vec alpha_vec, arma::vec mu0_cluster, arma::vec lambda_cluster, arma::vec a_sigma_cluster, arma::vec b_sigma_cluster, arma::vec xi_cluster, int launch_iter, double a_theta, double b_theta);
RcppExport SEXP _ClusterNormal_SFDM_SM(SEXP KSEXP, SEXP old_assignSEXP, SEXP ySEXP, SEXP alpha_vecSEXP, SEXP mu0_clusterSEXP, SEXP lambda_clusterSEXP, SEXP a_sigma_clusterSEXP, SEXP b_sigma_clusterSEXP, SEXP xi_clusterSEXP, SEXP launch_iterSEXP, SEXP a_thetaSEXP, SEXP b_thetaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type old_assign(old_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type alpha_vec(alpha_vecSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu0_cluster(mu0_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda_cluster(lambda_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type a_sigma_cluster(a_sigma_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type b_sigma_cluster(b_sigma_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi_cluster(xi_clusterSEXP);
    Rcpp::traits::input_parameter< int >::type launch_iter(launch_iterSEXP);
    Rcpp::traits::input_parameter< double >::type a_theta(a_thetaSEXP);
    Rcpp::traits::input_parameter< double >::type b_theta(b_thetaSEXP);
    rcpp_result_gen = Rcpp::wrap(SFDM_SM(K, old_assign, y, alpha_vec, mu0_cluster, lambda_cluster, a_sigma_cluster, b_sigma_cluster, xi_cluster, launch_iter, a_theta, b_theta));
    return rcpp_result_gen;
END_RCPP
}
// SFDM_alpha
Rcpp::List SFDM_alpha(arma::vec clus_assign, arma::vec xi_cluster, arma::vec alpha_vec, double old_u);
RcppExport SEXP _ClusterNormal_SFDM_alpha(SEXP clus_assignSEXP, SEXP xi_clusterSEXP, SEXP alpha_vecSEXP, SEXP old_uSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type clus_assign(clus_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi_cluster(xi_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type alpha_vec(alpha_vecSEXP);
    Rcpp::traits::input_parameter< double >::type old_u(old_uSEXP);
    rcpp_result_gen = Rcpp::wrap(SFDM_alpha(clus_assign, xi_cluster, alpha_vec, old_u));
    return rcpp_result_gen;
END_RCPP
}
// SFDM_model
Rcpp::List SFDM_model(int iter, int K, arma::vec init_assign, arma::vec y, arma::vec mu0_cluster, arma::vec lambda_cluster, arma::vec a_sigma_cluster, arma::vec b_sigma_cluster, arma::vec xi_cluster, double a_theta, double b_theta, int launch_iter, int print_iter);
RcppExport SEXP _ClusterNormal_SFDM_model(SEXP iterSEXP, SEXP KSEXP, SEXP init_assignSEXP, SEXP ySEXP, SEXP mu0_clusterSEXP, SEXP lambda_clusterSEXP, SEXP a_sigma_clusterSEXP, SEXP b_sigma_clusterSEXP, SEXP xi_clusterSEXP, SEXP a_thetaSEXP, SEXP b_thetaSEXP, SEXP launch_iterSEXP, SEXP print_iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type iter(iterSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type init_assign(init_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu0_cluster(mu0_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda_cluster(lambda_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type a_sigma_cluster(a_sigma_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type b_sigma_cluster(b_sigma_clusterSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi_cluster(xi_clusterSEXP);
    Rcpp::traits::input_parameter< double >::type a_theta(a_thetaSEXP);
    Rcpp::traits::input_parameter< double >::type b_theta(b_thetaSEXP);
    Rcpp::traits::input_parameter< int >::type launch_iter(launch_iterSEXP);
    Rcpp::traits::input_parameter< int >::type print_iter(print_iterSEXP);
    rcpp_result_gen = Rcpp::wrap(SFDM_model(iter, K, init_assign, y, mu0_cluster, lambda_cluster, a_sigma_cluster, b_sigma_cluster, xi_cluster, a_theta, b_theta, launch_iter, print_iter));
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
    {"_ClusterNormal_log_sum_exp", (DL_FUNC) &_ClusterNormal_log_sum_exp, 1},
    {"_ClusterNormal_log_marginal", (DL_FUNC) &_ClusterNormal_log_marginal, 6},
    {"_ClusterNormal_log_posterior", (DL_FUNC) &_ClusterNormal_log_posterior, 7},
    {"_ClusterNormal_rmultinom_1", (DL_FUNC) &_ClusterNormal_rmultinom_1, 2},
    {"_ClusterNormal_fmm_iter", (DL_FUNC) &_ClusterNormal_fmm_iter, 8},
    {"_ClusterNormal_fmm", (DL_FUNC) &_ClusterNormal_fmm, 9},
    {"_ClusterNormal_adjust_alpha", (DL_FUNC) &_ClusterNormal_adjust_alpha, 2},
    {"_ClusterNormal_split_launch", (DL_FUNC) &_ClusterNormal_split_launch, 8},
    {"_ClusterNormal_log_proposal", (DL_FUNC) &_ClusterNormal_log_proposal, 10},
    {"_ClusterNormal_log_prior_cluster", (DL_FUNC) &_ClusterNormal_log_prior_cluster, 2},
    {"_ClusterNormal_SFDM_realloc", (DL_FUNC) &_ClusterNormal_SFDM_realloc, 8},
    {"_ClusterNormal_SFDM_SM", (DL_FUNC) &_ClusterNormal_SFDM_SM, 12},
    {"_ClusterNormal_SFDM_alpha", (DL_FUNC) &_ClusterNormal_SFDM_alpha, 4},
    {"_ClusterNormal_SFDM_model", (DL_FUNC) &_ClusterNormal_SFDM_model, 13},
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
