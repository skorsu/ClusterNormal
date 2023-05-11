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
// fmm_log_alloc_prob
arma::vec fmm_log_alloc_prob(int K, int i, arma::vec old_assign, arma::vec xi, arma::vec y, arma::vec a_sigma, arma::vec b_sigma, arma::vec lambda, arma::vec mu0);
RcppExport SEXP _ClusterNormal_fmm_log_alloc_prob(SEXP KSEXP, SEXP iSEXP, SEXP old_assignSEXP, SEXP xiSEXP, SEXP ySEXP, SEXP a_sigmaSEXP, SEXP b_sigmaSEXP, SEXP lambdaSEXP, SEXP mu0SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type i(iSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type old_assign(old_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi(xiSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type a_sigma(a_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type b_sigma(b_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu0(mu0SEXP);
    rcpp_result_gen = Rcpp::wrap(fmm_log_alloc_prob(K, i, old_assign, xi, y, a_sigma, b_sigma, lambda, mu0));
    return rcpp_result_gen;
END_RCPP
}
// fmm_samp_new
int fmm_samp_new(int K, arma::vec log_alloc);
RcppExport SEXP _ClusterNormal_fmm_samp_new(SEXP KSEXP, SEXP log_allocSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type log_alloc(log_allocSEXP);
    rcpp_result_gen = Rcpp::wrap(fmm_samp_new(K, log_alloc));
    return rcpp_result_gen;
END_RCPP
}
// fmm_mod
arma::mat fmm_mod(int t, int K, arma::vec old_assign, arma::vec xi, arma::vec y, arma::vec a_sigma, arma::vec b_sigma, arma::vec lambda, arma::vec mu0);
RcppExport SEXP _ClusterNormal_fmm_mod(SEXP tSEXP, SEXP KSEXP, SEXP old_assignSEXP, SEXP xiSEXP, SEXP ySEXP, SEXP a_sigmaSEXP, SEXP b_sigmaSEXP, SEXP lambdaSEXP, SEXP mu0SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type t(tSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type old_assign(old_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi(xiSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type a_sigma(a_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type b_sigma(b_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu0(mu0SEXP);
    rcpp_result_gen = Rcpp::wrap(fmm_mod(t, K, old_assign, xi, y, a_sigma, b_sigma, lambda, mu0));
    return rcpp_result_gen;
END_RCPP
}
// log_alloc_prob
arma::mat log_alloc_prob(int i, arma::vec active_clus, arma::vec old_assign, arma::vec xi, arma::vec y, arma::vec a_sigma, arma::vec b_sigma, arma::vec lambda, arma::vec mu0, bool restricted);
RcppExport SEXP _ClusterNormal_log_alloc_prob(SEXP iSEXP, SEXP active_clusSEXP, SEXP old_assignSEXP, SEXP xiSEXP, SEXP ySEXP, SEXP a_sigmaSEXP, SEXP b_sigmaSEXP, SEXP lambdaSEXP, SEXP mu0SEXP, SEXP restrictedSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type i(iSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type active_clus(active_clusSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type old_assign(old_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi(xiSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type a_sigma(a_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type b_sigma(b_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu0(mu0SEXP);
    Rcpp::traits::input_parameter< bool >::type restricted(restrictedSEXP);
    rcpp_result_gen = Rcpp::wrap(log_alloc_prob(i, active_clus, old_assign, xi, y, a_sigma, b_sigma, lambda, mu0, restricted));
    return rcpp_result_gen;
END_RCPP
}
// samp_new
int samp_new(arma::mat log_prob_mat);
RcppExport SEXP _ClusterNormal_samp_new(SEXP log_prob_matSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type log_prob_mat(log_prob_matSEXP);
    rcpp_result_gen = Rcpp::wrap(samp_new(log_prob_mat));
    return rcpp_result_gen;
END_RCPP
}
// log_marginal
double log_marginal(arma::vec clus_assign, arma::vec y, arma::vec a_sigma, arma::vec b_sigma, arma::vec lambda, arma::vec mu0);
RcppExport SEXP _ClusterNormal_log_marginal(SEXP clus_assignSEXP, SEXP ySEXP, SEXP a_sigmaSEXP, SEXP b_sigmaSEXP, SEXP lambdaSEXP, SEXP mu0SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type clus_assign(clus_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type a_sigma(a_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type b_sigma(b_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu0(mu0SEXP);
    rcpp_result_gen = Rcpp::wrap(log_marginal(clus_assign, y, a_sigma, b_sigma, lambda, mu0));
    return rcpp_result_gen;
END_RCPP
}
// log_likelihood
double log_likelihood(arma::vec clus_assign, arma::vec y, arma::vec a_sigma, arma::vec b_sigma, arma::vec lambda, arma::vec mu0);
RcppExport SEXP _ClusterNormal_log_likelihood(SEXP clus_assignSEXP, SEXP ySEXP, SEXP a_sigmaSEXP, SEXP b_sigmaSEXP, SEXP lambdaSEXP, SEXP mu0SEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type clus_assign(clus_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type a_sigma(a_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type b_sigma(b_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu0(mu0SEXP);
    rcpp_result_gen = Rcpp::wrap(log_likelihood(clus_assign, y, a_sigma, b_sigma, lambda, mu0));
    return rcpp_result_gen;
END_RCPP
}
// log_cluster_param
double log_cluster_param(arma::vec clus_assign, arma::vec alpha);
RcppExport SEXP _ClusterNormal_log_cluster_param(SEXP clus_assignSEXP, SEXP alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type clus_assign(clus_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(log_cluster_param(clus_assign, alpha));
    return rcpp_result_gen;
END_RCPP
}
// log_gamma_cluster
double log_gamma_cluster(arma::vec alpha, arma::vec xi, arma::vec clus_assign);
RcppExport SEXP _ClusterNormal_log_gamma_cluster(SEXP alphaSEXP, SEXP xiSEXP, SEXP clus_assignSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi(xiSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type clus_assign(clus_assignSEXP);
    rcpp_result_gen = Rcpp::wrap(log_gamma_cluster(alpha, xi, clus_assign));
    return rcpp_result_gen;
END_RCPP
}
// log_proposal
double log_proposal(arma::vec c1, arma::vec c2, arma::uvec S, arma::vec s_clus, arma::vec y, arma::vec xi, arma::vec mu0, arma::vec a_sigma, arma::vec b_sigma, arma::vec lambda);
RcppExport SEXP _ClusterNormal_log_proposal(SEXP c1SEXP, SEXP c2SEXP, SEXP SSEXP, SEXP s_clusSEXP, SEXP ySEXP, SEXP xiSEXP, SEXP mu0SEXP, SEXP a_sigmaSEXP, SEXP b_sigmaSEXP, SEXP lambdaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type c1(c1SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type c2(c2SEXP);
    Rcpp::traits::input_parameter< arma::uvec >::type S(SSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type s_clus(s_clusSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi(xiSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu0(mu0SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type a_sigma(a_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type b_sigma(b_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda(lambdaSEXP);
    rcpp_result_gen = Rcpp::wrap(log_proposal(c1, c2, S, s_clus, y, xi, mu0, a_sigma, b_sigma, lambda));
    return rcpp_result_gen;
END_RCPP
}
// log_prior
double log_prior(arma::vec clus_assign, arma::vec xi);
RcppExport SEXP _ClusterNormal_log_prior(SEXP clus_assignSEXP, SEXP xiSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::vec >::type clus_assign(clus_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi(xiSEXP);
    rcpp_result_gen = Rcpp::wrap(log_prior(clus_assign, xi));
    return rcpp_result_gen;
END_RCPP
}
// SFDM_allocate
Rcpp::List SFDM_allocate(int K, arma::vec old_assign, arma::vec xi, arma::vec y, arma::vec a_sigma, arma::vec b_sigma, arma::vec lambda, arma::vec mu0, arma::vec old_alpha);
RcppExport SEXP _ClusterNormal_SFDM_allocate(SEXP KSEXP, SEXP old_assignSEXP, SEXP xiSEXP, SEXP ySEXP, SEXP a_sigmaSEXP, SEXP b_sigmaSEXP, SEXP lambdaSEXP, SEXP mu0SEXP, SEXP old_alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type old_assign(old_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi(xiSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type a_sigma(a_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type b_sigma(b_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu0(mu0SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type old_alpha(old_alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(SFDM_allocate(K, old_assign, xi, y, a_sigma, b_sigma, lambda, mu0, old_alpha));
    return rcpp_result_gen;
END_RCPP
}
// SFDM_SM
Rcpp::List SFDM_SM(int K, arma::vec old_assign, arma::vec old_alpha, arma::vec xi, arma::vec y, arma::vec mu0, arma::vec a_sigma, arma::vec b_sigma, arma::vec lambda, double a_theta, double b_theta, int launch_iter);
RcppExport SEXP _ClusterNormal_SFDM_SM(SEXP KSEXP, SEXP old_assignSEXP, SEXP old_alphaSEXP, SEXP xiSEXP, SEXP ySEXP, SEXP mu0SEXP, SEXP a_sigmaSEXP, SEXP b_sigmaSEXP, SEXP lambdaSEXP, SEXP a_thetaSEXP, SEXP b_thetaSEXP, SEXP launch_iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type old_assign(old_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type old_alpha(old_alphaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi(xiSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu0(mu0SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type a_sigma(a_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type b_sigma(b_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type a_theta(a_thetaSEXP);
    Rcpp::traits::input_parameter< double >::type b_theta(b_thetaSEXP);
    Rcpp::traits::input_parameter< int >::type launch_iter(launch_iterSEXP);
    rcpp_result_gen = Rcpp::wrap(SFDM_SM(K, old_assign, old_alpha, xi, y, mu0, a_sigma, b_sigma, lambda, a_theta, b_theta, launch_iter));
    return rcpp_result_gen;
END_RCPP
}
// SFDM_model
Rcpp::List SFDM_model(int iter, int K, arma::vec init_assign, arma::vec xi, arma::vec y, arma::vec mu0, arma::vec a_sigma, arma::vec b_sigma, arma::vec lambda, double a_theta, double b_theta, int sm_iter, int print_iter);
RcppExport SEXP _ClusterNormal_SFDM_model(SEXP iterSEXP, SEXP KSEXP, SEXP init_assignSEXP, SEXP xiSEXP, SEXP ySEXP, SEXP mu0SEXP, SEXP a_sigmaSEXP, SEXP b_sigmaSEXP, SEXP lambdaSEXP, SEXP a_thetaSEXP, SEXP b_thetaSEXP, SEXP sm_iterSEXP, SEXP print_iterSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type iter(iterSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type init_assign(init_assignSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type xi(xiSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type mu0(mu0SEXP);
    Rcpp::traits::input_parameter< arma::vec >::type a_sigma(a_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type b_sigma(b_sigmaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< double >::type a_theta(a_thetaSEXP);
    Rcpp::traits::input_parameter< double >::type b_theta(b_thetaSEXP);
    Rcpp::traits::input_parameter< int >::type sm_iter(sm_iterSEXP);
    Rcpp::traits::input_parameter< int >::type print_iter(print_iterSEXP);
    rcpp_result_gen = Rcpp::wrap(SFDM_model(iter, K, init_assign, xi, y, mu0, a_sigma, b_sigma, lambda, a_theta, b_theta, sm_iter, print_iter));
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
    {"_ClusterNormal_fmm_log_alloc_prob", (DL_FUNC) &_ClusterNormal_fmm_log_alloc_prob, 9},
    {"_ClusterNormal_fmm_samp_new", (DL_FUNC) &_ClusterNormal_fmm_samp_new, 2},
    {"_ClusterNormal_fmm_mod", (DL_FUNC) &_ClusterNormal_fmm_mod, 9},
    {"_ClusterNormal_log_alloc_prob", (DL_FUNC) &_ClusterNormal_log_alloc_prob, 10},
    {"_ClusterNormal_samp_new", (DL_FUNC) &_ClusterNormal_samp_new, 1},
    {"_ClusterNormal_log_marginal", (DL_FUNC) &_ClusterNormal_log_marginal, 6},
    {"_ClusterNormal_log_likelihood", (DL_FUNC) &_ClusterNormal_log_likelihood, 6},
    {"_ClusterNormal_log_cluster_param", (DL_FUNC) &_ClusterNormal_log_cluster_param, 2},
    {"_ClusterNormal_log_gamma_cluster", (DL_FUNC) &_ClusterNormal_log_gamma_cluster, 3},
    {"_ClusterNormal_log_proposal", (DL_FUNC) &_ClusterNormal_log_proposal, 10},
    {"_ClusterNormal_log_prior", (DL_FUNC) &_ClusterNormal_log_prior, 2},
    {"_ClusterNormal_SFDM_allocate", (DL_FUNC) &_ClusterNormal_SFDM_allocate, 9},
    {"_ClusterNormal_SFDM_SM", (DL_FUNC) &_ClusterNormal_SFDM_SM, 12},
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
