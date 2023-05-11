### Required Library: ----------------------------------------------------------
library(Rcpp)
library(RcppArmadillo)
library(devtools)
library(LaplacesDemon)
library(mvtnorm)
library(tidyverse)
library(DirichletReg)
library(salso)
library(rootSolve)
library(metRology)

### Required Commands for build the packages: ----------------------------------
uninstall()
compileAttributes()
build()
install()
library(ClusterNormal)
