#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;

// [[Rcpp::export]]
arma::mat rwishart(unsigned int df, const arma::mat& S){
  // Dimension of returned wishart
  unsigned int m = S.n_rows;
  
  // Z composition: sqrt chisqs on diagonal - random normals below diagonal - misc above diagonal
  arma::mat Z(m,m);
  
  // Fill the diagonal
  for(unsigned int i = 0; i < m; i++){
    Z(i,i) = sqrt(R::rchisq(df-i));
  }
  
  // Fill the lower matrix with random guesses
  for(unsigned int j = 0; j < m; j++){  
    for(unsigned int i = j+1; i < m; i++){    
      Z(i,j) = R::rnorm(0,1);
    }
  }
  
  // Lower triangle * chol decomp
  arma::mat C = arma::trimatl(Z).t() * arma::chol(S);
  
  // Return random wishart
  return C.t()*C;
}

// [[Rcpp::export]]
Rcpp::List gibbsC(arma::mat theta,  arma::mat S_0, arma::mat data, int nsim, 
                 arma::vec mu_0, arma::vec x_bar_n) {
  
  // Obtain environment containing function
  Rcpp::Environment package_env("package:mvtnorm");
  // Make function callable from C++
  Rcpp::Function rmvnorm = package_env["rmvnorm"];
  
  // Initialize object I will need
  arma::mat S_theta, S_n, Sigma, Lambda_0_solved, Sigma_solved, Lambda_n;
  arma::vec mu_n;
  NumericVector theta2;
  int n = data.n_rows;
  int p = data.n_cols;
  int degof = n + p + 2;
  // Initialize output matrices
  NumericMatrix GS_Sigma(nsim, p*(p+1)*0.5);
  GS_Sigma(0,_) = NumericVector::create(S_0(0,0), S_0(0,1), S_0(0,2), S_0(1,1), S_0(1,2), S_0(2,2));
  NumericMatrix GS_theta(nsim, p);
  GS_theta(0,_) =  NumericVector::create(theta(0,0), theta(0,1), theta(0,2));
  
  // Loop over -> simulations!
  for(int i = 1; i < nsim; i++){
    // sample Sigma from the current full-conditional
    S_theta = (data - theta).t() * (data - theta);
    S_n = S_0 + S_theta;
    Sigma = arma::inv(rwishart(degof, inv(S_n)));
    // compute the new parameters of the full conditionals
    Lambda_0_solved = arma::inv(S_0);
    Sigma_solved = arma::inv(Sigma);
    Lambda_n = arma::inv(Lambda_0_solved  + 182 * Sigma_solved);
    mu_n = Lambda_n * (Lambda_0_solved * mu_0 + 182 * Sigma_solved * x_bar_n);
    theta2 = rmvnorm(1, mu_n, Lambda_n);
    
    GS_Sigma(i,_) = NumericVector::create(Sigma(0,0), Sigma(0,1), Sigma(0,2), Sigma(1,1), Sigma(1,2), Sigma(2,2));
    GS_theta(i,_) =  NumericVector::create(theta2[0], theta2[1], theta2[2]);
    
    for(int j = 0; j<182; j++){
      theta.row(j) = as<arma::rowvec>(theta2);
    }
  } 
  
  return Rcpp::List::create(GS_Sigma, GS_theta);
  
}
