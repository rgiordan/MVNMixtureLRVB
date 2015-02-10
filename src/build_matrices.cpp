#include <RcppEigen.h>
#include <Rcpp.h>
#include <math.h>

// [[Rcpp::depends(RcppEigen)]]
using Eigen::Map;                 // 'maps' rather than copies 
using Eigen::MatrixXd;                  // variable size matrix, double precision
using Eigen::VectorXd;                  // variable size vector, double precision
using Eigen::SparseMatrix;              // sparse matrix
using Eigen::SparseLU;
using Eigen::HouseholderQR;
typedef Eigen::MappedSparseMatrix<double> MappedSpMat;
typedef Eigen::Map<MatrixXd> MappedMat;
typedef Eigen::Map<VectorXd> MappedVec;
using Eigen::HouseholderQR;
typedef Eigen::Triplet<double> Triplet; // For populating sparse matrices


void PrintTriplet(std::vector<Triplet> t) {
  // Print a triplet object for debugging.
  Rcpp::Rcout << "Printing triplet.\n";
  std::vector<Triplet>::const_iterator i;
  for (i = t.begin(); i != t.end(); ++i) {
    Rcpp::Rcout << i->col() << "," << i->row() << "," << i->value() << "\n";
  }
  Rcpp::Rcout << "Done printing triplet.\n";
}


void PrintNumericMatrix(Rcpp::NumericMatrix x) {
  Rcpp::Rcout << "\n";
  for (int i = 0; i < x.rows(); i++) {
    for (int j = 0; j < x.cols(); j++) {
      Rcpp::Rcout << x(i, j) << ", ";
    }
    Rcpp::Rcout << "\n";
  }
  Rcpp::Rcout << "\n";
}

///////////////////////////////////////////
// Indexing functions
///////////////////////////////////////////

// [[Rcpp::export]]
int GetMatrixSizeFromUTSize(int ut_size) {
  // Get the size of a square matrix from the length
  // of a vector containing its upper triangluar section.
  int n = (sqrt(1 + 8 * (double)ut_size) - 1) / 2;
  if (ut_size != n * (n + 1) / 2) {
    Rcpp::Rcout << "The size of the vector is not a triangular number.\n";
    return -1;
  }
  return n;
}

// [[Rcpp::export]]
int GetUpperTriangularIndex(int p1, int p2) {
  // Get the linear index of an element indexing the upper
  // triangular part of a matrix.  It is expected that p1 <= p2.
  // Args:
  //   - p1: the row of the element
  //   - p2: the column of the element
  // Returns:
  //   The 0-based linear index of the element.

  if (p1 > p2) {
    //Rcpp::Rcout << "Error: Bad indices for GetUpperTriangularIndex\n";
    //return 0;
    // Switch the indices if p1 > p2 under the assumption that the matrix
    // is symmetric.
    int p1_temp = p1;
    p1 = p2;
    p2 = p1_temp;
  }

  // There are p2 * (p2 + 1) / 2 elements preceding the
  //  p2-th column.

  return (p2 * (p2 + 1)) / 2 + p1;
}

// [[Rcpp::export]]
VectorXd ConvertSymmetricMatrixToVector(const MatrixXd x) {
  int n = x.rows();
  VectorXd x_vec((n * (n + 1)) / 2);
  if (n != x.cols()) {
    Rcpp::Rcout << "x is not square.\n";
    return x_vec;
  }
  for (int j = 0; j < n; j++) {
    for (int i = 0; i <= j; i++) {
      x_vec(GetUpperTriangularIndex(i, j)) = x(i, j);
    }
  }
  return x_vec;
}

// [[Rcpp::export]]
MatrixXd ConvertVectorToSymmetricMatrix(const VectorXd x_vec) {
  int vec_n = x_vec.size();
  int n = GetMatrixSizeFromUTSize(vec_n);
  MatrixXd x(n, n);
  if (n == -1) {
    Rcpp::Rcout << "The size of x_vec is not a triangular number.\n";
    return x;
  }

  for (int j = 0; j < n; j++) {
    for (int i = 0; i <= j; i++) {
      x(i, j) = x_vec(GetUpperTriangularIndex(i, j));
      x(j, i) = x(i, j);
    }
  }
  return x;
}


// // [[Rcpp::export]]
// Rcpp::CharacterVector GetSymmetricMatrixVectorNames(int p) {
//   // Get the column names of the terms of a symmetric matrix that has been vectorized.
//   //
//   // Args:
//   //   - p: The dimension of the symmetric matrix.
//   //
//   // Returns:
//   //   The column names of the symmetric matrix after ConvertSymmetricMatrixToVector
//   //   has been applied to it.

//   // This is a little inefficient but robust to changes in 
//   // ConvertSymmetricMatrixToVector, and p should always be small.
//   MatrixXd row_mat(p, p);
//   MatrixXd col_mat(p, p);
//   for (int a = 0; a < p; a++) {
//     for (int b =0; b < p; b++) {
//       row_mat(a, b) = a;
//       col_mat(a, b) = b;
//     }
//   }
//   VectorXd row_vec = ConvertSymmetricMatrixToVector(row_mat);
//   VectorXd col_vec = ConvertSymmetricMatrixToVector(col_mat);

//   Rcpp::CharacterVector result(row_vec.size());
//   for (int a = 0; a < row_vec.size(); a++) {
//     // I don't know how to do this.
//     result(a) = row_vec(a) + "_" + col_vec(a);
//   }
//   return result;
// }


// This class provides linear indices for each
// parameter in the full parameter matrix, ordered by
// mu, pi, lambda.
// n_tot is the number of data points, k_tot the number of components,
// and p_tot is the dimension of each observation.
class FullParameterIndices {
  int k_tot, p_tot;
  int mu_offset, mu2_offset, log_pi_offset;
  int lambda_offset, log_det_lambda_offset;
  int matrix_size, dim;
  int mu_size, mu2_size, log_pi_size;
  int lambda_size, log_det_lambda_size;
  public:
  FullParameterIndices(int, int);
  int MuCoord(int, int);
  int Mu2Coord(int, int, int);
  int MuOnlyMuCoord(int, int);
  int MuOnlyMu2Coord(int, int, int);
  int LogPiCoord(int);
  int LambdaCoord(int, int, int);
  int LogDetLambdaCoord(int);
  int LambdaOnlyLambdaCoord(int, int, int);
  int LambdaOnlyLogDetLambdaCoord(int);
  int Dim();
  int MatrixSize();
  int FullMuSize();
  int FullLambdaSize();
  VectorXd MuIndices();
  VectorXd LogPiIndices();
  VectorXd LambdaIndices();
};


FullParameterIndices::FullParameterIndices(int k, int p) {
  k_tot = k;
  p_tot = p;

  // The number of elements needed to store a symmetric matrix.
  matrix_size = (p_tot * (p_tot + 1)) / 2;

  // Set the beginning points for each block, and then the full
  // size of the matrix.
  mu_size = k_tot * p_tot;
  mu2_size = k_tot * matrix_size;
  log_pi_size = k_tot;
  lambda_size = k_tot * matrix_size;
  log_det_lambda_size = k_tot;

  mu_offset = 0;
  mu2_offset = mu_offset + mu_size;
  log_pi_offset = mu2_offset + mu2_size;
  lambda_offset = log_pi_offset + log_pi_size;
  log_det_lambda_offset = lambda_offset + lambda_size;
  dim = log_det_lambda_offset + log_det_lambda_size;
}

int FullParameterIndices::MuCoord(int k, int p) {
  // Get the index of a mu parameter.  The inputs are return are 0-indexed.
  return mu_offset + k * p_tot + p;
}

int FullParameterIndices::Mu2Coord(int k, int p1, int p2) {
  // Get the index of a mu2 parameter.  The inputs are return are 0-indexed,
  // and p1 is expected to be less or equal to p2, i.e. this is the index
  // of the upper triangular part of the matrix, including the diagonal.
  // The elements are stored in column-major order.
  return mu2_offset + k * matrix_size + GetUpperTriangularIndex(p1, p2);
}

int FullParameterIndices::MuOnlyMuCoord(int k, int p) {
  // Get the index of a mu parameter for a mu-only matrix.
  return MuCoord(k, p) - mu_offset;
}

int FullParameterIndices::MuOnlyMu2Coord(int k, int p1, int p2) {
  // Get the index of a mu2 parameter for a mu-only matrix.
  return Mu2Coord(k, p1, p2) - mu_offset;
}

int FullParameterIndices::LogPiCoord(int k) {
  // Get the index of a log pi parameter.  The inputs are return are 0-indexed.
  return log_pi_offset + k;
}

int FullParameterIndices::LambdaCoord(int k, int p1, int p2) {
  // Get the index of a lambda parameter.  The inputs are return are 0-indexed,
  // and p1 is expected to be less or equal to p2, i.e. this is the index
  // of the upper triangular part of the matrix, including the diagonal.
  // The elements are stored in column-major order.
  return lambda_offset + k * matrix_size + GetUpperTriangularIndex(p1, p2);
}

int FullParameterIndices::LogDetLambdaCoord(int k) {
  // Get the index of a log det lambda parameter.  The inputs are return are 0-indexed.
  return log_det_lambda_offset + k;
}

int FullParameterIndices::LambdaOnlyLambdaCoord(int k, int p1, int p2) {
  // Get the index of a lambda parameter.  The inputs are return are 0-indexed,
  // and p1 is expected to be less or equal to p2, i.e. this is the index
  // of the upper triangular part of the matrix, including the diagonal.
  // The elements are stored in column-major order.
  return LambdaCoord(k, p1, p2) - lambda_offset;
}

int FullParameterIndices::LambdaOnlyLogDetLambdaCoord(int k) {
  // Get the index of a log det lambda parameter.  The inputs are return are 0-indexed.
  return LogDetLambdaCoord(k) - lambda_offset;
}

int FullParameterIndices::Dim() {
  return dim;
}

int FullParameterIndices::MatrixSize() {
  return matrix_size;
}

int FullParameterIndices::FullMuSize() {
  return mu_size + mu2_size;
}

int FullParameterIndices::FullLambdaSize() {
  return lambda_size + log_det_lambda_size;
}

VectorXd FullParameterIndices::MuIndices() {
  // Return, for R, a 1-based set of indices for the mu and mu2 components.
  VectorXd mu_indices(FullMuSize());
  for (int i = 0; i < FullMuSize(); i++) {
    mu_indices(i) = mu_offset + i + 1;
  }
  return mu_indices;
}

VectorXd FullParameterIndices::LambdaIndices() {
  // Return, for R, a 1-based set of indices for the
  // lambda and log det lambda components.
  VectorXd lambda_indices(FullLambdaSize());
  for (int i = 0; i < FullLambdaSize(); i++) {
    lambda_indices(i) = lambda_offset + i + 1;
  }
  return lambda_indices;
}

VectorXd FullParameterIndices::LogPiIndices() {
  // Return, for R, a 1-based set of indices for the
  // lambda and log det lambda components.
  VectorXd log_pi_indices(FullLambdaSize());
  for (int i = 0; i < k_tot; i++) {
    log_pi_indices(i) = log_pi_offset + i;
  }
  return log_pi_indices;
}



// This class provides linear indices for each
// parameter in the z parameter matrix, ordered by
// n_tot is the number of data points, k_tot the number of components,
// and p_tot is the dimension of each observation.
class ZParameterIndices {
  int k_tot, n_tot, dim;
  public:
  ZParameterIndices(int, int);
  int ZCoord(int, int);
  int Dim();
};

ZParameterIndices::ZParameterIndices(int n, int k) {
  k_tot = k;
  n_tot = n;
  dim = k_tot * n_tot;
}

int ZParameterIndices::ZCoord(int n, int k) {
  return k_tot * n + k;
}

int ZParameterIndices::Dim() {
  return dim;
}


// This class provides linear indices for each
// parameter in the x parameter matrix
class XParameterIndices {
  int p_tot, n_tot, matrix_size, dim;
  int x_size, x2_size;
  int x_offset, x2_offset;
  public:
  XParameterIndices(int, int);
  int XCoord(int, int);
  int X2Coord(int, int, int);
  int MatrixSize();
  int Dim();
};

XParameterIndices::XParameterIndices(int n, int p) {
  p_tot = p;
  n_tot = n;
  matrix_size = (p * (p + 1)) / 2;
  x_size = n_tot * p_tot;
  x2_size = n_tot * matrix_size;
  x_offset = 0;
  x2_offset = x_offset + x_size;
  dim = x2_offset + x2_size;
}

int XParameterIndices::XCoord(int n, int p) {
  return x_offset + n * p_tot + p;
}

int XParameterIndices::X2Coord(int n, int p1, int p2) {
  return x2_offset + n * matrix_size + GetUpperTriangularIndex(p1, p2);
}

int XParameterIndices::Dim() {
  return dim;
}

int XParameterIndices::MatrixSize() {
  return matrix_size;
}

// [[Rcpp::export]]
int GetXCoordinate(int n, int p, int n_tot, int p_tot) {
  // Get the 0-indexed linear coordinate of an x term.
  XParameterIndices x_ind(n_tot, p_tot);
  return x_ind.XCoord(n, p);
}


// This class parameterizes the parameters of an MLE
// into a single vector that can be passed to numerical
// optimization routines.
class MLEParameterIndices {
  int k_tot, p_tot, matrix_size, dim;
  int mu_size, pi_size, lambda_size;
  int mu_offset, lambda_offset, pi_offset;
  public:
  MLEParameterIndices(int, int);
  int MuCoord(int, int);
  int LambdaCoord(int, int, int);
  int PiCoord(int);
  int MatrixSize();
  int Dim();
};

MLEParameterIndices::MLEParameterIndices(int k, int p) {
  k_tot = k;
  p_tot = p;

  matrix_size = (p_tot * (p_tot + 1)) / 2;

  mu_size = k_tot * p_tot;

  // For the MLE, one pi parameter is redundant.
  pi_size = k_tot - 1;
  lambda_size = k_tot * matrix_size;

  mu_offset = 0;
  pi_offset = mu_offset + mu_size;
  lambda_offset = pi_offset + pi_size;
  dim = lambda_offset + lambda_size;
}

int MLEParameterIndices::MuCoord(int k, int p) {
  // Get the index of a mu parameter.  The inputs are return are 0-indexed.
  return mu_offset + k * p_tot + p;
}

int MLEParameterIndices::PiCoord(int k) {
  // Get the index of a pi parameter.  The inputs are return are 0-indexed.
  return pi_offset + k;
}

int MLEParameterIndices::LambdaCoord(int k, int p1, int p2) {
  // Get the index of a lambda parameter.  The inputs are return are 0-indexed,
  // and p1 is expected to be less or equal to p2, i.e. this is the index
  // of the upper triangular part of the matrix, including the diagonal.
  // The elements are stored in column-major order.
  return lambda_offset + k * matrix_size + GetUpperTriangularIndex(p1, p2);
}


int MLEParameterIndices::MatrixSize() {
  return matrix_size;
}

int MLEParameterIndices::Dim() {
  return dim;
}


// [[Rcpp::export]]
VectorXd LinearlyPackParameters(const MatrixXd mu,
				const MatrixXd lambda,
				const VectorXd pi) {
  // Pack and transform the parameters into a single vector that can be
  // passed to unconstrained numerical optimization routines.
  //
  // Args:
  //  - mu: A p by k matrix of means.
  //  - lambda: A (p + 1) * p / 2 by k matrix of precision parameters.
  //  - pi: A k-length vector of component weights which sum to one.

  int k_tot = mu.cols();
  int p_tot = mu.rows();

  MLEParameterIndices ind(k_tot, p_tot);
  VectorXd par(ind.Dim());

  // Pack the unconstrained mu parameters.
  for (int k = 0; k < k_tot; k++) {
    for (int p = 0; p < p_tot; p++) {
      par(ind.MuCoord(k, p)) = mu(p, k);
    }
  }

  // I will use the log cholesky representation of the precision matrix
  // following this paper:
  // Unconstrained Parameterizations for Variance-Covariance Matrices
  // by Pinheiro, Bates
  for (int k = 0; k < k_tot; k++) {
    MatrixXd this_lambda =
      ConvertVectorToSymmetricMatrix(lambda.block(0, k, ind.MatrixSize(), 1));
    Eigen::LLT<MatrixXd> lambda_chol(this_lambda);
    MatrixXd lambda_l = lambda_chol.matrixL(); // retrieve factor L in the decomposition
    for (int a = 0; a < p_tot; a++) {
      for (int b = 0; b <= a; b++) {
	if (a == b) {
	  par(ind.LambdaCoord(k, a, b)) = log(lambda_l(a, b));
	} else {
	  par(ind.LambdaCoord(k, a, b)) = lambda_l(a, b);
	}
      }
    }
  }

  // Store the pi in the non-redundant form log(pi(k) / pi(0))
  // for k > 0.
  double log_pi_0 = log(pi(0));
  for (int k = 0; k < k_tot - 1; k++) {
    par(ind.PiCoord(k)) = log(pi(k + 1)) - log_pi_0;
  }

  return par;
}

// [[Rcpp::export]]
Rcpp::List LinearlyUnpackParameters(const VectorXd par, int k_tot, int p_tot) {
  // Unpack the parameter vector created by LinearlyPackParameters
  // into an R list.
  //
  // Args:
  //  - par: The parameter vector created by LinearlyPackParameters.
  //  - k_tot: The number of components.
  //  - p_tot: The size of the mu vector.

  MLEParameterIndices ind(k_tot, p_tot);
  if (par.size() != ind.Dim()) {
    Rcpp::Rcout << "Size of par inconsistent with k_tot and p_tot.\n ";\
    return Rcpp::List::create(Rcpp::Named("empty") = 0.0);
  }

  MatrixXd mu(p_tot, k_tot);
  MatrixXd lambda(ind.MatrixSize(), k_tot);
  VectorXd pi(k_tot);

  // Unpack mu.
  for (int k = 0; k < k_tot; k++) {
    for (int p = 0; p < p_tot; p++) {
      mu(p, k) = par(ind.MuCoord(k, p));
    }
  }

  // Unpack lambda, which is stored in log Cholesky form.
  for (int k = 0; k < k_tot; k++) {
    MatrixXd lambda_l(p_tot, p_tot);
    lambda_l.setZero();
    for (int a = 0; a < p_tot; a++) {
      for (int b = 0; b <= a; b++) {
	if (a == b) {
	  lambda_l(a, b) = exp(par(ind.LambdaCoord(k, a, b)));
	} else {
	  lambda_l(a, b) = par(ind.LambdaCoord(k, a, b));
	}
      }
    }
    MatrixXd this_lambda = lambda_l * lambda_l.transpose();
    lambda.block(0, k, ind.MatrixSize(), 1) =
      ConvertSymmetricMatrixToVector(this_lambda);
  }

  // Unpack pi.  Dig that
  // pi0 + pi1 + ... pik = 1 =>
  // 1 + pi1 / pi0 + ... + pik / pi0 = 1 / pi0 =>
  // pi0 = (1 + pi1 / pi0 + ... + pik / pi0) ^ (-1)
  double pi_ratio_tot = 1.0;
  for (int k = 0; k < k_tot - 1; k++) {
    pi_ratio_tot += exp(par(ind.PiCoord(k)));
  }
  pi(0) = 1 / pi_ratio_tot;
  for (int k = 0; k < k_tot - 1; k++) {
    pi(k + 1) = pi(0) * exp(par(ind.PiCoord(k)));
  }

  return Rcpp::List::create(Rcpp::Named("mu") = mu,
                            Rcpp::Named("lambda") = lambda,
			    Rcpp::Named("pi") = pi);

}

// [[Rcpp::export]]
VectorXd PackVBParameters(const MatrixXd e_z,
			       const MatrixXd e_mu,
			       const MatrixXd e_mu2,
			       const MatrixXd e_lambda,
			       const VectorXd e_log_det_lambda,
			       const VectorXd e_log_pi) {
  // Make a single vector containing all the variational 
  // parameters.  (The other LinearizeParameters function
  // converts them to a form that can be used for unconstrained
  // optimization.)

  int n_tot = e_z.rows();
  int p_tot = e_mu.rows();
  int k_tot = e_mu.cols();

  // I'm not going to check the sizes since this will really only
  // be used for testing.

  FullParameterIndices ind(k_tot, p_tot);
  ZParameterIndices z_ind(n_tot, k_tot);
  
  // This will contain theta first, then z.
  VectorXd par(ind.Dim() + z_ind.Dim());

  // First the theta values
  for (int k = 0; k < k_tot; k++) {
    par(ind.LogDetLambdaCoord(k)) = e_log_det_lambda(k);
    par(ind.LogPiCoord(k)) = e_log_pi(k);
    for (int a = 0; a < p_tot; a++) {
      par(ind.MuCoord(k, a)) = e_mu(a, k);
      for (int b = 0; b <= a; b++) {
	int ut_index = GetUpperTriangularIndex(b, a);
	par(ind.Mu2Coord(k, a, b)) = e_mu2(ut_index, k);
	par(ind.LambdaCoord(k, a, b)) = e_lambda(ut_index, k);
      }
    }
  }

  // Next the z values, which start at the end of the theta values.
  for (int n = 0; n < n_tot; n++) {
    for (int k = 0; k < k_tot; k++) {
      par(z_ind.ZCoord(n, k) + ind.Dim()) = e_z(n, k); 
    }
  }
  return par;
}


// [[Rcpp::export]]
Rcpp::List UnpackVBParameters(const VectorXd par,
			      const int n_tot,
			      const int p_tot,
			      const int k_tot) {
  // Convert a single vector containing all the VB parameterse
  // into a list with each parameter contained separately.

  FullParameterIndices ind(k_tot, p_tot);
  ZParameterIndices z_ind(n_tot, k_tot);

  if (par.size() != ind.Dim() + z_ind.Dim()) {
    Rcpp::Rcout << "Bad size for par input.\n";
    return Rcpp::List::create(Rcpp::Named("empty") = 0.0);
  }
  
  MatrixXd e_mu(p_tot, k_tot);
  MatrixXd e_mu2(ind.MatrixSize(), k_tot);
  MatrixXd e_lambda(ind.MatrixSize(), k_tot);
  VectorXd e_log_det_lambda(k_tot);
  VectorXd e_log_pi(k_tot);
  MatrixXd e_z(n_tot, k_tot);

  // First the theta values
  for (int k = 0; k < k_tot; k++) {
    e_log_det_lambda(k) = par(ind.LogDetLambdaCoord(k));
     e_log_pi(k) = par(ind.LogPiCoord(k));
    for (int a = 0; a < p_tot; a++) {
      e_mu(a, k) = par(ind.MuCoord(k, a));
      for (int b = 0; b <= a; b++) {
	int ut_index = GetUpperTriangularIndex(b, a);
	e_mu2(ut_index, k) = par(ind.Mu2Coord(k, a, b));
	e_lambda(ut_index, k) = par(ind.LambdaCoord(k, a, b));
      }
    }
  }

  // Next the z values, which start at the end of the theta values.
  for (int n = 0; n < n_tot; n++) {
    for (int k = 0; k < k_tot; k++) {
      e_z(n, k) = par(z_ind.ZCoord(n, k) + ind.Dim()); 
    }
  }

  return Rcpp::List::create(Rcpp::Named("e_mu") = e_mu,
			    Rcpp::Named("e_mu2") = e_mu2,
                            Rcpp::Named("e_lambda") = e_lambda,
                            Rcpp::Named("e_log_det_lambda") = e_log_det_lambda,
			    Rcpp::Named("e_log_pi") = e_log_pi,
			    Rcpp::Named("e_z") = e_z);
}


// [[Rcpp::export]]
Rcpp::List UnpackVBThetaParameters(const VectorXd par,
				   const int n_tot,
				   const int p_tot,
				   const int k_tot) {
  // Convert a single vector containing the VB theta parameters
  // into a list with each parameter contained separately.

  // TODO: combine this with UnpackVBParameters in order to
  // not duplicate code.

  FullParameterIndices ind(k_tot, p_tot);

  if (par.size() != ind.Dim()) {
    Rcpp::Rcout << "Bad size for par input.\n";
    return Rcpp::List::create(Rcpp::Named("empty") = 0.0);
  }
  
  MatrixXd e_mu(p_tot, k_tot);
  MatrixXd e_mu2(ind.MatrixSize(), k_tot);
  MatrixXd e_lambda(ind.MatrixSize(), k_tot);
  VectorXd e_log_det_lambda(k_tot);
  VectorXd e_log_pi(k_tot);

  for (int k = 0; k < k_tot; k++) {
    e_log_det_lambda(k) = par(ind.LogDetLambdaCoord(k));
     e_log_pi(k) = par(ind.LogPiCoord(k));
    for (int a = 0; a < p_tot; a++) {
      e_mu(a, k) = par(ind.MuCoord(k, a));
      for (int b = 0; b <= a; b++) {
	int ut_index = GetUpperTriangularIndex(b, a);
	e_mu2(ut_index, k) = par(ind.Mu2Coord(k, a, b));
	e_lambda(ut_index, k) = par(ind.LambdaCoord(k, a, b));
      }
    }
  }

  return Rcpp::List::create(Rcpp::Named("e_mu") = e_mu,
			    Rcpp::Named("e_mu2") = e_mu2,
                            Rcpp::Named("e_lambda") = e_lambda,
                            Rcpp::Named("e_log_det_lambda") = e_log_det_lambda,
			    Rcpp::Named("e_log_pi") = e_log_pi);
}

// [[Rcpp::export]]
Rcpp::List GetCoreVBThetaIndices( const int n_tot,
                                   const int p_tot,
                                   const int k_tot) {
  // Get a list of the indices for the core VB theta parameters:
  // mu, lambda, and log(pi).

  FullParameterIndices ind(k_tot, p_tot);

  MatrixXd mu_indices(p_tot, k_tot);
  VectorXd log_pi_indices(k_tot);
  MatrixXd lambda_indices(ind.MatrixSize(), k_tot);

  for (int k = 0; k < k_tot; k++) {
    log_pi_indices(k) = ind.LogPiCoord(k);
    for (int a = 0; a < p_tot; a++) {
      mu_indices(a, k) = ind.MuCoord(k, a);
      for (int b = 0; b <= a; b++) {
        int ut_index = GetUpperTriangularIndex(b, a);
        lambda_indices(ut_index, k) = ind.LambdaCoord(k, a, b);
      }
    }
  }

  return Rcpp::List::create(Rcpp::Named("mu_indices") = mu_indices,
                            Rcpp::Named("lambda_indices") = lambda_indices,
                            Rcpp::Named("log_pi_indices") = log_pi_indices);
}


// [[Rcpp::export]]
VectorXd PackXParameters(const MatrixXd x) {
  // Make a single vector containing all the x parameters
  // in the same order as the sensitivity matrices.

  int n_tot = x.rows();
  int p_tot = x.cols();

  XParameterIndices x_ind(n_tot, p_tot);

  VectorXd par(x_ind.Dim());

  for (int n = 0; n < n_tot; n++)  {
    for (int a = 0; a < p_tot; a++) {
      par(x_ind.XCoord(n, a)) = x(n, a);
      for (int b = 0; b < p_tot; b++) {
	par(x_ind.X2Coord(n, a, b)) = x(n, a) * x(n, b);
      }
    }
  }
  return par;
}

// [[Rcpp::export]]
Rcpp::List UnpackXParameters(const VectorXd par,
			     const int n_tot,
			     const int p_tot) {
  // Reverse PackXParameters.
  XParameterIndices x_ind(n_tot, p_tot);

  if (par.size() != x_ind.Dim()) {
    Rcpp::Rcout << "Bad size for par input.\n";
    return Rcpp::List::create(Rcpp::Named("empty") = 0.0);
  }

  MatrixXd x(n_tot, p_tot);
  MatrixXd x2(n_tot, x_ind.MatrixSize());

  for (int n = 0; n < n_tot; n++)  {
    for (int a = 0; a < p_tot; a++) {
      x(n, a) = par(x_ind.XCoord(n, a));
      for (int b = 0; b <= a; b++) {
	x2(n, GetUpperTriangularIndex(a, b)) = par(x_ind.X2Coord(n, a, b));
      }
    }
  }

    return Rcpp::List::create(Rcpp::Named("x") = x,
			      Rcpp::Named("x2") = x2);
}


// [[Rcpp::export]]
MatrixXd InvertLinearizedMatrices(const MatrixXd par_mat) {
  // Invert each row of a matrix of vectorized UT matrices
  // and re-pack the result as a matrix of vectorized UT matrices.

  MatrixXd par_mat_inv(par_mat.rows(), par_mat.cols());
  int p_tot = GetMatrixSizeFromUTSize(par_mat.rows());
  int k_tot = par_mat.cols();
  if (p_tot == -1) {
    Rcpp::Rcout << "InvertLinearizedMatrices: Bad matrix size.\n";
    return par_mat_inv;
  }
  for (int k = 0; k < k_tot; k++) {
    MatrixXd mat(p_tot, p_tot);
    for (int a = 0; a < p_tot; a++) {
      for (int b = 0; b < p_tot; b++) {
	mat(a, b) = par_mat(GetUpperTriangularIndex(a, b), k);
      }
    }
    MatrixXd mat_inv = mat.inverse();
    for (int a = 0; a < p_tot; a++) {
      for (int b = 0; b < p_tot; b++) {
	par_mat_inv(GetUpperTriangularIndex(a, b), k) = mat_inv(a, b);
      }
    }
  } // k loop
  return par_mat_inv;
}


// [[Rcpp::export]]
VectorXd LogDeterminantOfLinearizedMatrices(const MatrixXd par_mat) {
  // Invert each row of a matrix of vectorized UT matrices
  // and re-pack the result as a matrix of vectorized UT matrices.

  int p_tot = GetMatrixSizeFromUTSize(par_mat.rows());
  int k_tot = par_mat.cols();
  VectorXd mat_det(k_tot);
  if (p_tot == -1) {
    Rcpp::Rcout << "InvertLinearizedMatrices: Bad matrix size.\n";
    return mat_det;
  }
  for (int k = 0; k < k_tot; k++) {
    MatrixXd mat(p_tot, p_tot);
    for (int a = 0; a < p_tot; a++) {
      for (int b = 0; b < p_tot; b++) {
	mat(a, b) = par_mat(GetUpperTriangularIndex(a, b), k);
      }
    }
    mat_det(k) = log(mat.determinant());
  }
  return mat_det;
}


// [[Rcpp::export]]
MatrixXd GetVectorizedOuterProductMatrix(const MatrixXd mat) {
  // Args:
  //   mat: A p by k matrix of vectors, m1 ... mk
  //
  // Returns:
  //   First, computes mi %*% t(mi) for each column,
  //   and then converts the result to a p * (p + 1) / 2 by  k
  //   matrix of vectorized upper triangular matrices.  

  int p_tot = mat.rows();
  int k_tot = mat.cols();

  MatrixXd result((p_tot * (p_tot + 1)) / 2, k_tot);
  for (int k = 0; k < k_tot; k++) {
    for (int a = 0; a < p_tot; a++) {
      for (int b = 0; b <= a; b++) {
	result(GetUpperTriangularIndex(a, b), k) = mat(a, k) * mat(b, k);
      }
    }
  }
  return result;
}



///////////////////////////////////////////
// Gamma functions
///////////////////////////////////////////


// [[Rcpp::export]]
double CppLgamma(double x) {
  // TODO: get gsl working in R so I don't have to do this.
  Rcpp::NumericVector x_v(1);
  x_v = x;
  x_v = lgamma(x_v);
  x = x_v(0);
  return x;
}


// [[Rcpp::export]]
double CppDigamma(double x) {
  // TODO: get gsl working in R so I don't have to do this.
  Rcpp::NumericVector x_v(1);
  x_v = x;
  x_v = digamma(x_v);
  x = x_v(0);
  return x;
}

// [[Rcpp::export]]
double CppTrigamma(double x) {
  // TODO: get gsl working in R so I don't have to do this.
  Rcpp::NumericVector x_v(1);
  x_v = x;
  x_v = trigamma(x_v);
  x = x_v(0);
  return x;
}


// [[Rcpp::export]]
double CppMultivariateLgamma(double x, int p) {
  double result = 0.0;
  for (int i = 1; i <= p; i++) {
    result += CppLgamma(x + (1 - (double)i) / 2);
  }
  return result;
}

// [[Rcpp::export]]
double CppMultivariateDigamma(double x, int p) {
  double result = 0.0;
  for (int i = 1; i <= p; i++) {
    result += CppDigamma(x + (1 - (double)i) / 2);
  }
  return result;
}

// [[Rcpp::export]]
double CppMultivariateTrigamma(double x, int p) {
 double result = 0.0;
  for (int i = 1; i <= p; i++) {
    result += CppTrigamma(x + (1 - (double)i) / 2);
  }
  return result;
}


///////////////////////////////////////////
// Likelihood functions
///////////////////////////////////////////


// [[Rcpp::export]]
double MVNLogLikelihoodPoint(const VectorXd x,
			     const VectorXd e_mu,
			     const VectorXd e_mu2,
			     const VectorXd e_lambda,
			     const double e_log_det_lambda,
			     const double e_log_pi) {
  // Get the log likelihood for a single observation from a single component.

  int p_tot = e_mu.size();
  int matrix_size = e_mu2.size();
  if (e_lambda.size() != matrix_size) {
    Rcpp::Rcout << "MVN Log likelihood: lambda_par has the wrong number of elements.\n";
    return 0.0;
  }
  if ((p_tot * (p_tot + 1)) / 2 != matrix_size) {
    Rcpp::Rcout << "MVN Log likelihood: matrix_size does not match p_tot.\n";
    return 0.0;
  }

  double log_lik = 0.0;
  for (int a = 0; a < p_tot; a++) {
    for (int b = 0; b < p_tot; b++) {
      int ab_index = GetUpperTriangularIndex(a, b);
      // Get the expectations of the wishart sufficient statistics.
      double e_lambda_ab = e_lambda(ab_index);
      log_lik += (-0.5 * e_lambda_ab *
		  (x(a) * x(b) -
		   e_mu(a) * x(b) - e_mu(b) * x(a) +
		   e_mu2(ab_index)));
    }
  }
  log_lik += e_log_pi;
  log_lik += 0.5 * e_log_det_lambda;
  return log_lik;
}


// [[Rcpp::export]]
double MVNLogLikelihoodPointWithX2(const VectorXd x,
				   const VectorXd x2,
				   const VectorXd e_mu,
				   const VectorXd e_mu2,
				   const VectorXd e_lambda,
				   const double e_log_det_lambda,
				   const double e_log_pi) {
  // Get the log likelihood for a single observation from a single component
  // where you can specify the x2 matrix.  This is currently only
  // used for unit testing the x sensitivity functions.
  //
  // Args:
  //   -- the usual plus
  //   - x2: A p * (p + 1) / 2 vector of the upper triangular part of x x^T

  int p_tot = e_mu.size();
  int matrix_size = e_mu2.size();
  if (e_lambda.size() != matrix_size) {
    Rcpp::Rcout << "MVN Log likelihood: lambda_par has the wrong number of elements.\n";
    return 0.0;
  }
  if ((p_tot * (p_tot + 1)) / 2 != matrix_size) {
    Rcpp::Rcout << "MVN Log likelihood: matrix_size does not match p_tot.\n";
    return 0.0;
  }

  double log_lik = 0.0;
  for (int a = 0; a < p_tot; a++) {
    for (int b = 0; b < p_tot; b++) {
      int ab_index = GetUpperTriangularIndex(a, b);
      // Get the expectations of the wishart sufficient statistics.
      double e_lambda_ab = e_lambda(ab_index);
      log_lik += (-0.5 * e_lambda_ab *
		  (x2(ab_index) -
		   e_mu(a) * x(b) - e_mu(b) * x(a) +
		   e_mu2(ab_index)));
    }
  }
  log_lik += e_log_pi;
  log_lik += 0.5 * e_log_det_lambda;
  return log_lik;
}


// [[Rcpp::export]]
VectorXd LogPriorMu(const MatrixXd e_mu,
		    const MatrixXd e_mu2,
		    const MatrixXd mu_prior_mean,
		    const MatrixXd mu_prior_info) {
  //  Return the log prior on mu up to a constant.
  //
  // Args:
  //  - mu_prior_mean: p by k vector of prior means.
  //  - mu_prior_info: (p + 1) * p / 2 by k vector of prior info.

  int p_tot = e_mu.rows();
  int k_tot = e_mu.cols();
  int matrix_size = (p_tot * (p_tot + 1)) / 2;

  VectorXd log_prior(k_tot);
  if (e_mu2.cols() != k_tot) {
    Rcpp::Rcout << "e_mu2 has the wrong number of columns.\n";
    return log_prior;
  }
  if (e_mu2.rows() != matrix_size) {
    Rcpp::Rcout << "e_mu2 has the wrong number of columns.\n";
    return log_prior;
  }
  if (mu_prior_mean.size() != p_tot) {
    Rcpp::Rcout << "mu_prior_mean is the wrong size.\n";
    return log_prior;
  }
  if (mu_prior_info.cols() != k_tot) {
    Rcpp::Rcout << "mu_prior_info has the wrong number of columns.\n";
    return log_prior;
  }
  if (mu_prior_info.rows() != matrix_size) {
    Rcpp::Rcout << "mu_prior_info has the wrong number of columns.\n";
    return log_prior;
  }

  MatrixXd mu_prior_log_det_info = LogDeterminantOfLinearizedMatrices(mu_prior_info);

  for (int k = 0; k < k_tot; k++) {
    // Since mu is MVN itself, we can use the MVN log likelihood function
    log_prior(k) = MVNLogLikelihoodPoint(mu_prior_mean.block(0, k, p_tot, 1),
					 e_mu.block(0, k, p_tot, 1),
					 e_mu2.block(0, k, matrix_size, 1),
					 mu_prior_info.block(0, k, matrix_size, 1),
					 mu_prior_log_det_info(k),
					 0);
  }

  return log_prior;
}


// [[Rcpp::export]]
VectorXd LogPriorLambda(const MatrixXd e_lambda,
			const VectorXd e_log_det_lambda,
			const MatrixXd lambda_prior_v_inv,
			const VectorXd lambda_prior_n) {
  // The logarithm of the Wishart prior up to a constant.
  //
  // Args:
  //   - lambda_prior_v_inv: (p + 1) * p / 2 by k prior value of the v inverse term.
  //   - lambda_prior_n: k - vector of prior n parameters.

  int k_tot = e_lambda.cols();
  int matrix_size = e_lambda.rows();
  int p_tot = GetMatrixSizeFromUTSize(matrix_size);
  
  VectorXd log_prior(k_tot);
  if (e_log_det_lambda.size() != k_tot) {
    Rcpp::Rcout << "e_log_det_lambda is the wrong size.\n";
    return log_prior;
  }
  if (lambda_prior_n.size() != k_tot) {
    Rcpp::Rcout << "lambda_prior_n is the wrong size.\n";
    return log_prior;
  }
  if (lambda_prior_v_inv.cols() != k_tot) {
    Rcpp::Rcout << "lambda_prior_v_inv has the wrong number of columns.\n";
    return log_prior;
  }
  if (lambda_prior_v_inv.rows() != matrix_size) {
    Rcpp::Rcout << "lambda_prior_v_inv has the wrong number of rows.\n";
    return log_prior;
  }
  
  for (int k = 0; k < k_tot; k++) {
    log_prior(k) = lambda_prior_n(k) * e_log_det_lambda(k) / 2.0;
    for (int a = 0; a < p_tot; a++) {
      for (int b = 0; b < p_tot; b++) {
	int this_index = GetUpperTriangularIndex(a, b);
	log_prior(k) -= 0.5 * e_lambda(this_index) * lambda_prior_v_inv(this_index);
      }
    }
  }
  return log_prior;
}


// [[Rcpp::export]]
VectorXd LogPriorPi(const VectorXd e_log_pi,
		    const VectorXd pi_prior_alpha) {
  // Returns the log pi prior up to a constant.
  //
  // Args:
  //   - pi_prior_alpha: Size k vector of pi prior parameters.

  int k_tot = e_log_pi.size();
  VectorXd log_prior(k_tot);
  if (pi_prior_alpha.size() != k_tot) {
    Rcpp::Rcout << "pi_prior_alpha has the wrong size.";
    return log_prior;
  }
  
  for (int k = 0; k < k_tot; k++) {
    log_prior(k) = pi_prior_alpha(k) * e_log_pi(k);
  }
  return log_prior;
}


// [[Rcpp::export]]
double MarginalLogLikelihoodWithPriors(const MatrixXd x,
				       const MatrixXd mu,
				       const MatrixXd lambda,
				       const VectorXd pi,
				       const bool use_mu_prior,
				       const bool use_lambda_prior,
				       const bool use_pi_prior,
				       const MatrixXd mu_prior_mean,
				       const MatrixXd mu_prior_info,
				       const MatrixXd lambda_prior_v_inv,
				       const VectorXd lambda_prior_n,
		 		       const VectorXd pi_prior_alpha,
				       const bool debug = false) {
  // Get the marginal log likelihood given a set of parameters.
  //
  // Args:
  //   - x: An n by p matrix of observations
  //   - mu: A p by k matrix of multivariate means
  //   - lambda: A p * (p + 1) / 2 by k matrix of the upper triangular elements
  //     of the lambda matrix.
  //   - pi: A k by one vector of the component weights.
  //   - use_X_prior: Whether or not to add the contribution of prior X.
  //   - mu_prior_mean: p by k vector of prior means.
  //   - mu_prior_info: (p + 1) * p / 2 by k vector of prior info.
  //   - lambda_prior_v_inv: (p + 1) * p / 2 by k prior value of the v inverse term.
  //   - lambda_prior_n: k - vector of prior n parameters.
  //   - pi_prior_alpha: Size k vector of pi prior parameters.


  int n_tot = x.rows();
  int p_tot = x.cols();
  int k_tot = mu.cols();
  int matrix_size = (p_tot * (p_tot + 1)) / 2;

  if (lambda.cols() != k_tot) {
    Rcpp::Rcout << "lambda has the wrong number of columns.\n";
    return 0.0;
  }
  if (lambda.rows() != matrix_size) {
    Rcpp::Rcout << "lambda has the wrong number of columns.\n";
    return 0.0;
  }
  if (pi.size() != k_tot) {
    Rcpp::Rcout << "pi is the wrong size.\n";
    return 0.0;
  }

  VectorXd lik_vec(n_tot);
  lik_vec.setZero();

  // The likelihood, which is used for variational updates, needs these
  // extra parameters.
  MatrixXd mu2 = GetVectorizedOuterProductMatrix(mu);
  VectorXd log_det_lambda = LogDeterminantOfLinearizedMatrices(lambda);

  double total_log_likelihood = 0.0;
  if (use_mu_prior) {
    VectorXd log_prior_mu = LogPriorMu(mu,
				       mu2,
				       mu_prior_mean,
				       mu_prior_info);
    for (int k = 0; k < k_tot; k++) {
      total_log_likelihood += log_prior_mu(k);
    }
  }
  if (use_lambda_prior) {
    VectorXd log_prior_lambda = LogPriorLambda(lambda,
					       log_det_lambda,
					       lambda_prior_v_inv,
					       lambda_prior_n);
    for (int k = 0; k < k_tot; k++) {
      total_log_likelihood += log_prior_lambda(k);
    }

  }
  if (use_pi_prior) {
    VectorXd log_prior_pi = LogPriorPi(pi, pi_prior_alpha);
    for (int k = 0; k < k_tot; k++) {
      total_log_likelihood += log_prior_pi(k);
    }
  }
  

  // First accumulate the likelihood.
  for (int k = 0; k < k_tot; k++) {

    VectorXd this_mu = mu.block(0, k, p_tot, 1);
    VectorXd this_mu2 = mu2.block(0, k, matrix_size, 1);
    VectorXd this_lambda = lambda.block(0, k, matrix_size, 1);
    for (int n = 0; n < n_tot; n++) {
      VectorXd this_x = x.block(n, 0, 1, p_tot).adjoint();
      double log_lik = MVNLogLikelihoodPoint(this_x,
					     this_mu,
					     this_mu2,
					     this_lambda,
					     log_det_lambda(k),
					     log(pi(k)));
      if (debug) {
	Rcpp::Rcout << n << ", " << k << ", " << log_lik << "\n";
      }
      // The weight, pi, is already in the log likelihood.
      lik_vec(n) += exp(log_lik);

      // On the last pass through, add up the log likelihood.
      if (k == k_tot - 1) {
	total_log_likelihood += log(lik_vec(n));
      }
    }
  }
  return total_log_likelihood;
}


// [[Rcpp::export]]
double MarginalLogLikelihood(const MatrixXd x,
			     const MatrixXd mu,
			     const MatrixXd lambda,
			     const VectorXd pi) {
  // Marginal log likelihood with no priors.
  MatrixXd empty_matrix(1, 1);
  VectorXd empty_vector(1);
  return MarginalLogLikelihoodWithPriors(x, mu, lambda, pi,
					 false, false, false,
					 empty_matrix, empty_matrix,
					 empty_matrix, empty_vector,
					 empty_vector);
}


// [[Rcpp::export]]
double CompleteLogLikelihoodWithPriors(const MatrixXd x,
				       const MatrixXd z,
				       const MatrixXd mu,
				       const MatrixXd mu2,
				       const MatrixXd lambda,
				       const VectorXd log_det_lambda,
				       const VectorXd log_pi,
				       const bool use_mu_prior,
				       const bool use_lambda_prior,
				       const bool use_pi_prior,
				       const MatrixXd mu_prior_mean,
				       const MatrixXd mu_prior_info,
				       const MatrixXd lambda_prior_v_inv,
				       const VectorXd lambda_prior_n,
		 		       const VectorXd pi_prior_alpha,
				       const bool debug = false) {
  // Get the marginal log likelihood given a set of parameters.
  //
  // Args:
  //   TODO

  int n_tot = x.rows();
  int p_tot = x.cols();
  int k_tot = mu.cols();
  int matrix_size = (p_tot * (p_tot + 1)) / 2;

  // TODO: dimension checks

  double tot_log_lik = 0.0;
  // Currently only the lambda prior is implemented.
  if (use_lambda_prior) {
    VectorXd log_prior_lambda = LogPriorLambda(lambda,
					       log_det_lambda,
					       lambda_prior_v_inv,
					       lambda_prior_n);
    for (int k = 0; k < k_tot; k++) {
      tot_log_lik += log_prior_lambda(k);
    }

  }

  for (int k = 0; k < k_tot; k++) {
    VectorXd this_mu = mu.block(0, k, p_tot, 1);
    VectorXd this_mu2 = mu2.block(0, k, matrix_size, 1);
    VectorXd this_lambda = lambda.block(0, k, matrix_size, 1);
    for (int n = 0; n < n_tot; n++) {
      VectorXd this_x = x.block(n, 0, 1, p_tot).adjoint();
      double log_lik = MVNLogLikelihoodPoint(this_x,
					     this_mu,
					     this_mu2,
					     this_lambda,
					     log_det_lambda(k),
					     log_pi(k));
      if (debug) {
	Rcpp::Rcout << n << ", " << k << ", " << log_lik << "\n";
      }
      tot_log_lik += z(n, k) * log_lik;
    }
  }
  return tot_log_lik;
}


// [[Rcpp::export]]
double CompleteLogLikelihoodWithX2(const MatrixXd x,
				   const MatrixXd x2,
				   const MatrixXd z,
				   const MatrixXd mu,
				   const MatrixXd mu2,
				   const MatrixXd lambda,
				   const VectorXd log_det_lambda,
				   const VectorXd log_pi,
				   const bool debug = false) {
  // Get the marginal log likelihood given a set of parameters
  // where you can specify x2 as somthing other than the actual
  // square of x.  This is currently used only for unit testing
  //  the x sensitivity functions.

  int n_tot = x.rows();
  int p_tot = x.cols();
  int k_tot = mu.cols();
  int matrix_size = (p_tot * (p_tot + 1)) / 2;

  // TODO: dimension checks

  double tot_log_lik = 0.0;

  for (int k = 0; k < k_tot; k++) {
    VectorXd this_mu = mu.block(0, k, p_tot, 1);
    VectorXd this_mu2 = mu2.block(0, k, matrix_size, 1);
    VectorXd this_lambda = lambda.block(0, k, matrix_size, 1);
    for (int n = 0; n < n_tot; n++) {
      VectorXd this_x = x.block(n, 0, 1, p_tot).adjoint();
      VectorXd this_x2 = x2.block(n, 0, 1, matrix_size).transpose();
      double log_lik = MVNLogLikelihoodPointWithX2(this_x,
						   this_x2,
						   this_mu,
						   this_mu2,
						   this_lambda,
						   log_det_lambda(k),
						   log_pi(k));
      if (debug) {
	Rcpp::Rcout << n << ", " << k << ", " << log_lik << "\n";
      }
      tot_log_lik += z(n, k) * log_lik;
    }
  }
  return tot_log_lik;
}


// [[Rcpp::export]]
double WishartEntropy(const double log_det_v,
		      const double n_par,
		      const int p_tot) {
  // Get the entropy of a Wishart distribution up to a constant.
  //
  // Args:
  //   - log_det_v: The log of the determinant of the V parameter.
  //   - n_par: The n parameter of the Wishart distribution.
  // Returns:
  //   - The entropy of the Wishart distribution up to a constant.

  double p_float = (double)p_tot;
  return (0.5 * log_det_v * (p_float + 1.0) +
	  CppMultivariateLgamma(0.5 * n_par, p_float) -
	  0.5 * (n_par - p_float - 1.0) * CppMultivariateDigamma(0.5 * n_par, p_float) +
	  0.5 * n_par * p_float);
}


// [[Rcpp::export]]
double DirichletEntropy(const VectorXd alpha) {
  // Get the entropy of a dirichlet distribution up to a constant.
  //
  // Args:
  //   - alpha: A vector of the Dirichlet parameters.
  //
  // Returns:
  //   The entropy of the distribution up to a constant.

  int k_tot = alpha.size();
  double entropy = 0.0;
  double alpha_tot = 0.0;
  for (int k = 0; k < k_tot; k++) {
    alpha_tot += alpha(k);
    entropy += CppLgamma(alpha(k)) - (alpha(k) - 1) * CppDigamma(alpha(k));
  }
  entropy += (alpha_tot - (double)k_tot) * CppDigamma(alpha_tot) - CppLgamma(alpha_tot);
  return entropy;
}


// [[Rcpp::export]]
double MultinouliiEntropy(const MatrixXd p_mat) {
  // Get the total entropy of a number of one-observation multinomial distribution.
  //
  // Args:
  //   - p_mat: An n by k matrix containing the probabilities
  //     of n multinomial distributions, each of k dimensions each.
  //
  // Returns:
  //   The total entropy of all n multinomial distributions.

  int n_tot = p_mat.rows();
  int k_tot = p_mat.cols();
  
  double entropy = 0.0;
  for (int n = 0; n < n_tot; n++) {
    for (int k = 0; k < k_tot; k++) {
      if (p_mat(n, k) > 1e-8) {
	// Anything smaller than this is probably small enough
	// that we do not need to add it.
	// TODO: Choose this bound in a more principled way.
	entropy -= p_mat(n, k) * log(p_mat(n, k));
      }
    }
  }
  return entropy;
}


// [[Rcpp::export]]
double GetVariationalEntropy(const MatrixXd z,
			     const MatrixXd mu,
			     const MatrixXd mu2,
			     const MatrixXd lambda_par,
			     const VectorXd n_par,
			     const VectorXd pi_par) {
  // Get the entropy of the variational distribution (which can
  // be used with the log likelihood to get the ELBO).
  //
  // Args:
  //   TODO

  double entropy = 0.0;

  int n_tot = z.rows();
  int k_tot = z.cols();
  int p_tot = mu.rows();

  // TODO: dimension checks

  // Get the log deterimnant of the mu covariances.
  MatrixXd mu_cov = mu2 - GetVectorizedOuterProductMatrix(mu);
  VectorXd mu_log_det_cov = LogDeterminantOfLinearizedMatrices(mu_cov);

  // Get the log determinant of the lambda parameters.
  VectorXd log_det_lambda_par = LogDeterminantOfLinearizedMatrices(lambda_par);
  for (int k = 0; k < k_tot; k++) {
    // First get the mu entropy.
    entropy += 0.5 * mu_log_det_cov(k);

    // Next get the lambda entropy.
    entropy += WishartEntropy(log_det_lambda_par(k), n_par(k), p_tot);
  }

  // Next get the z entropy.
  entropy += MultinouliiEntropy(z);

  // Finally get the pi entropy.
  entropy += DirichletEntropy(pi_par);

  return entropy;
}




///////////////////////////////////////////
// Update functions
///////////////////////////////////////////


// [[Rcpp::export]]
Rcpp::List UpdateMuPosterior(const MatrixXd x,
			     const MatrixXd e_lambda_inv_mat,
			     const MatrixXd e_z) {
  // Args:
  //     to be updated.
  //   - x: An n by p data matrix
  //   - e_sigma_mat: The inverse of the expectation of the variational
  //     lambda matrix, which is (p + 1) * p / 2 by k.
  //   - e_z: A z by k matrix of E_q(z).
  //
  // Returns:
  //   A list containing:
  //   - e_mu: A p by k matrix of E_q (mu) to be updated
  //   - e_mu2: A (p + 1) * p / 2 by k matrix of Cov_q(mu_i, mu_j)

  int n_tot = x.rows();
  int p_tot = x.cols();
  int k_tot = e_z.cols();
  int matrix_size = (p_tot * (p_tot + 1)) / 2;

  MatrixXd e_mu(p_tot, k_tot);
  MatrixXd e_mu2(matrix_size, k_tot);

  // if (e_mu.rows() != p_tot) {
  //   Rcpp::Rcout << "e_mu has the wrong number of rows.\n";
  //   return error_code;
  // }
  // if (e_mu.cols() != k_tot) {
  //   Rcpp::Rcout << "e_mu has the wrong number of columns.\n";
  //   return error_code;
  // }
  // if (e_mu2.cols() != k_tot) {
  //   Rcpp::Rcout << "e_mu2 has the wrong number of columns.\n";
  //   return error_code;
  // }
  // if (e_mu2.rows() != matrix_size) {
  //   Rcpp::Rcout << "e_mu2 has the wrong number of columns.\n";
  //   return error_code;
  // }
  if (e_lambda_inv_mat.cols() != k_tot) {
    Rcpp::Rcout << "e_lambda_inv_mat has the wrong number of columns.\n";
    return Rcpp::List::create(Rcpp::Named("empty") = 0.0);
  }
  if (e_lambda_inv_mat.rows() != matrix_size) {
    Rcpp::Rcout << "e_lambda_inv_mat has the wrong number of columns.\n";
    return Rcpp::List::create(Rcpp::Named("empty") = 0.0);
  }
  if (e_z.rows() != n_tot) {
    Rcpp::Rcout << "e_z has the wrong number of rows.\n";
    return Rcpp::List::create(Rcpp::Named("empty") = 0.0);
  }

  for (int k = 0; k < k_tot; k++) {

    // Update the E(mu) terms.
    double z_tot = 0;
    for (int p = 0; p < p_tot; p++) {
      double sum_x_z = 0.0;
      for (int n = 0; n < n_tot; n++) {
	if (p == 0) {
	  // Only add up the z total for this k on the first pass through p
	  // since it won't change.
	  z_tot += e_z(n, k);
	}
	sum_x_z += e_z(n, k) * x(n, p);
      }
      e_mu(p, k) = sum_x_z / z_tot;
    } // p loop

    // Update the Cov(mu_a, mu_b) terms.
    for (int a = 0; a < p_tot; a++) {
      for (int b = 0; b < p_tot; b++) {
	int this_index = GetUpperTriangularIndex(a, b);
	e_mu2(this_index, k) = (e_lambda_inv_mat(this_index, k) / z_tot +
				e_mu(a, k) * e_mu(b, k));
      }
    }
  } // k loop

  return Rcpp::List::create(Rcpp::Named("e_mu") = e_mu,
			    Rcpp::Named("e_mu2") = e_mu2) ;
}


// [[Rcpp::export]]
Rcpp::List UpdateLambdaPosterior(const MatrixXd x,
				 const MatrixXd e_mu,
				 const MatrixXd e_mu2,
				 const MatrixXd e_z,
				 const bool use_prior,
				 const MatrixXd lambda_prior_v_inv,
				 const VectorXd lambda_prior_n) {
  // Args:
  //  - e_mu: A p by k matrix of E_q (mu) to be updated
  //  - e_mu2: A (p + 1) * p / 2 by k matrix of Cov_q(mu_i, mu_j)
  //    to be updated.
  //  - x: An n by p data matrix
  //  - e_z: A z by k matrix of E_q(z).
  //  - use_prior: If true, shrink towards the specified prior.
  //  - lambda_prior_v_inv: (p + 1) * p / 2 by k prior value of the v inverse term.
  //  - lambda_prior_n: k - vector of prior n parameters.
  //
  // Returns:
  //  A list containing
  //  - lambda_par: A (p + 1) * p / 2 by k matrix of the "V" terms from 
  //    the wishart distribution of lambda.
  //  - n_par: A k-length vector of the Wishart n parameters for Lambda.
  //  NB: these are not the posterior expectations,
  //  but the Wishart parameters.

  int n_tot = x.rows();
  int p_tot = x.cols();
  int k_tot = e_mu.cols();
  int matrix_size = (p_tot * (p_tot + 1)) / 2;
  
  if (e_z.cols() != k_tot) {
    Rcpp::Rcout << "e_z has the wrong number of columns.\n";
    return Rcpp::List::create(Rcpp::Named("empty") = 0.0);
  }
  if (e_z.rows() != n_tot) {
    Rcpp::Rcout << "e_z has the wrong number of rows.\n";
    return Rcpp::List::create(Rcpp::Named("empty") = 0.0);
  }
  if (e_mu.cols() != k_tot) {
    Rcpp::Rcout << "e_mu has the wrong number of columns.\n";
    return Rcpp::List::create(Rcpp::Named("empty") = 0.0);
  }
  if (e_mu.rows() != p_tot) {
    Rcpp::Rcout << "e_mu has the wrong number of rows.\n";
    return Rcpp::List::create(Rcpp::Named("empty") = 0.0);
  }
  if (e_mu2.cols() != k_tot) {
    Rcpp::Rcout << "e_mu2 has the wrong number of columns.\n";
    return Rcpp::List::create(Rcpp::Named("empty") = 0.0);
  }
  if (e_mu2.rows() != matrix_size) {
    Rcpp::Rcout << "e_mu2 has the wrong number of rows.\n";
    return Rcpp::List::create(Rcpp::Named("empty") = 0.0);
  }

  MatrixXd lambda_par(matrix_size, k_tot);
  VectorXd n_par(k_tot);

  for (int k = 0; k < k_tot; k++) {
    MatrixXd v_inv(p_tot, p_tot);
    if (use_prior) {
      v_inv =
	ConvertVectorToSymmetricMatrix(lambda_prior_v_inv.block(0, k,
								matrix_size, 1));
      n_par(k) = lambda_prior_n(k);
    } else {
      v_inv.setZero();
      n_par(k) = 0;
    }
    n_par(k) += 1 + p_tot;
    for (int n = 0; n < n_tot; n++) {
      double this_z = e_z(n, k);
      n_par(k) += this_z;
      for (int a = 0; a < p_tot; a++) {
	for (int b = 0; b <= a; b++) {
	  // TODO: upon reflection, maybe it makes more sense for x to
	  // be stored with each column being one observation rather
	  // than each row to avoid looking values up across columns.
	  double this_term = (x(n, a) * x(n, b) -
			      e_mu(a, k) * x(n, b) -
			      e_mu(b, k) * x(n, a) +
			      e_mu2(GetUpperTriangularIndex(a, b), k)) * this_z;
	  v_inv(a, b) += this_term;
	}
      }
    } // n loop

    // Make v_inv symmetric.
    for (int a = 0; a < p_tot; a++) {
      for (int b = 0; b < a; b++) {
	v_inv(b, a) = v_inv(a, b);
      }
    }
    
    MatrixXd v = v_inv.inverse();
    for (int a = 0; a < p_tot; a++) {
      for (int b = 0; b <= a; b++) {
	// TODO: remove this sanity check
	int row_index = GetUpperTriangularIndex(a, b);
	if (row_index > lambda_par.rows()) {
	  Rcpp::Rcout << "Bad row index: " << row_index << "\n";
	}
	if (k > lambda_par.cols()) {
	  Rcpp::Rcout << "Bad column index: " << k << "\n";
	}
	lambda_par(row_index, k) = v(a, b);
      }
    }
  } // k loop
  return Rcpp::List::create(Rcpp::Named("lambda_par") = lambda_par,
			    Rcpp::Named("n_par") = n_par);

}


// [[Rcpp::export]]
VectorXd WishartELogDet(const MatrixXd lambda_par,
			const VectorXd n_par) {
  // Args:
  //   - lambda_par: A (p + 1) * p / 2 by k matrix of the "V" terms from 
  //    the wishart distribution of lambda.
  //   - n_par: A k-length vector of the Wishart n parameters for Lambda.
  //
  // Returns:
  //   A k-length vector of the expectations of log det Lambda for each
  //   component.

  int matrix_size = lambda_par.rows();
  int k_tot = n_par.size();
  int p_tot = GetMatrixSizeFromUTSize(matrix_size);
  VectorXd e_log_det(k_tot);  
  if (lambda_par.cols() != k_tot) {
    Rcpp::Rcout << "lambda_par has the wrong number of columns.\n";
    return e_log_det;
  }
  if (p_tot == -1) {
    Rcpp::Rcout << "WishartELogDet: lambda_par's rows is not a triangular number.\n";
    return e_log_det;
  }

  for (int k = 0; k < k_tot; k++) {
    MatrixXd v(p_tot, p_tot);
    for (int a = 0; a < p_tot; a++) {
      for (int b = 0; b < p_tot; b++) {
	v(a, b) = lambda_par(GetUpperTriangularIndex(a, b), k);
      }
    }

    e_log_det(k) = (p_tot * log(2) +
		    log(v.determinant()) +
		    CppMultivariateDigamma(n_par(k) / 2, p_tot));
  }
  return e_log_det;
}



///////////////////////////////////////////
// Covariance functions
///////////////////////////////////////////


// [[Rcpp::export]]
VectorXd GetELogDirichlet(const VectorXd alpha) {
  // Args:
  //   - q: The dirichlet parameters for a dirichlet variable x.
  // Returns:
  //   - A vector of E(log x) of the same dimension as alpha.

  int k = alpha.size();
  double alpha_sum = 0.0;
  for (int index = 0; index < k; index++) {
    alpha_sum += alpha(index);
  }
  double digamma_alpha_sum = CppDigamma(alpha_sum);
  VectorXd e_log_alpha(k);
  for (int index = 0; index < k; index++) {
    e_log_alpha(index) = CppDigamma(alpha(index)) - digamma_alpha_sum;
  }
  return e_log_alpha;
}

// [[Rcpp::export]]
MatrixXd GetLogDirichletCovariance(const VectorXd alpha) {
  // Args:
  //  - alpha: A vector of dirichlet parameters.
  //
  // Returns:
  //  - The covariance of the log of a dirichlet distribution
  //    with parameters alpha.
  
  int k = alpha.size();
  int k_index;
  MatrixXd cov_mat(k, k);
  
  // Precomute the total.
  double alpha_0 = 0.0;
  for (k_index = 0; k_index < k; k_index++) {
    alpha_0 += alpha(k_index);
  }
  double covariance_term = -1.0 * CppTrigamma(alpha_0);
  cov_mat.setConstant(covariance_term);
  
  // Only the diagonal entries deviate from covariance_term.
  for (k_index = 0; k_index < k; k_index++) {
    cov_mat(k_index, k_index) += CppTrigamma(alpha(k_index));
  }
  return cov_mat;
}


// [[Rcpp::export]]
SparseMatrix<double> GetLogPiVariance(const VectorXd pi_par) {
  // Args:
  //   - pi_par: K-length vector of the posterior dirichlet parameters
  //     for individuals' population memberships.
  // Returns:
  //   The variational covariance of log pi (see notes) of dimension k x k.
  //   I return a sparse matrix because it will be used in sparse
  //   expressions later.
  
  int k = pi_par.size();
  
  SparseMatrix<double> var_log_pi(k, k);
  // There will be k non-zero terms per column.
  var_log_pi.reserve(Eigen::VectorXi::Constant(k, k));

  MatrixXd dense_cov = GetLogDirichletCovariance(pi_par);
  for (int k1_index = 0; k1_index < k; k1_index++) {
    for (int k2_index = 0; k2_index < k; k2_index++) {
      var_log_pi.insert(k1_index, k2_index) = dense_cov(k1_index, k2_index);
    }
  }
  var_log_pi.makeCompressed();
  return var_log_pi;
}


MatrixXd GetMVNCovariances(const MatrixXd mu_par,
			   const MatrixXd mu2_par) {
  // Get the covariances from the first and second non-central
  // moments.
  int k_tot = mu_par.cols();
  int p_tot = mu_par.rows();
  MatrixXd mu_cov(mu2_par.rows(), mu2_par.cols());

  // TODO: check the input sizes?
  for (int k = 0; k < k_tot; k++) {
    for (int j = 0; j < p_tot; j++) {
      for (int i = 0; i <= j; i++) {
	int index = GetUpperTriangularIndex(i, j);
	mu_cov(index, k) = mu2_par(index, k) - mu_par(i, k) * mu_par(j, k);
      }
    }
  }
  return mu_cov;
}


double GetFourthOrderCovariance(const MatrixXd mu_par,
			        const MatrixXd mu_cov, int k,
				int a, int b, int c, int d) {
  // Get Cov(mu_a mu_b, mu_c mu_d) from the means and parwise covariances.
  double cov_ac = mu_cov(GetUpperTriangularIndex(a, c), k);
  double cov_ad = mu_cov(GetUpperTriangularIndex(a, d), k);
  double cov_bc = mu_cov(GetUpperTriangularIndex(b, c), k);
  double cov_bd = mu_cov(GetUpperTriangularIndex(b, d), k);
  double m_a = mu_par(a, k);
  double m_b = mu_par(b, k);
  double m_c = mu_par(c, k);
  double m_d = mu_par(d, k);

  return (cov_ac * cov_bd + cov_ad * cov_bc +
	  cov_ac * m_b * m_d + cov_ad * m_b * m_c +
	  cov_bc * m_a * m_d + cov_bd * m_a * m_c);
};


// [[Rcpp::export]]
SparseMatrix<double> GetMuVariance(const MatrixXd mu_par,
				   const MatrixXd mu2_par) {
  // Args:
  //   - mu_par: A p by k matrix of variational E(mu)
  //   - mu2_par: A (p + 1) * p / 2 by k matrix of variational E(mu_i mu_j)
  // Returns:
  //   A sparse matrix of the
  //   covariance of the (mu, mu2) vector as indexed by
  //   the FullParameterIndices class.
  //
  // I won't bother trying to detect the symmetries, so this has
  // some redundant computation.

  int p_tot = mu_par.rows();
  int k_tot = mu_par.cols();

  int matrix_tot = mu2_par.rows();

  SparseMatrix<double> var_mu((matrix_tot + p_tot) * k_tot,
			      (matrix_tot + p_tot) * k_tot);
  if (k_tot != mu2_par.cols()) {
    Rcpp::Rcout << "Error: mu2_par has the wrong number of columns.\n";
    return var_mu;
  }

  if (p_tot * (p_tot + 1) / 2 != matrix_tot) {
    Rcpp::Rcout << "Error: mu2_par has the wrong number of rows.\n";
    return var_mu;
  }

  // The covariances are a more convenient way to compute the
  // fourth-order covariances.
  MatrixXd mu_cov = GetMVNCovariances(mu_par, mu2_par);

  FullParameterIndices ind(k_tot, p_tot);
  std::vector<Triplet> var_mu_t;
  // There are k * (p + matrix_tot) ^ 2 nonzero entries
  var_mu_t.reserve((p_tot + matrix_tot) * (p_tot + matrix_tot) * k_tot);
  for (int k = 0; k < k_tot; k++) {
    // Get the mu covariances first.
    for (int j = 0; j < p_tot; j++) {
      int this_mu_index = ind.MuOnlyMuCoord(k, j);
      for (int i = 0; i <= j; i++) {
	double this_cov = mu_cov(GetUpperTriangularIndex(i, j), k);
	var_mu_t.push_back(Triplet(this_mu_index,
				   ind.MuOnlyMuCoord(k, i),
				   this_cov));
	if (i != j) {
	  var_mu_t.push_back(Triplet(ind.MuOnlyMuCoord(k, i),
				     this_mu_index,
				     this_cov));
	}
      }
    } // End of mu_j loop

    // Get covariance of linear terms with quadratic terms.
    for (int a = 0; a < p_tot; a++) {
      double m_a = mu_par(a, k);
      for (int b = 0; b <= a; b++) {
	double m_b = mu_par(b, k);
	for (int c = 0; c < p_tot; c++) {
	  double cov_ac = mu_cov(GetUpperTriangularIndex(a, c), k);
	  double cov_bc = mu_cov(GetUpperTriangularIndex(b, c), k);
	  double this_cov = m_a * cov_bc + m_b * cov_ac;
	  var_mu_t.push_back(Triplet(ind.MuOnlyMuCoord(k, c),
				     ind.MuOnlyMu2Coord(k, a, b),
				     this_cov));
	  var_mu_t.push_back(Triplet(ind.MuOnlyMu2Coord(k, a, b),
				     ind.MuOnlyMuCoord(k, c),
				     this_cov));
	}
      }
    }

    // Get covariance of quadratic terms with other quadratic terms.
    for (int a = 0; a < p_tot; a++) {
      for (int b = 0; b <= a; b++) {
	for (int c = 0; c < p_tot; c++) {
	  for (int d = 0; d <= c; d++) {
	    double this_cov = GetFourthOrderCovariance(mu_par,
						       mu_cov, k,
						       a, b, c, d);
	    var_mu_t.push_back(Triplet(ind.MuOnlyMu2Coord(k, a, b),
				       ind.MuOnlyMu2Coord(k, c, d),
				       this_cov));
	  }
	}
      }
    }
  } // End of k loop

  // Rcpp::Rcout << "Making matrix\n";
  // std::vector<Triplet>::const_iterator i;
  // for (i = var_mu_t.begin(); i != var_mu_t.end(); ++i) {
  //   Rcpp::Rcout << i->col() << "," << i->row() << "," << i->value() << "\n";
  // }
  // Rcpp::Rcout << "That's all\n";
  var_mu.setFromTriplets(var_mu_t.begin(), var_mu_t.end());
  var_mu.makeCompressed();
  return var_mu;
}


// [[Rcpp::export]]
MatrixXd GetWishartLinearCovariance(const VectorXd w_par, double n_par) {
  // Construct the covariance of the elements of a Wishart-distributed
  // matrix.
  //
  // Args:
  //   - w_par: A linearized representation of the upper triangular portion
  //            of the wishart parameter.
  //   - n_par: The n parameter of the Wishart distribution.
  // Returns:
  //   - The covariance of the linear terms of the Wishart distribution.
 
  int vec_n = w_par.size();
  MatrixXd w_cov(vec_n, vec_n);
  int n = GetMatrixSizeFromUTSize(vec_n);
  if (n == -1) {
    Rcpp::Rcout << "Bad size for w_par.\n";
    return w_cov;
  }

  // Iterate over the matrix, not the vector representation.
  for (int j1 = 0; j1 < n; j1++) {
    for (int i1 = 0; i1 <= j1; i1++) {
      for (int j2 = 0; j2 < n; j2++) {
	for (int i2 = 0; i2 <= j2; i2++) {
	  int p = GetUpperTriangularIndex(i1, j1);
	  int q = GetUpperTriangularIndex(i2, j2);
	  w_cov(p, q) = n_par * (w_par(GetUpperTriangularIndex(i1, j2)) *
				 w_par(GetUpperTriangularIndex(i2, j1)) +
				 w_par(GetUpperTriangularIndex(i1, i2)) *
				 w_par(GetUpperTriangularIndex(j1, j2)));
	  if (p != q) {
	    w_cov(q, p) = w_cov(p, q);
	  }
	}
      }
    }
  }
  return w_cov;
}


// [[Rcpp::export]]
VectorXd GetWishartLinearLogDetCovariance(const VectorXd w_par) {
  // Construct the covariance between the elements of a Wishart-distributed
  // matrix and the log determinant.
  //
  // Args:
  //   - w_par: A linearized representation of the upper triangular portion
  //            of the wishart parameter.
  //
  // Returns:
  //   - The covariance between the linear terms of the Wishart distribution
  //     and the log determinant.

  return 2.0 * w_par;
}


// [[Rcpp::export]]
double GetWishartLogDetVariance(double n_par, int p) {
  return CppMultivariateTrigamma(n_par / 2, p);
}


// [[Rcpp::export]]
SparseMatrix<double> GetLambdaVariance(const MatrixXd lambda_par, const VectorXd n_par) {
  // Args:
  //   - lambda_par: A p * (p + 1) / 2 by k matrix of the upper triangular elements
  //     of the v matrices of the wishart parameterization of lambda.
  //   - n_par: A k by 1 vector of the n parameters of the wishart distributions.

  int p_tot = GetMatrixSizeFromUTSize(lambda_par.rows());
  int k_tot = lambda_par.cols();

  FullParameterIndices ind(k_tot, p_tot);
  SparseMatrix<double> lambda_cov(ind.FullLambdaSize(),
				  ind.FullLambdaSize());
  if (p_tot == -1) {
    Rcpp::Rcout << "The number of rows in lambda_par is not a triangular number.\n";
    return lambda_cov;
  }
  
  if (n_par.size() != k_tot) {
    Rcpp::Rcout << "The length of n_par is not the same as the number of columns of lambda_par\n";
    return lambda_cov;
  }

  std::vector<Triplet> lambda_cov_t;
  // There are (MatrixSize() + 1) ^ 2 nonzero entries for every k.
  lambda_cov_t.reserve(((ind.MatrixSize() + 1) * (ind.MatrixSize() + 1)) * k_tot);

  MatrixXd linear_cov(ind.MatrixSize(), ind.MatrixSize());
  VectorXd linear_det_cov(ind.MatrixSize());
  double det_var;
  for (int k = 0; k < k_tot; k++) {
    VectorXd this_lambda_par = lambda_par.block(0, k, 1, ind.MatrixSize());
    double this_n_par = n_par(k);
    linear_cov = GetWishartLinearCovariance(this_lambda_par, this_n_par);
    linear_det_cov = GetWishartLinearLogDetCovariance(this_lambda_par);
    det_var = GetWishartLogDetVariance(this_n_par, p_tot);

    int log_det_index = ind.LambdaOnlyLogDetLambdaCoord(k);
    lambda_cov_t.push_back(Triplet(log_det_index,
				   log_det_index,
				   det_var));

    for (int a = 0; a < p_tot; a++) {
      for (int b = 0; b <= a; b++) {
	int ab_index = ind.LambdaOnlyLambdaCoord(k, a, b);
	double this_cov = linear_det_cov(GetUpperTriangularIndex(a, b));
	lambda_cov_t.push_back(Triplet(ab_index,
				       log_det_index,
				       this_cov));
	lambda_cov_t.push_back(Triplet(log_det_index,
				       ab_index,
				       this_cov));
	for (int c = 0; c < p_tot; c++) {
	  for (int d = 0; d <= c; d++) {
	    int cd_index = ind.LambdaOnlyLambdaCoord(k, c, d);
	    double this_cov = linear_cov(GetUpperTriangularIndex(a, b),
					 GetUpperTriangularIndex(c, d));
	    lambda_cov_t.push_back(Triplet(ab_index, cd_index, this_cov));
	  }
	}
      }
    }
  }
  lambda_cov.setFromTriplets(lambda_cov_t.begin(), lambda_cov_t.end());
  lambda_cov.makeCompressed();
  return lambda_cov;
}


SparseMatrix<double> GetXVarianceCore (const MatrixXd x,
				       const VectorXd x_indices,
				       const bool full_x) {
  // Return a matrix proportional to the covariance matrix
  // of the x sufficient statistics with proportionality constant epsilon.
  //
  // If full_x = true, then x_indices is ignored and it gets the sensitivity
  // for the whole x vector.  Otherwise, it gets it only for the rows indexed
  // by x_indices.

  int n_tot = x.rows();
  int p_tot = x.cols();

  int x_tot;
  if (full_x) {
    x_tot = n_tot;
  } else {
    x_tot = x_indices.size();
  }

  XParameterIndices x_ind(x_tot, p_tot);
  SparseMatrix<double> x_cov(x_ind.Dim(), x_ind.Dim());

  if (x_tot > n_tot) {
    Rcpp::Rcout << "x_indices is longer than n_tot.\n";
    return x_cov;
  }

  std::vector<Triplet> t;
  // Each element of the matrix x has 2 + p_tot non-zero terms in the
  //covariance matrix: var(x_ij), var(x_ij * x_ij), and cov(x_ij, x_ij * x_ij')
  t.reserve(n_tot * p_tot * (2 + p_tot));

  // Make the linearized covariance matrix.
  // Dig that GetFourthOrderCovariance requires matrix input.
  // MatrixXd p_identity(p_tot, p_tot);
  // p_identity.setIdentity();
  // VectorXd x_cov_vector = ConvertSymmetricMatrixToVector(p_identity);
  // MatrixXd x_row_cov(x_cov_vector.size(), 1);
  // x_row_cov.block(0, 0, x_cov_vector.size(), 1) = x_cov_vector;

  // Loop over the specified rows.
  for (int x_ind_row = 0; x_ind_row < x_tot; x_ind_row++) {
    int n;
    if (full_x) {
      n = x_ind_row;
    } else {
      n = x_indices(x_ind_row);
    }

    if (n >= n_tot) {
      Rcpp::Rcout << "Error: an x_index is greater than n_tot.\n";
      return x_cov;
    }

    // For each row, get all the covariances.
    for (int a = 0; a < p_tot; a++) {
      int xa_index = x_ind.XCoord(x_ind_row, a);

      // The variance of linear terms.
      t.push_back(Triplet(xa_index, xa_index, 1.0));
      for (int b = 0; b < p_tot; b++) {

	// The covariance of the linear and quadratic terms.
	int xa_xb_index = x_ind.X2Coord(x_ind_row, a, b);

	if (a != b) {
	  t.push_back(Triplet(xa_index, xa_xb_index,
			      x(n, b)));
	  t.push_back(Triplet(xa_xb_index, xa_index,
			      x(n, b)));
	} else {
	  t.push_back(Triplet(xa_index, xa_xb_index,
			      2 * x(n, a)));
	  t.push_back(Triplet(xa_xb_index, xa_index,
			      2 * x(n, a)));
	}
      }
    }

    // Get covariance of quadratic terms with other quadratic terms.

    // Define a MatrixXd input for GetFourthOrderCovariance.
    MatrixXd x_row_mat(p_tot, 1);
    // x_row_mat.block(0, 0, p_tot, 1) = x.block(n, 0, 1, p_tot).transpose();
    for (int a = 0; a < p_tot; a++) {
      for (int b = 0; b <= a; b++) {
	for (int c = 0; c < p_tot; c++) {
	  for (int d = 0; d <= c; d++) {
	    // double this_cov = GetFourthOrderCovariance(x_row_mat,
	    // 					       x_row_cov, 0,
	    // 					       a, b, c, d);

	    // This is the reduced fourth order covariance
	    // for the case of infinitesimal variance.
	    double this_cov = ((double)(a == c) * x(n, b) * x(n, d) +
			       (double)(a == d) * x(n, b) * x(n, c) +
			       (double)(b == c) * x(n, a) * x(n, d) +
			       (double)(b == d) * x(n, a) * x(n, c));
	    t.push_back(Triplet(x_ind.X2Coord(x_ind_row, a, b),
				x_ind.X2Coord(x_ind_row, c, d),
				this_cov));
	  }
	}
      }
    } // a loop
  } // n loop
  x_cov.setFromTriplets(t.begin(), t.end());
  x_cov.makeCompressed();
  return x_cov;
}


// [[Rcpp::export]]
SparseMatrix<double> GetXVariance(const MatrixXd x) {
  VectorXd empty_x(1);
  return GetXVarianceCore(x, empty_x, true);
}


// [[Rcpp::export]]
SparseMatrix<double> GetXVarianceSubset(const MatrixXd x,
					const VectorXd x_indices) {
  return GetXVarianceCore(x, x_indices, false);
}


// [[Rcpp::export]]
int GetZMatrixInPlace(Rcpp::NumericMatrix z,
		      const MatrixXd x,
		      const MatrixXd e_mu,
		      const MatrixXd e_mu2,
		      const MatrixXd e_lambda,
		      const VectorXd e_log_det_lambda,
		      const VectorXd e_log_pi) {
  // Args:
  //   - z: An n by k matrix that will be populated with the z values.
  //   - x: An n by p matrix of observations
  //   - mu_par: A p by k matrix of variational E(mu)
  //   - mu2_par: A (p + 1) * p / 2 by k matrix of variational E(mu_i mu_j)
  //   - lambda_par: A p * (p + 1) / 2 by k matrix of the upper triangular elements
  //     of the lambda matrix.
  //   - n_par: A k by 1 vector of the n parameters.
  //   - pi_par: A k by one vector of the pi dirichlet parameters.
  //
  // Returns:
  //   Updates z in place to an n by k matrix of z values, where the rows sum to one.

  // TODO: is this corrupting memory somehow?

  int n_tot = x.rows();
  int p_tot = x.cols();
  int k_tot = e_mu.cols();

  int error_code = -1;
  int success_code = 0;

  FullParameterIndices ind(k_tot, p_tot);
  if (e_mu2.cols() != k_tot) {
    Rcpp::Rcout << "e_mu2 has the wrong number of columns.\n";
    return error_code;
  }
  if (e_lambda.cols() != k_tot) {
    Rcpp::Rcout << "e_lambda has the wrong number of columns.\n";
    return error_code;
  }
  if (e_log_det_lambda.size() != k_tot) {
    Rcpp::Rcout << "n_par has the wrong number of rows.\n";
    return error_code;
  }
  if (e_log_pi.size() != k_tot) {
    Rcpp::Rcout << "pi_par has the wrong number of rows.\n";
    return error_code;
  }
  if (e_mu2.rows() != ind.MatrixSize()) {
    Rcpp::Rcout << "e_mu2 has the wrong number of columns.\n";
    return error_code;
  }
  if (e_lambda.rows() != ind.MatrixSize()) {
    Rcpp::Rcout << "e_lambda has the wrong number of columns.\n";
    return error_code;
  }

  // First, calculate the log likelihoods and the row maxima.
  VectorXd z_row_max(n_tot);
  for (int k = 0; k < k_tot; k++) { 
    for (int n = 0; n < n_tot; n++) {
      // VectorXd must be a column vector.
      z(n, k) = MVNLogLikelihoodPoint(x.block(n, 0, 1, p_tot).adjoint(),
				      e_mu.block(0, k, p_tot, 1),
				      e_mu2.block(0, k, ind.MatrixSize(), 1),
				      e_lambda.block(0, k, ind.MatrixSize(), 1),
				      e_log_det_lambda(k),
				      e_log_pi(k));
      if (k == 0 || z(n, k) > z_row_max(n)) {
	z_row_max(n) = z(n, k);
      }
    }
  }

  VectorXd z_row_totals(n_tot);
  z_row_totals.setZero();
  for (int k = 0; k < k_tot; k++) {
    for (int n = 0; n < n_tot; n++) {
      z(n, k) = exp(z(n, k) - z_row_max(n));
      z_row_totals(n) += z(n, k);
    }
  }

  for (int k = 0; k < k_tot; k++) {
    for (int n = 0; n < n_tot; n++) {
      z(n, k) /= (z_row_totals(n) == 0 ? 1: z_row_totals(n));
    }
  }  

  return success_code;
}


// [[Rcpp::export]]
Rcpp::NumericMatrix GetZMatrix(const MatrixXd x,
			       const MatrixXd e_mu,
			       const MatrixXd e_mu2,
			       const MatrixXd e_lambda,
			       const VectorXd e_log_det_lambda,
			       const VectorXd e_log_pi) {
  // Args and returns: The same as GetZMatrixInPlace.
  int n_tot = x.rows();
  int k_tot = e_mu.cols();
  Rcpp::NumericMatrix z(n_tot, k_tot);

  // Dimension checks are run in GetZMatrixInPlace.
  GetZMatrixInPlace(z,
		    x, e_mu, e_mu2, e_lambda,
		    e_log_det_lambda, e_log_pi);
  return z;
}



// [[Rcpp::export]]
MatrixXd GetSingleZCovariance(VectorXd z) {
  // Args:
  //   z: A size k vector of the z probabilities.
  // Returns:
  //   The covariance matrix.
  MatrixXd z_outer = (-1) * z * z.transpose();
  MatrixXd z_diagonal = z.asDiagonal();
  z_outer = z_outer + z_diagonal;
  return z_outer;
}


// [[Rcpp::export]]
SparseMatrix<double> GetZCovariance(MatrixXd z_mat) {
  // Args:
  //   - z_mat: An n by k matrix of z indicator values, where the 
  //     rows sum to one.
  //
  // Returns:
  //   The covariance of indicators, where the indicators
  //   are vectorized according to ZParameterIndices.

  int n_tot = z_mat.rows();
  int k_tot = z_mat.cols();

  ZParameterIndices ind(n_tot, k_tot);
  SparseMatrix<double> var_z(ind.Dim(), ind.Dim());
  std::vector<Triplet> var_z_t;
  // For each n, there is a k^2 sized covariance matrix.
  var_z_t.reserve(k_tot * k_tot * n_tot);
  
  for (int n = 0; n < n_tot; n++) {
    VectorXd this_z = z_mat.block(n, 0, 1, k_tot).adjoint();
    MatrixXd this_z_cov = GetSingleZCovariance(this_z);
    //Rcpp::Rcout << this_z_cov << "\n";
    for (int k1 = 0; k1 < k_tot; k1++) {
      for (int k2 = 0; k2 <= k1; k2++) {
	var_z_t.push_back(Triplet(ind.ZCoord(n, k1),
				  ind.ZCoord(n, k2),
				  this_z_cov(k1, k2)));
	if (k1 != k2) {
	  var_z_t.push_back(Triplet(ind.ZCoord(n, k2),
				    ind.ZCoord(n, k1),
				    this_z_cov(k2, k1)));
	}
      }
    }
  }
  var_z.setFromTriplets(var_z_t.begin(), var_z_t.end());
  var_z.makeCompressed();
  return var_z;
}




///////////////////////////////////////////
// Sensitivity matrices
///////////////////////////////////////////

// [[Rcpp::export]]
SparseMatrix<double> GetHThetaZ(const MatrixXd x,
				const MatrixXd e_mu,
				const MatrixXd e_mu2,
				const MatrixXd e_lambda) {
  // Theta refers to the complete collection of parameters (mu, lambda, pi).
  // This constructs d^2 H / d theta dZ'.  The rows are theta, and the columns
  // are z.

  int n_tot = x.rows();
  int p_tot = x.cols();
  int k_tot = e_mu.cols();

  FullParameterIndices ind(k_tot, p_tot);
  ZParameterIndices z_ind(n_tot, k_tot);

  SparseMatrix<double> h_theta_z(ind.Dim(), z_ind.Dim());

  if (e_mu2.cols() != k_tot) {
    Rcpp::Rcout << "e_mu2 has the wrong number of columns.\n";
    return h_theta_z;
  }
  if (e_lambda.cols() != k_tot) {
    Rcpp::Rcout << "e_lambda has the wrong number of columns.\n";
    return h_theta_z;
  }
  if (e_mu2.rows() != ind.MatrixSize()) {
    Rcpp::Rcout << "e_mu2 has the wrong number of columns.\n";
    return h_theta_z;
  }
  if (e_lambda.rows() != ind.MatrixSize()) {
    Rcpp::Rcout << "e_lambda has the wrong number of columns.\n";
    return h_theta_z;
  }

  std::vector<Triplet> t;

  // Calculate the number of terms that will be needed for each
  // set of sensititivies.  Each z_ij is sensitive to exactly one of
  // each of the parameter from each set.
  int mu_size = p_tot + ind.MatrixSize();
  int lambda_size = ind.MatrixSize() + 1;
  int pi_size = k_tot;
  t.reserve(z_ind.Dim() * (mu_size + lambda_size + pi_size));

  // We will loop over the z indices in an outer loop.
  double this_sens = 0.0;
  for (int k = 0; k < k_tot; k++) {
    for (int n = 0; n < n_tot; n++) {
      int z_index = z_ind.ZCoord(n, k);
      // First the mu_a terms.
      for (int a = 0; a < p_tot; a++) {
	this_sens = 0.0;
	for (int b = 0; b < p_tot; b++) {
	  this_sens += e_lambda(GetUpperTriangularIndex(a, b), k) * x(n, b);
	}
	t.push_back(Triplet(ind.MuCoord(k, a), z_index, this_sens));
      }

      // Next the mu2 terms.
      for (int a = 0; a < p_tot; a++) {
	for (int b = 0; b <= a; b++) {
	  this_sens = -1 * e_lambda(GetUpperTriangularIndex(b, a), k);
	  if (a == b) {
	    this_sens *= 0.5;
	  }
	  t.push_back(Triplet(ind.Mu2Coord(k, b, a), z_index, this_sens)); 
	}
      }

      // Next the lambda terms.
      for (int a = 0; a < p_tot; a++) {
	for (int b = 0; b <= a; b++) {
	  this_sens = (x(n, a) * x(n, b) -
		       x(n, a) * e_mu(b, k) -
		       x(n, b) * e_mu(a, k) +
		       e_mu2(GetUpperTriangularIndex(b, a), k));
	  double factor;
	  if (a == b) {
	    factor = -0.5;
	  } else {
	    factor = -1.0;
	  }
	  t.push_back(Triplet(ind.LambdaCoord(k, b, a), z_index, factor * this_sens));
	}
      }

      // Next the log det lambda term.
      t.push_back(Triplet(ind.LogDetLambdaCoord(k), z_index, 0.5));

      // Finally, the log pi term.
      t.push_back(Triplet(ind.LogPiCoord(k), z_index, 1.0));
    }
  }
  h_theta_z.setFromTriplets(t.begin(), t.end());
  h_theta_z.makeCompressed();
  return h_theta_z;
}


// [[Rcpp::export]]
SparseMatrix<double> GetHThetaTheta(const MatrixXd x,
				    const MatrixXd e_z) {

  // Theta refers to the complete collection of parameters (mu, lambda, pi).
  // This constructs d^2 H / d theta d theta'.  Note that many of
  // these terms are zero, but we populate the whole matrix
  // so it can be easily used in formulas.

  int n_tot = x.rows();
  int p_tot = x.cols();
  int k_tot = e_z.cols();

  FullParameterIndices ind(k_tot, p_tot);

  // "tt" stands for "theta theta".
  SparseMatrix<double> h_tt(ind.Dim(), ind.Dim());
  h_tt.setZero();

  if (e_z.rows() != n_tot) {
    Rcpp::Rcout << "e_z has the wrong number of rows.\n";
    return h_tt;
  }

  std::vector<Triplet> t;

  // Calculate the number of terms that will be needed for each
  // set of sensititivies.  

  // Each mu is sensitive to p_tot of the lambda terms.
  // Each quadratic mu term is sensitive to one of the lambda terms.
  int mu_size = p_tot;
  int mu2_size = ind.MatrixSize();
  t.reserve(k_tot * (mu_size + mu2_size));

  // First calculate a few summary statistics from e_z.
  VectorXd z_sum(k_tot);
  z_sum.setZero();

  // The typical (p, k) element of x_z_sum is \sum_n x_n_p * z_n_k
  MatrixXd x_z_sum(p_tot, k_tot);  
  x_z_sum.setZero();

  for (int k = 0; k < k_tot; k++) {
    for (int n = 0; n < n_tot; n++) {
      z_sum(k) += e_z(n , k);
      for (int p = 0; p < p_tot; p++) {
        x_z_sum(p, k) += e_z(n, k) * x(n, p);
      }
    }
  }


  for (int k = 0; k < k_tot; k++) {
    // First the mu sensitivities.
    for (int a = 0; a < p_tot; a++) {
      int mu_index = ind.MuCoord(k, a);
      for (int b = 0; b < p_tot; b++) {
	// This is d2H / dmu_a dLambda_ab
	t.push_back(Triplet(mu_index,
			    ind.LambdaCoord(k, a, b),
			    x_z_sum(b, k)));
	t.push_back(Triplet(ind.LambdaCoord(k, a, b),
			    mu_index,
			    x_z_sum(b, k)));
      }
    }

    // Next the second order mu sensitivities.
    for (int a = 0; a < p_tot; a++) {
      for (int b = 0; b <= a; b++) {
	int mu2_index = ind.Mu2Coord(k, a, b);
	
	double this_sens;
	if (a == b) {
	  this_sens = -0.5 * z_sum(k);
	} else {
	  this_sens = -1.0 * z_sum(k);
	}

	// This is d2H / d(mu_a mu_b) dLambda_ab
	t.push_back(Triplet(mu2_index,
			    ind.LambdaCoord(k, a, b),
			    this_sens));
	t.push_back(Triplet(ind.LambdaCoord(k, a, b),
			    mu2_index,
			    this_sens));
      }
    }     
  }

  h_tt.setFromTriplets(t.begin(), t.end());
  h_tt.makeCompressed();
  return h_tt;
}


SparseMatrix<double> GetHThetaXCore(const MatrixXd e_z,
				    const MatrixXd e_mu,
				    const MatrixXd e_lambda,
				    const VectorXd x_indices,
				    const bool full_x = true) {
  // Theta refers to the complete collection of parameters (mu, lambda, pi).
  // This constructs d^2 H / d theta dx'.  The rows are theta, and the columns
  // are x.
  //
  // If full_x = true, then x_indices is ignored and it gets the sensitivity
  // for the whole x vector.  Otherwise, it gets it only for the rows indexed
  // by x_indices.

  int n_tot = e_z.rows();
  int p_tot = e_mu.rows();
  int k_tot = e_mu.cols();

  int x_tot;
  if (full_x) {
    x_tot = n_tot;
  } else {
    x_tot = x_indices.size();
  }
    
  FullParameterIndices ind(k_tot, p_tot);
  XParameterIndices x_ind(x_tot, p_tot);

  SparseMatrix<double> h_theta_x(ind.Dim(), x_ind.Dim());

  if (e_z.cols() != k_tot) {
    Rcpp::Rcout << "e_z has the wrong number of columns.\n";
    return h_theta_x;
  }
  if (x_tot > n_tot) {
    Rcpp::Rcout << "x_indices is longer than n_tot.\n";
    return h_theta_x;
  }
  if (e_lambda.cols() != k_tot) {
    Rcpp::Rcout << "e_lambda has the wrong number of columns.\n";
    return h_theta_x;
  }
  if (e_lambda.rows() != ind.MatrixSize()) {
    Rcpp::Rcout << "e_lambda has the wrong number of columns.\n";
    return h_theta_x;
  }

  std::vector<Triplet> t;

  // Calculate the number of terms that will be needed for each
  // set of sensititivies.
  // Each first-order x_ij term is sensitive to
  // each of the first-order mu and each of the lambda parameters.
  // Each second-order x_ij term is sensitive only to the lambda
  // parameters.
  int mu_size = p_tot;
  int lambda_size = ind.MatrixSize();
  t.reserve(x_tot * p_tot * (mu_size + lambda_size) +
	    x_tot * x_ind.MatrixSize() * lambda_size);

  // We will loop over the x indices in an outer loop.
  for (int x_ind_row = 0; x_ind_row < x_tot; x_ind_row++) {
    int n;
    if (full_x) {
      n = x_ind_row;
    } else {
      n = x_indices(x_ind_row);
    }

    if (n >= n_tot) {
      Rcpp::Rcout << "Error: an x_index is greater than n_tot.\n";
      return h_theta_x;
    }

    if (n >= n_tot) {
      Rcpp::Rcout << "Error: an x index is greater than n_tot\n";
      return h_theta_x;
    }
    
    // Loop over the p and k dimensions to set the sensitivities.
    for (int a = 0; a < p_tot; a++) {
      int this_x_index = x_ind.XCoord(x_ind_row, a);
      for (int k = 0; k < k_tot; k++) {
	for (int b = 0; b < p_tot; b++) {
	  double this_z = e_z(n, k);
	  int this_ab_index = GetUpperTriangularIndex(a, b);

	  // Sensitivity of x to the first order means.
	  t.push_back(Triplet(ind.MuCoord(k, b),
			      this_x_index,
			      this_z * e_lambda(this_ab_index, k)));

	  // Sensitivity of x to the lambda.
	  t.push_back(Triplet(ind.LambdaCoord(k, b, a),
			      this_x_index,
			      this_z * e_mu(b, k)));

	  // Sensitivity of xa_xb to lambda.
	  int this_x2_index = x_ind.X2Coord(x_ind_row, a, b);
	  if (a == b) {
	    t.push_back(Triplet(ind.LambdaCoord(k, b, a),
				this_x2_index,
				-0.5 * this_z));
	  } else if (a < b) {
	    // Only when a < b so as not to double count.
	    t.push_back(Triplet(ind.LambdaCoord(k, b, a),
				this_x2_index,
				-1.0 * this_z));
	  }
	} // b loop
      } // k loop
    } // a loop
  }

  h_theta_x.setFromTriplets(t.begin(), t.end());
  h_theta_x.makeCompressed();
  return h_theta_x;
}


// [[Rcpp::export]]
SparseMatrix<double> GetHThetaX(const MatrixXd e_z,
				const MatrixXd e_mu,
				const MatrixXd e_lambda) {
  VectorXd empty_x(1);
  return GetHThetaXCore(e_z, e_mu, e_lambda, empty_x, true);
}


// [[Rcpp::export]]
SparseMatrix<double> GetHThetaXSubset(const MatrixXd e_z,
				      const MatrixXd e_mu,
				      const MatrixXd e_lambda,
				      const VectorXd x_indices) {
  return GetHThetaXCore(e_z, e_mu, e_lambda, x_indices, false);
}


SparseMatrix<double> GetHZXCore(const int n_tot,
				const MatrixXd e_mu,
				const MatrixXd e_lambda,
				const VectorXd x_indices,
				const bool full_x) {
  // This constructs d^2 H / dz dx'.  The rows are z, and the columns
  // are x.
  //
  // If full_x = true, then x_indices is ignored and it gets the sensitivity
  // for the whole x vector.  Otherwise, it gets it only for the rows indexed
  // by x_indices.


  int p_tot = e_mu.rows();
  int k_tot = e_mu.cols();
  
  int x_tot;
  if (full_x) {
    x_tot = n_tot;
  } else {
    x_tot = x_indices.size();
  }

  ZParameterIndices z_ind(n_tot, k_tot);
  XParameterIndices x_ind(x_tot, p_tot);
  
  SparseMatrix<double> h_z_x(z_ind.Dim(), x_ind.Dim());

  if (e_lambda.cols() != k_tot) {
    Rcpp::Rcout << "e_lambda has the wrong number of columns.\n";
    return h_z_x;
  }
  if (e_lambda.rows() != x_ind.MatrixSize()) {
    Rcpp::Rcout << "e_lambda has the wrong number of columns.\n";
    return h_z_x;
  }
  if (x_tot > n_tot) {
    Rcpp::Rcout << "x_indices is longer than n_tot.\n";
    return h_z_x;
  }

  std::vector<Triplet> t;

  // Calculate the number of terms that will be needed for each
  // set of sensititivies.
  // Each first- and second-order x_ij term is sensitive to k_tot z_ij terms.
  t.reserve(x_ind.Dim() * k_tot);

  // We will loop over the x indices in an outer loop.
    for (int x_ind_row = 0; x_ind_row < x_tot; x_ind_row++) {
    int n;
    if (full_x) {
      n = x_ind_row;
    } else {
      n = x_indices(x_ind_row);
    }

    if (n >= n_tot) {
      Rcpp::Rcout << "Error: an x_index is greater than n_tot.\n";
      return h_z_x;
    }
    
    // Loop over the p and k dimensions to set the sensitivities.
    for (int a = 0; a < p_tot; a++) {
      int this_x_index = x_ind.XCoord(x_ind_row, a);
      for (int k = 0; k < k_tot; k++) {
	int this_z_index = z_ind.ZCoord(n, k);
	
	double this_sens = 0.0;
	for (int b = 0; b < p_tot; b++) {
	  this_sens += e_mu(b, k) * e_lambda(GetUpperTriangularIndex(a, b), k);
	}
	// Sensitivity of x.
	t.push_back(Triplet(this_z_index,
			    this_x_index,
			    this_sens));

	for (int b = 0; b <= a ; b++) {
	  // Sensitivity of xa_xb.
	  double this_lambda = e_lambda(GetUpperTriangularIndex(a, b), k);
	  int this_x2_index = x_ind.X2Coord(x_ind_row, a, b);
	  if (a == b) {
	    t.push_back(Triplet(this_z_index,
				this_x2_index,
				-0.5 * this_lambda));
	  } else {
	    t.push_back(Triplet(this_z_index,
				this_x2_index,
				-1.0 * this_lambda));
	  }
	} // b loop
      } // k loop
    } // a loop
  }

  h_z_x.setFromTriplets(t.begin(), t.end());
  h_z_x.makeCompressed();
  return h_z_x;
}

// [[Rcpp::export]]
SparseMatrix<double> GetHZX(const int n_tot,
			    const MatrixXd e_mu,
			    const MatrixXd e_lambda) {
  VectorXd empty_x(1);
  return GetHZXCore(n_tot, e_mu, e_lambda, empty_x, true);
}


// [[Rcpp::export]]
SparseMatrix<double> GetHZXSubset(const int n_tot,
				  const MatrixXd e_mu,
				  const MatrixXd e_lambda,
				  const VectorXd x_indices) {
  VectorXd empty_x(1);
  return GetHZXCore(n_tot, e_mu, e_lambda, x_indices, false);
}



// [[Rcpp::export]]
VectorXd PredictPerturbationEffect(MatrixXd x,
				   MatrixXd tx_cov,
				   int x_n, int x_p,
				   double delta) {
  // NB: after more careful thinking, I think this is wrong and should not be used.
  //
  // Predict the effects on theta by changing the datapoint
  // x(n, p) by delta given the covariance between theta and x.
  // Args:
  //   - x: An n by p matrix containing the data.
  //   - tx_cov: The estimated covariance between theta and the vectorized
  //     x parameter.  Theta is in the rows and x in the columns.
  //   - x_n: The zero-indexed row of the x matrix to be changed.
  //   - x_p: The zero-indexed column of the x matrix to be changed.
  //   - delta: The amount of the change.
  //
  // Returns:
  //   A vector of the estimated effects on theta by perturbing x.

  int n_tot = x.rows();
  int p_tot = x.cols();
  
  // This is the total dimension of theta.
  int t_tot = tx_cov.rows();
  VectorXd sensitivity(t_tot);
  XParameterIndices x_ind(n_tot, p_tot);
  if (tx_cov.cols() != x_ind.Dim()) {
    Rcpp::Rcout << "Wrong number of columns for tx_cov.\n";
    return sensitivity;
  }

  for (int t = 0; t < t_tot; t++) {

    // The change due to changing x(n, p) directly.
    double this_sens = tx_cov(t, x_ind.XCoord(x_n, x_p)) * delta;

    // Add the changes due to the quadratic terms.
    for (int p = 0; p < p_tot; p++) {
      double x_diff;
      if (p == x_p) {
        x_diff = 2 * delta * x(x_n, x_p) + delta * delta;
      } else {
	x_diff = delta * x(x_n, p);
      }
      this_sens += tx_cov(t, x_ind.X2Coord(x_n, x_p, p)) * x_diff;
    }

    sensitivity(t) = this_sens;
  }
  return sensitivity;
}


// [[Rcpp::export]]
MatrixXd GetLRVBCorrectionTerm(MatrixXd x,
			       MatrixXd e_mu,
			       MatrixXd e_mu2,
			       MatrixXd e_lambda,
			       MatrixXd e_z,
			       SparseMatrix<double> theta_cov,
			       bool verbose=false) {
  if (verbose) {
    Rcpp::Rcout << "Getting htz\n";
  }
  SparseMatrix<double> htz = GetHThetaZ(x, e_mu, e_mu2, e_lambda);

  if (verbose) {
    Rcpp::Rcout << "Getting htt\n";
  }
  SparseMatrix<double> htt = GetHThetaTheta(x, e_z);

  if (verbose) {
    Rcpp::Rcout << "Getting z_cov\n";
  }
  SparseMatrix<double> z_cov = GetZCovariance(e_z);
  MatrixXd t_identity(htt.rows(), htt.cols());
  t_identity.setIdentity();

  if (verbose) {
    Rcpp::Rcout << "Getting rttx\n";
  }
  SparseMatrix<double> rtt = theta_cov * htt;

  if (verbose) {
    Rcpp::Rcout << "Getting htz_rzt\n";
  }
  SparseMatrix<double> htz_rzt = htz * z_cov * htz.transpose();

  if (verbose) {
    Rcpp::Rcout << "Getting rtz_rzt\n";
  }
  SparseMatrix<double> rtz_rzt = theta_cov * htz_rzt;

  if (verbose) {
    Rcpp::Rcout << "Getting lrvb_correction term 1\n";
  }
  MatrixXd lrvb_correction = MatrixXd(rtt) + MatrixXd(rtz_rzt);

  if (verbose) {
    Rcpp::Rcout << "Getting lrvb_correction term 2\n";
  }
  lrvb_correction = t_identity - lrvb_correction;

  return lrvb_correction;
}


// [[Rcpp::export]]
MatrixXd CPPGetLRVBCovariance(MatrixXd x,
			      MatrixXd e_mu,
			      MatrixXd e_mu2,
			      MatrixXd e_lambda,
			      MatrixXd e_z,
			      SparseMatrix<double> theta_cov,
			      bool verbose=false) {

  MatrixXd lrvb_correction = GetLRVBCorrectionTerm(x, e_mu, e_mu2, e_lambda,
						   e_z, theta_cov, verbose);
  if (verbose) {
    Rcpp::Rcout << "Getting householder QR\n";
  }
  HouseholderQR<MatrixXd> lrvb_correction_inv = lrvb_correction.householderQr();
  if (verbose) {
    Rcpp::Rcout << "Getting lrvb_cov\n";
  }
  MatrixXd lrvb_cov = lrvb_correction_inv.solve(MatrixXd(theta_cov));

  return lrvb_cov;
}

// [[Rcpp::export]]
MatrixXd CPPGetLRVBCovarianceFromCorrection(MatrixXd lrvb_correction,
 					    SparseMatrix<double> theta_cov,
					    bool verbose=false) {
  if (verbose) {
    Rcpp::Rcout << "Getting householder QR\n";
  }
  HouseholderQR<MatrixXd> lrvb_correction_inv = lrvb_correction.householderQr();
  if (verbose) {
    Rcpp::Rcout << "Getting lrvb_cov\n";
  }
  MatrixXd lrvb_cov = lrvb_correction_inv.solve(MatrixXd(theta_cov));

  return lrvb_cov;
}



// [[Rcpp::export]]
MatrixXd CPPGetLeverageScores(SparseMatrix<double> z_cov,
			      SparseMatrix<double> x_cov,
			      SparseMatrix<double> htx,
			      SparseMatrix<double> htz,
			      SparseMatrix<double> hzx,
			      MatrixXd lrvb_correction,
			      SparseMatrix<double> theta_cov,
			      bool verbose=false) {
  if (verbose) {
    Rcpp::Rcout << "Getting htz_rzx\n";
  }
  SparseMatrix<double> htz_rzx = htz * z_cov * hzx;

  if (verbose) {
    Rcpp::Rcout << "Getting inner term\n";
  }
  MatrixXd inner_term = MatrixXd(htx) + MatrixXd(htz_rzx);

  if (verbose) {
    Rcpp::Rcout << "Multiplying inner term by x_cov\n";
  }  
  inner_term = inner_term * MatrixXd(x_cov);

  if (verbose) {
    Rcpp::Rcout << "Multiplying inner term by theta_cov\n";
  }  
  inner_term = MatrixXd(theta_cov) * inner_term;

  if (verbose) {
    Rcpp::Rcout << "Getting householder QR\n";
  }
  HouseholderQR<MatrixXd> lrvb_correction_inv = lrvb_correction.householderQr();

  if (verbose) {
    Rcpp::Rcout << "Multiplying by lrvb_correction_inv\n";
  }
  MatrixXd tx_cov = lrvb_correction_inv.solve(inner_term);
  return tx_cov;
}
