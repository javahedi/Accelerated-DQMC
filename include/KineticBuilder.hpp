#pragma once
#include "ILattice.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>

#pragma once
#include "ILattice.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse> // Added for SparseMatrix

class KineticBuilder
{
public:
    // Dense implementation
    static Eigen::MatrixXd buildKineticMatrix(
        const ILattice& lattice,
        double t,
        double mu,
        double delta_tau);

    // Sparse implementation (Added to class scope)
    static Eigen::SparseMatrix<double> buildKineticMatrixSparse(
        const ILattice& lattice,
        double t,
        double mu,
        double delta_tau);
};