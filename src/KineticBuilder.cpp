#include "KineticBuilder.hpp"
#include <vector>

// ============================================================================
// Dense Matrix Representation
// ============================================================================
Eigen::MatrixXd KineticBuilder::buildKineticMatrix(
    const ILattice& lattice, 
    double t, 
    double mu, 
    double delta_tau
) {
    int N = lattice.num_sites();
    Eigen::MatrixXd k = Eigen::MatrixXd::Zero(N, N);

    for (int i = 0; i < N; ++i) {
        k(i, i) = -delta_tau * mu;

        std::vector<int> neighbors = lattice.get_neighbors(i);
        for (int neighbor : neighbors) {
            k(i, neighbor) = -delta_tau * t;
        }
    }
    return k;
}

// ============================================================================
// Sparse Matrix Representation
// ============================================================================
Eigen::SparseMatrix<double> KineticBuilder::buildKineticMatrixSparse(
    const ILattice& lattice, 
    double t, 
    double mu, 
    double delta_tau
) {
    int N = lattice.num_sites();
    Eigen::SparseMatrix<double> k(N, N);
    
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(N * 5); 

    for (int i = 0; i < N; ++i) {
        tripletList.push_back(Eigen::Triplet<double>(i, i, -delta_tau * mu));

        std::vector<int> neighbors = lattice.get_neighbors(i);
        for (int neighbor : neighbors) {
            tripletList.push_back(Eigen::Triplet<double>(i, neighbor, -delta_tau * t));
        }
    }

    k.setFromTriplets(tripletList.begin(), tripletList.end());
    return k;
}