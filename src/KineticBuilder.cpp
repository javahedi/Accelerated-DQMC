#include "KineticBuilder.hpp"

Eigen::MatrixXd KineticBuilder::buildKineticMatrix(
    const ILattice& lattice, 
    double t, 
    double mu, 
    double delta_tau
) {
    int N = lattice.num_sites();
    
    // Initialize an N x N matrix filled with zeros
    Eigen::MatrixXd k = Eigen::MatrixXd::Zero(N, N);

    for (int i = 0; i < N; ++i) {
        // Set diagonal elements: -\Delta\tau * \mu
        k(i, i) = -delta_tau * mu;

        // Set off-diagonal elements for neighbors: -\Delta\tau * t
        std::vector<int> neighbors = lattice.get_neighbors(i);
        for (int neighbor : neighbors) {
            k(i, neighbor) = -delta_tau * t;
        }
    }

    return k;
}