#include <iostream>
#include <cassert>
#include <cmath>
#include "SquareLattice.hpp"
#include "KineticBuilder.hpp"
#include "InteractionConfig.hpp"
#include "GreensEngine.hpp"

#include <unsupported/Eigen/MatrixFunctions>

void test_greens_engine_modes() {
    std::cout << "Running test_greens_engine_modes...\n";

    int N = 2; // Linear size
    int num_sites = N * N; // Total spatial sites (4)
    int L = 4;
    double t = 1.0;
    double mu = 0.0;
    double delta_tau = 0.1;
    double U = 0.0; 

    // Build the 2x2 multi-dimensional lattice structure
    SquareLattice lattice(N, N, false);
    Eigen::MatrixXd k = KineticBuilder::buildKineticMatrix(lattice, t, mu, delta_tau);
    
    InteractionConfig config(num_sites, L, U, delta_tau);

    // 1. Compute exact analytical reference limit: G = (I + exp(L * k))^-1
    Eigen::MatrixXd total_k = double(L) * k;
    Eigen::MatrixXd exp_total_k = total_k.exp();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(num_sites, num_sites);
    Eigen::MatrixXd G_exact = (I + exp_total_k).inverse();

    // 2. Test Standard Engine Mode (stabilize = false)
    std::cout << "  -> Testing Standard Mode... ";
    GreensEngine standard_engine(num_sites, L, k, false);
    standard_engine.recompute_all_from_scratch(config);
    Eigen::MatrixXd G_standard = standard_engine.G_up();

    for (int i = 0; i < num_sites; ++i) {
        for (int j = 0; j < num_sites; ++j) {
            assert(std::abs(G_standard(i, j) - G_exact(i, j)) < 1e-9);
        }
    }
    std::cout << "PASSED!\n";

    // 3. Test Stabilized Engine Mode (stabilize = true)
    std::cout << "  -> Testing Stabilized UDV Mode... ";
    GreensEngine stabilized_engine(num_sites, L, k, true, 2); 
    stabilized_engine.recompute_all_from_scratch(config);
    Eigen::MatrixXd G_stabilized = stabilized_engine.G_up();

    for (int i = 0; i < num_sites; ++i) {
        for (int j = 0; j < num_sites; ++j) {
            assert(std::abs(G_stabilized(i, j) - G_exact(i, j)) < 1e-9);
        }
    }
    std::cout << "PASSED!\n";
}

int main() {
    test_greens_engine_modes();
    std::cout << "All Green's Engine verification tests completed successfully." << std::endl;
    return 0;
}