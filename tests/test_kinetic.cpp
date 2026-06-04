#include <iostream>
#include <cassert>
#include <cmath>
#include "SquareLattice.hpp"
#include "KineticBuilder.hpp"

void test_kinetic_matrix_2x1_open() {
    std::cout << "Running test_kinetic_matrix_2x1_open... ";

    // Create a 2x1 lattice with Open Boundary Conditions (periodic = false)
    // Site 0 is at (0,0), Site 1 is at (1,0)
    SquareLattice lattice(2, 1, false);

    double t = 1.0;
    double mu = 0.5;
    double delta_tau = 0.1;

    // Expected matrix values:
    // Diagonal: -\Delta\tau * \mu = -0.1 * 0.5 = -0.05
    // Off-diagonal: -\Delta\tau * t = -0.1 * 1.0 = -0.10
    Eigen::MatrixXd k = KineticBuilder::buildKineticMatrix(lattice, t, mu, delta_tau);

    // Verify matrix size is N x N (2 x 2)
    assert(k.rows() == 2);
    assert(k.cols() == 2);

    // Check diagonal elements
    assert(std::abs(k(0, 0) - (-0.05)) < 1e-9);
    assert(std::abs(k(1, 1) - (-0.05)) < 1e-9);

    // Check hopping connections (0 <-> 1 are neighbors)
    assert(std::abs(k(0, 1) - (-0.10)) < 1e-9);
    assert(std::abs(k(1, 0) - (-0.10)) < 1e-9);

    std::cout << "PASSED!" << std::endl;
}

int main() {
    test_kinetic_matrix_2x1_open();
    std::cout << "Kinetic matrix tests completed successfully." << std::endl;
    return 0;
}