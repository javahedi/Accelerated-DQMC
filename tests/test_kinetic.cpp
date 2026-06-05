#include <iostream>
#include <cassert>
#include <cmath>
#include "SquareLattice.hpp"
#include "KineticBuilder.hpp"

// Helper function to check close floating-point equality
bool is_close(double val, double target, double tol = 1e-9) {
    return std::abs(val - target) < tol;
}

void test_kinetic_matrix_2x1_open() {
    std::cout << "Running test_kinetic_matrix_2x1_open... ";

    SquareLattice lattice(2, 1, false);
    double t = 1.0;
    double mu = 0.5;
    double delta_tau = 0.1;

    Eigen::MatrixXd k = KineticBuilder::buildKineticMatrix(lattice, t, mu, delta_tau);

    assert(k.rows() == 2);
    assert(k.cols() == 2);

    assert(is_close(k(0, 0), -0.05));
    assert(is_close(k(1, 1), -0.05));
    assert(is_close(k(0, 1), -0.10));
    assert(is_close(k(1, 0), -0.10));

    std::cout << "PASSED!" << std::endl;
}

void test_kinetic_matrix_4x4_dense_periodic() {
    std::cout << "Running test_kinetic_matrix_4x4_dense_periodic... ";

    int width = 4;
    int height = 4;
    int N = width * height; // 16 sites
    SquareLattice lattice(width, height, true);

    double t = 1.0;
    double mu = 0.5;
    double delta_tau = 0.1;

    double expected_diag = -delta_tau * mu; // -0.05
    double expected_hop  = -delta_tau * t;  // -0.10

    Eigen::MatrixXd k = KineticBuilder::buildKineticMatrix(lattice, t, mu, delta_tau);

    assert(k.rows() == N);
    assert(k.cols() == N);

    // Loop over all sites to verify the dense matrix topology
    for (int i = 0; i < N; ++i) {
        // 1. Verify chemical potential on the diagonal
        assert(is_close(k(i, i), expected_diag));

        // 2. Count hopping terms in row 'i' to ensure no over/under-counting
        int hopping_count = 0;
        for (int j = 0; j < N; ++j) {
            if (i != j && is_close(k(i, j), expected_hop)) {
                hopping_count++;
            }
        }
        // In a true 2D periodic square lattice, each site must couple to exactly 4 distinct neighbors
        assert(hopping_count == 4);
    }

    std::cout << "PASSED!" << std::endl;
}

void test_kinetic_matrix_10x10_sparse_periodic() {
    std::cout << "Running test_kinetic_matrix_10x10_sparse_periodic... ";

    int width = 10;
    int height = 10;
    int N = width * height; // 100 sites
    SquareLattice lattice(width, height, true);

    double t = 1.2;
    double mu = -0.3;
    double delta_tau = 0.05;

    double expected_diag = -delta_tau * mu; // -0.05 * (-0.3) = 0.015
    double expected_hop  = -delta_tau * t;  // -0.05 * 1.2 = -0.06

    Eigen::SparseMatrix<double> k = KineticBuilder::buildKineticMatrixSparse(lattice, t, mu, delta_tau);

    // 1. Verify structural dimensions
    assert(k.rows() == N);
    assert(k.cols() == N);

    // 2. Verify total allocated non-zero elements
    // Each of the 100 sites has 1 diagonal element + 4 neighbor bonds = 5 entries per row.
    // Total non-zeros should be exactly N * 5 = 500.
    assert(k.nonZeros() == N * 5);

    // 3. Spot check a boundary coordinate to verify periodic wrapping via sparse coefficients
    // Site at bottom-left corner: (x=0, y=0) -> index 0
    // Neighbors should be:
    //   Right: (1,0) -> index 1
    //   Left:  (9,0) -> index 9  (wrapped)
    //   Up:    (0,1) -> index 10
    //   Down:  (0,9) -> index 90 (wrapped)
    assert(is_close(k.coeff(0, 0),  expected_diag));
    assert(is_close(k.coeff(0, 1),  expected_hop));
    assert(is_close(k.coeff(0, 9),  expected_hop));
    assert(is_close(k.coeff(0, 10), expected_hop));
    assert(is_close(k.coeff(0, 90), expected_hop));

    // Ensure non-neighbor elements remain zero
    assert(is_close(k.coeff(0, 2),  0.0));
    assert(is_close(k.coeff(0, 50), 0.0));

    std::cout << "PASSED!" << std::endl;
}

int main() {
    test_kinetic_matrix_2x1_open();
    test_kinetic_matrix_4x4_dense_periodic();
    test_kinetic_matrix_10x10_sparse_periodic();

    std::cout << "All kinetic matrix tests completed successfully." << std::endl;
    return 0;
}