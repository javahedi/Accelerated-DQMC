#include <iostream>
#include <cassert>
#include <cmath>
#include <Eigen/Dense>

#include "SquareLattice.hpp"
#include "KineticBuilder.hpp"
#include "InteractionConfig.hpp"
#include "GreensEngine.hpp"
#include "DqmcSampler.hpp"

void assert_matrices_close(
    const Eigen::MatrixXd& A,
    const Eigen::MatrixXd& B,
    double tol = 1e-9
) {
    assert(A.rows() == B.rows());
    assert(A.cols() == B.cols());

    double max_diff = 0.0;
    int max_i = 0;
    int max_j = 0;

    for (int i = 0; i < A.rows(); ++i) {
        for (int j = 0; j < A.cols(); ++j) {
            double diff = std::abs(A(i, j) - B(i, j));

            if (diff > max_diff) {
                max_diff = diff;
                max_i = i;
                max_j = j;
            }
        }
    }

    if (max_diff >= tol) {
        std::cerr << "\nMatrix mismatch!\n";
        std::cerr << "max_diff = " << max_diff
                  << " at (" << max_i << ", " << max_j << ")\n";
        std::cerr << "A(" << max_i << "," << max_j << ") = "
                  << A(max_i, max_j) << "\n";
        std::cerr << "B(" << max_i << "," << max_j << ") = "
                  << B(max_i, max_j) << "\n";

        assert(false);
    }
}

void test_sherman_morrison_single_slice() {
    std::cout << "Running test_sherman_morrison_single_slice... ";

    int N = 2;
    int L = 1;
    double t = 1.0;
    double mu = 0.0;
    double delta_tau = 0.1;
    double U = 4.0;

    SquareLattice lattice(N, 1, false);
    Eigen::MatrixXd k =
        KineticBuilder::buildKineticMatrix(lattice, t, mu, delta_tau);

    InteractionConfig config(N, L, U, delta_tau);
    GreensEngine engine(N, L, k);

    engine.recompute_all_from_scratch(config);

    DqmcSampler sampler(N, L, k, 12345);
    sampler.perform_sweep(config, engine);

    GreensEngine reference_engine(N, L, k);
    reference_engine.recompute_all_from_scratch(config);

    assert_matrices_close(engine.G_up(), reference_engine.G_up());
    assert_matrices_close(engine.G_down(), reference_engine.G_down());

    std::cout << "PASSED!" << std::endl;
}

void test_sampler_many_sweeps_multi_slice() {
    std::cout << "Running test_sampler_many_sweeps_multi_slice... ";

    int width = 2;
    int height = 2;
    int N = width * height;
    int L = 8;

    double t = 1.0;
    double mu = 0.0;
    double delta_tau = 0.1;
    double U = 4.0;

    SquareLattice lattice(width, height, true);
    Eigen::MatrixXd k =
        KineticBuilder::buildKineticMatrix(lattice, t, mu, delta_tau);

    InteractionConfig config(N, L, U, delta_tau);
    config.randomize();

    GreensEngine engine(N, L, k);
    engine.recompute_all_from_scratch(config);

    DqmcSampler sampler(N, L, k, 12345);

    for (int sweep = 0; sweep < 100; ++sweep) {
        sampler.perform_sweep(config, engine);

        // Temporary stabilization / refresh
        engine.recompute_all_from_scratch(config);

        GreensEngine reference_engine(N, L, k);
        reference_engine.recompute_all_from_scratch(config);

        assert_matrices_close(engine.G_up(), reference_engine.G_up(), 1e-9);
        assert_matrices_close(engine.G_down(), reference_engine.G_down(), 1e-9);
    }

    std::cout << "PASSED!" << std::endl;
}

int main() {
    test_sherman_morrison_single_slice();
    test_sampler_many_sweeps_multi_slice();

    std::cout << "Sampler tests completed successfully." << std::endl;
    return 0;
}