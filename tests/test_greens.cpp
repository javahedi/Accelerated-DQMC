#include <iostream>
#include <cassert>
#include <cmath>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "InteractionConfig.hpp"
#include "GreensEngine.hpp"

bool is_close(double val, double target, double tol = 1e-9) {
    return std::abs(val - target) < tol;
}

bool matrix_close(
    const Eigen::MatrixXd& A,
    const Eigen::MatrixXd& B,
    double tol = 1e-8
) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        return false;
    }

    return (A - B).norm() < tol;
}

Eigen::MatrixXd manual_B_product(
    const Eigen::MatrixXd& k,
    const InteractionConfig& config,
    int N,
    int L,
    double spin_factor
) {
    Eigen::MatrixXd exp_k = k.exp();
    Eigen::MatrixXd product = Eigen::MatrixXd::Identity(N, N);

    for (int l = 0; l < L; ++l) {
        Eigen::VectorXd exp_v_diag =
            config.get_exponential_diagonal(l, spin_factor);

        Eigen::MatrixXd B_l = exp_k;

        for (int col = 0; col < N; ++col) {
            B_l.col(col) *= exp_v_diag(col);
        }

        // Must match GreensEngine convention:
        // B_total = B_{L-1} ... B_1 B_0
        product = B_l * product;
    }

    return product;
}

void test_greens_engine_initialization() {
    std::cout << "Running test_greens_engine_initialization... ";

    int N = 4;
    int L = 6;

    Eigen::MatrixXd k = Eigen::MatrixXd::Zero(N, N);

    GreensEngine engine(N, L, k, false);

    assert(engine.G_up().rows() == N);
    assert(engine.G_up().cols() == N);
    assert(engine.G_down().rows() == N);
    assert(engine.G_down().cols() == N);

    assert(matrix_close(engine.G_up(), Eigen::MatrixXd::Identity(N, N)));
    assert(matrix_close(engine.G_down(), Eigen::MatrixXd::Identity(N, N)));

    std::cout << "PASSED!" << std::endl;
}

void test_B_product_identity_kinetic() {
    std::cout << "Running test_B_product_identity_kinetic... ";

    int N = 3;
    int L = 5;

    double U = 4.0;
    double delta_tau = 0.1;

    Eigen::MatrixXd k = Eigen::MatrixXd::Zero(N, N);

    InteractionConfig config(N, L, U, delta_tau);
    GreensEngine engine(N, L, k, false);

    Eigen::MatrixXd B_up =
        engine.compute_B_product(config, 1.0);

    Eigen::MatrixXd B_up_manual =
        manual_B_product(k, config, N, L, 1.0);

    assert(matrix_close(B_up, B_up_manual));

    std::cout << "PASSED!" << std::endl;
}

void test_recompute_greens_against_manual_inverse() {
    std::cout << "Running test_recompute_greens_against_manual_inverse... ";

    int N = 4;
    int L = 8;

    double U = 4.0;
    double delta_tau = 0.1;

    Eigen::MatrixXd k = Eigen::MatrixXd::Zero(N, N);

    // Small hopping-like off-diagonal terms.
    k(0, 1) = -0.1;
    k(1, 0) = -0.1;
    k(1, 2) = -0.1;
    k(2, 1) = -0.1;
    k(2, 3) = -0.1;
    k(3, 2) = -0.1;

    InteractionConfig config(N, L, U, delta_tau);

    config.flip_spin(0, 0);
    config.flip_spin(2, 1);
    config.flip_spin(4, 3);

    GreensEngine engine(N, L, k, false);
    engine.recompute_all_from_scratch(config);

    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(N, N);

    Eigen::MatrixXd B_up =
        manual_B_product(k, config, N, L, 1.0);

    Eigen::MatrixXd B_down =
        manual_B_product(k, config, N, L, -1.0);

    Eigen::MatrixXd G_up_expected =
        (I + B_up).partialPivLu().solve(I);

    Eigen::MatrixXd G_down_expected =
        (I + B_down).partialPivLu().solve(I);

    assert(matrix_close(engine.G_up(), G_up_expected, 1e-8));
    assert(matrix_close(engine.G_down(), G_down_expected, 1e-8));

    std::cout << "PASSED!" << std::endl;
}

void test_stabilized_matches_unstabilized_small_system() {
    std::cout << "Running test_stabilized_matches_unstabilized_small_system... ";

    int N = 4;
    int L = 10;

    double U = 4.0;
    double delta_tau = 0.1;

    Eigen::MatrixXd k = Eigen::MatrixXd::Zero(N, N);

    k(0, 1) = -0.1;
    k(1, 0) = -0.1;
    k(1, 2) = -0.1;
    k(2, 1) = -0.1;
    k(2, 3) = -0.1;
    k(3, 2) = -0.1;

    InteractionConfig config(N, L, U, delta_tau);

    config.flip_spin(0, 0);
    config.flip_spin(1, 2);
    config.flip_spin(3, 1);
    config.flip_spin(7, 3);

    GreensEngine plain_engine(N, L, k, false);
    GreensEngine stabilized_engine(N, L, k, true, 2);

    plain_engine.recompute_all_from_scratch(config);
    stabilized_engine.recompute_all_from_scratch(config);

    assert(matrix_close(
        plain_engine.G_up(),
        stabilized_engine.G_up(),
        1e-7
    ));

    assert(matrix_close(
        plain_engine.G_down(),
        stabilized_engine.G_down(),
        1e-7
    ));

    std::cout << "PASSED!" << std::endl;
}

void test_setters() {
    std::cout << "Running test_setters... ";

    int N = 3;
    int L = 4;

    Eigen::MatrixXd k = Eigen::MatrixXd::Zero(N, N);
    GreensEngine engine(N, L, k, false);

    Eigen::MatrixXd G_up = 2.0 * Eigen::MatrixXd::Identity(N, N);
    Eigen::MatrixXd G_down = 3.0 * Eigen::MatrixXd::Identity(N, N);

    engine.set_G_up(G_up);
    engine.set_G_down(G_down);

    assert(matrix_close(engine.G_up(), G_up));
    assert(matrix_close(engine.G_down(), G_down));

    std::cout << "PASSED!" << std::endl;
}

int main() {
    test_greens_engine_initialization();
    test_B_product_identity_kinetic();
    test_recompute_greens_against_manual_inverse();
    test_stabilized_matches_unstabilized_small_system();
    test_setters();

    std::cout << "All GreensEngine tests completed successfully." << std::endl;

    return 0;
}