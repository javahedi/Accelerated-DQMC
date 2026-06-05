#include <iostream>
#include <cassert>
#include <cmath>
#include "InteractionConfig.hpp"

void test_interaction_config() {
    std::cout << "Running test_interaction_config... ";

    int N = 4;
    int L = 10;
    double U = 4.0;
    double delta_tau = 0.1;

    InteractionConfig config(N, L, U, delta_tau);

    // Test default initialization (all spins initialized to 1) using (slice, site)
    assert(config.get_spin(0, 0) == 1);

    // Test flipping spin at slice 0, site 0
    config.flip_spin(0, 0);
    assert(config.get_spin(0, 0) == -1);

    // Verify exponential calculation for spin up (factor = +1.0) on slice 0
    Eigen::VectorXd exp_diag = config.get_exponential_diagonal(0, 1.0);

    double expected_lambda = std::acosh(std::exp((U * delta_tau) / 2.0));
    
    // Site 0 is flipped to -1: exp(1.0 * lambda * -1) = exp(-lambda)
    assert(std::abs(exp_diag(0) - std::exp(-expected_lambda)) < 1e-9);
    // Site 1 remains +1: exp(1.0 * lambda * 1) = exp(lambda)
    assert(std::abs(exp_diag(1) - std::exp(expected_lambda)) < 1e-9);

    std::cout << "PASSED!" << std::endl;
}

int main() {
    test_interaction_config();
    std::cout << "Interaction module tests completed successfully." << std::endl;
    return 0;
}