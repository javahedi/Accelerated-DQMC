#include "InteractionConfig.hpp"
#include <cmath>
#include <random>
#include <stdexcept>

InteractionConfig::InteractionConfig(int num_sites, int num_slices, double U, double delta_tau)
    : num_sites_(num_sites), num_slices_(num_slices) {
    
    // Calculate lambda: cosh(lambda) = exp(U * delta_tau / 2)
    // Therefore, lambda = acosh(exp(U * delta_tau / 2))
    double exponent = (U * delta_tau) / 2.0;
    lambda_ = std::acosh(std::exp(exponent));

    // Allocate the space-time grid
    storage_.resize(num_slices_, std::vector<int>(num_sites_, 1));
}

void InteractionConfig::randomize() {
    std::mt19937 gen(42); // Fixed seed for reproducible tests
    std::uniform_int_distribution<> dis(0, 1);

    for (int l = 0; l < num_slices_; ++l) {
        for (int i = 0; i < num_sites_; ++i) {
            storage_[l][i] = dis(gen) == 0 ? -1 : 1;
        }
    }
}

int InteractionConfig::get_spin(int site, int slice) const {
    return storage_[slice][site];
}

void InteractionConfig::flip_spin(int site, int slice) {
    storage_[slice][site] = -storage_[slice][site];
}

Eigen::VectorXd InteractionConfig::get_exponential_diagonal(int slice, double spin_factor) const {
    Eigen::VectorXd diag(num_sites_);
    for (int i = 0; i < num_sites_; ++i) {
        // According to the formula, we need the matrix exp(v_sigma)
        // Since v_sigma is diagonal, exp(v_sigma)_{ii} = exp(spin_factor * lambda * s(i, l))
        double argument = spin_factor * lambda_ * storage_[slice][i];
        diag(i) = std::exp(argument);
    }
    return diag;
}