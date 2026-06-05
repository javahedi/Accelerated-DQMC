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

    /// Precompute the only two possible exponential weights
    exp_plus_lambda_  = std::exp(lambda_);
    exp_minus_lambda_ = std::exp(-lambda_);

    // Allocate the space-time grid
    storage_.resize(num_slices_, std::vector<int>(num_sites_, 1));
}


void InteractionConfig::randomize() {
    randomize(42);
}

void InteractionConfig::randomize(unsigned int seed) {
    std::mt19937 gen(seed); // Use the provided seed
    std::uniform_int_distribution<> dis(0, 1);

    for (int l = 0; l < num_slices_; ++l) {
        for (int i = 0; i < num_sites_; ++i) {
            storage_[l][i] = dis(gen) == 0 ? -1 : 1;
        }
    }
}

int InteractionConfig::get_spin(int slice, int site) const {
    return storage_[slice][site];
}

void InteractionConfig::flip_spin(int slice, int site) {
    storage_[slice][site] = -storage_[slice][site];
}

Eigen::VectorXd InteractionConfig::get_exponential_diagonal(int slice, double spin_factor) const {
    Eigen::VectorXd diag(num_sites_);
    
    // Determine which precomputed values correspond to s_il = +1 and s_il = -1
    // If spin_factor == +1.0 (Up):   s=+1 -> e^λ,   s=-1 -> e^-λ
    // If spin_factor == -1.0 (Down): s=+1 -> e^-λ,  s=-1 -> e^λ
    double weight_if_plus_one  = (spin_factor > 0) ? exp_plus_lambda_  : exp_minus_lambda_;
    double weight_if_minus_one = (spin_factor > 0) ? exp_minus_lambda_ : exp_plus_lambda_;

    for (int i = 0; i < num_sites_; ++i) {
        diag(i) = (storage_[slice][i] == 1) ? weight_if_plus_one : weight_if_minus_one;
    }
    
    return diag;
}