#pragma once
#include <random>
#include "InteractionConfig.hpp"
#include "GreensEngine.hpp"

class DqmcSampler {
private:
    int num_sites_;
    int num_slices_;
    Eigen::MatrixXd exp_k_;    // Precomputed exp(k)
    Eigen::MatrixXd exp_neg_k_;// Precomputed exp(-k) for wrapping
    
    std::mt19937 rng_;
    std::uniform_real_distribution<double> dist_;

public:
    DqmcSampler(int num_sites, int num_slices, const Eigen::MatrixXd& k, unsigned int seed = 42);

    // Performs one full space-time sweep across all slices and sites
    void perform_sweep(InteractionConfig& config, GreensEngine& engine);
};