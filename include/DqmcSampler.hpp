#pragma once

#include <random>
#include <Eigen/Dense>

#include "InteractionConfig.hpp"
#include "GreensEngine.hpp"

class DqmcSampler {
private:
    int num_sites_;
    int num_slices_;

    Eigen::MatrixXd exp_k_;
    Eigen::MatrixXd exp_neg_k_;

    std::mt19937 rng_;
    std::uniform_real_distribution<double> dist_;

public:
    DqmcSampler(
        int num_sites,
        int num_slices,
        const Eigen::MatrixXd& k,
        unsigned int seed = 42
    );

    void perform_sweep(
        InteractionConfig& config,
        GreensEngine& engine
    );
};