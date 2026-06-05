#pragma once

#include <Eigen/Dense>
#include "InteractionConfig.hpp"

class GreensEngine {
private:
    int num_sites_;
    int num_slices_;
    bool stabilize_;
    int stabilization_frequency_;

    // The caller should pass k = -Δτ K if you want exp(-Δτ K).
    Eigen::MatrixXd exp_k_;

    Eigen::MatrixXd G_up_;
    Eigen::MatrixXd G_down_;

    Eigen::MatrixXd make_B_slice(
        const InteractionConfig& config,
        int slice,
        double spin_factor
    ) const;

    Eigen::MatrixXd compute_stabilized_greens(
        const InteractionConfig& config,
        double spin_factor
    ) const;

public:
    GreensEngine(
        int num_sites,
        int num_slices,
        const Eigen::MatrixXd& k,
        bool stabilize = false,
        int stabilization_frequency = 10
    );

    // Builds B_total = B_{L-1} ... B_1 B_0.
    Eigen::MatrixXd compute_B_product(
        const InteractionConfig& config,
        double spin_factor
    ) const;

    void recompute_all_from_scratch(const InteractionConfig& config);

    const Eigen::MatrixXd& G_up() const { return G_up_; }
    const Eigen::MatrixXd& G_down() const { return G_down_; }

    void set_G_up(const Eigen::MatrixXd& G);
    void set_G_down(const Eigen::MatrixXd& G);
};