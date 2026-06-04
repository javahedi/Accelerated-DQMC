#pragma once
#include <Eigen/Dense>
#include "InteractionConfig.hpp"

class GreensEngine {
private:
    int num_sites_;
    int num_slices_;
    bool stabilize_;              // Flag to switch between standard and stabilized calculation
    int stabilization_frequency_;  // Frequency of QR decompositions if stabilized
    Eigen::MatrixXd exp_k_;        // Precomputed matrix exponential: exp(k)

    // Current equal-time Green's functions for up and down spins
    Eigen::MatrixXd G_up_;
    Eigen::MatrixXd G_down_;

    // --- Persistent Pre-allocated Workspaces for Stabilization ---
    mutable Eigen::MatrixXd U_workspace_;
    mutable Eigen::VectorXd D_workspace_;
    mutable Eigen::MatrixXd V_workspace_;
    mutable Eigen::MatrixXd M_workspace_;
    mutable Eigen::MatrixXd target_workspace_;
    mutable Eigen::MatrixXd V_inv_workspace_;
    mutable Eigen::MatrixXd inner_workspace_;

    // Private helper that calculates the stabilized Green's function via UDV factorization
    Eigen::MatrixXd compute_stabilized_greens(const InteractionConfig& config, double spin_factor) const;

public:
    // By defaulting stabilize to false, your existing tests won't notice any change
    GreensEngine(int num_sites, int num_slices, const Eigen::MatrixXd& k, bool stabilize = false, int stabilization_frequency = 10);

    // Computes the original full chain matrix product B_1 * B_2 * ... * B_L (Kept intact!)
    Eigen::MatrixXd compute_B_product(const InteractionConfig& config, double spin_factor) const;

    // Fully recomputes G_up and G_down from scratch using the requested mode
    void recompute_all_from_scratch(const InteractionConfig& config);

    // Read accessors for the Green's functions
    const Eigen::MatrixXd& G_up() const { return G_up_; }
    const Eigen::MatrixXd& G_down() const { return G_down_; }

    void set_G_up(const Eigen::MatrixXd& G) { G_up_ = G; }
    void set_G_down(const Eigen::MatrixXd& G) { G_down_ = G; }
};