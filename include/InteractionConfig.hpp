#pragma once
#include <Eigen/Core>
#include <vector>

class InteractionConfig {
private:
    int num_sites_;
    int num_slices_;
    double lambda_;

    // New precomputed identifiers declared here:
    double exp_plus_lambda_;  
    double exp_minus_lambda_;
    
    // 2D storage: storage_[l][i] holds the spin at slice l, site i
    std::vector<std::vector<int>> storage_;

public:
    InteractionConfig(int num_sites, int num_slices, double U, double delta_tau);

    // Randomizes all spins to +1 or -1 (for initialization)
    void randomize();
    void randomize(unsigned int seed);

    // Gets the spin value at a specific space-time point
    int get_spin(int site, int slice) const;

    // Flips the spin at a specific space-time point (s -> -s)
    void flip_spin(int site, int slice);

    // Returns the diagonal elements of exp(v_sigma(l)) as an Eigen::VectorXd
    // spin_factor should be +1 for Spin-Up, and -1 for Spin-Down
    Eigen::VectorXd get_exponential_diagonal(int slice, double spin_factor) const;
};