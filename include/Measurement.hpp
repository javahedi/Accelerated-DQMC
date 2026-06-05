#pragma once
#include <Eigen/Dense>
#include <vector>

// Forward declaration of your lattice class
class SquareLattice;

class Measurement {
private:
    int num_sites_;
    int num_samples_;

    // Accumulators for local observables
    Eigen::VectorXd density_up_;
    Eigen::VectorXd density_down_;
    Eigen::VectorXd double_occ_;
    Eigen::VectorXd local_moment_;

    // --- Added Accumulators for Full Matrix Correlations ---
    Eigen::MatrixXd G_up_accum_;
    Eigen::MatrixXd G_down_accum_;

public:
    Measurement(int num_sites);

    // Samples observables using the current equal-time Green's functions at tau = 0
    void sample(const Eigen::MatrixXd& G_up, const Eigen::MatrixXd& G_down);

    // Normalizes accumulated values by the number of measurement sweeps
    void normalize();

    // Getters for the local physical results
    double get_avg_density() const;
    double get_avg_double_occ() const;
    double get_avg_local_moment() const;

    // --- Added Getters for Structural Geometries ---
    double get_nearest_neighbor_spin_corr(const SquareLattice& lattice) const;
    double get_af_structure_factor(const SquareLattice& lattice) const;
    
    void print_summary() const;
};