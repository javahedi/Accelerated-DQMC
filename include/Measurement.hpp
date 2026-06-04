#pragma once
#include <Eigen/Dense>
#include <vector>

class Measurement {
private:
    int num_sites_;
    int num_samples_;

    // Accumulators for observables
    Eigen::VectorXd density_up_;
    Eigen::VectorXd density_down_;
    Eigen::VectorXd double_occ_;
    Eigen::VectorXd local_moment_;

public:
    Measurement(int num_sites);

    // Samples observables using the current equal-time Green's functions at tau = 0
    void sample(const Eigen::MatrixXd& G_up, const Eigen::MatrixXd& G_down);

    // Normalizes accumulated values by the number of measurement sweeps
    void normalize();

    // Getters for the physical results
    double get_avg_density() const;
    double get_avg_double_occ() const;
    double get_avg_local_moment() const;
    
    void print_summary() const;
};