#pragma once

#include <Eigen/Dense>

class SquareLattice;

class Measurement {
private:
    int num_sites_;
    int num_samples_;

    Eigen::VectorXd density_up_;
    Eigen::VectorXd density_down_;
    Eigen::VectorXd double_occ_;
    Eigen::VectorXd local_moment_;

    double nn_spin_corr_sum_;
    double af_structure_sum_;

public:
    explicit Measurement(int num_sites);

    void sample(
        const Eigen::MatrixXd& G_up,
        const Eigen::MatrixXd& G_down,
        const SquareLattice& lattice
    );

    void normalize();

    double get_avg_density() const;
    double get_avg_double_occ() const;
    double get_avg_local_moment() const;

    double get_nearest_neighbor_spin_corr() const;
    double get_af_structure_factor() const;

    int get_num_samples() const;

    void print_summary() const;
};