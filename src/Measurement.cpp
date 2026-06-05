#include "Measurement.hpp"
#include "SquareLattice.hpp"

#include <iostream>
#include <stdexcept>
#include <cmath>

Measurement::Measurement(int num_sites)
    : num_sites_(num_sites),
      num_samples_(0),
      density_up_(Eigen::VectorXd::Zero(num_sites)),
      density_down_(Eigen::VectorXd::Zero(num_sites)),
      double_occ_(Eigen::VectorXd::Zero(num_sites)),
      local_moment_(Eigen::VectorXd::Zero(num_sites)),
      nn_spin_corr_sum_(0.0),
      af_structure_sum_(0.0) {

    if (num_sites_ <= 0) {
        throw std::invalid_argument("Measurement: num_sites must be positive.");
    }
}

void Measurement::sample(
    const Eigen::MatrixXd& G_up,
    const Eigen::MatrixXd& G_down,
    const SquareLattice& lattice
) {
    if (G_up.rows() != num_sites_ || G_up.cols() != num_sites_) {
        throw std::runtime_error("Measurement: G_up has wrong dimensions.");
    }

    if (G_down.rows() != num_sites_ || G_down.cols() != num_sites_) {
        throw std::runtime_error("Measurement: G_down has wrong dimensions.");
    }

    Eigen::VectorXd n_up(num_sites_);
    Eigen::VectorXd n_down(num_sites_);

    for (int i = 0; i < num_sites_; ++i) {
        n_up(i) = 1.0 - G_up(i, i);
        n_down(i) = 1.0 - G_down(i, i);

        double d_occ = n_up(i) * n_down(i);
        double moment = n_up(i) + n_down(i) - 2.0 * d_occ;

        density_up_(i) += n_up(i);
        density_down_(i) += n_down(i);
        double_occ_(i) += d_occ;
        local_moment_(i) += moment;
    }

    double nn_sum = 0.0;
    int nn_bonds = 0;

    for (int i = 0; i < num_sites_; ++i) {
        std::vector<int> neighbors = lattice.get_neighbors(i);

        for (int j : neighbors) {
            if (j <= i) {
                continue;
            }

            double splus_sminus =
                -G_up(j, i) * G_down(i, j);

            nn_sum += splus_sminus;
            nn_bonds++;
        }
    }

    if (nn_bonds > 0) {
        nn_spin_corr_sum_ += nn_sum / nn_bonds;
    }

    double af_sum = 0.0;

    for (int i = 0; i < num_sites_; ++i) {
        int xi = lattice.get_x(i);
        int yi = lattice.get_y(i);

        for (int j = 0; j < num_sites_; ++j) {
            int xj = lattice.get_x(j);
            int yj = lattice.get_y(j);

            int parity = (xi - xj + yi - yj) & 1;
            double phase = (parity == 0) ? 1.0 : -1.0;

            double corr = 0.0;

            if (i == j) {
                corr = local_moment_(i) / static_cast<double>(num_samples_ + 1);
            } else {
                corr = -G_up(j, i) * G_down(i, j);
            }

            af_sum += phase * corr;
        }
    }

    af_structure_sum_ += af_sum / static_cast<double>(num_sites_);

    num_samples_++;
}

void Measurement::normalize() {
    if (num_samples_ == 0) {
        return;
    }

    density_up_ /= num_samples_;
    density_down_ /= num_samples_;
    double_occ_ /= num_samples_;
    local_moment_ /= num_samples_;
    nn_spin_corr_sum_ /= num_samples_;
    af_structure_sum_ /= num_samples_;
}

double Measurement::get_avg_density() const {
    return density_up_.mean() + density_down_.mean();
}

double Measurement::get_avg_double_occ() const {
    return double_occ_.mean();
}

double Measurement::get_avg_local_moment() const {
    return local_moment_.mean();
}

double Measurement::get_nearest_neighbor_spin_corr() const {
    return nn_spin_corr_sum_;
}

double Measurement::get_af_structure_factor() const {
    return af_structure_sum_;
}

int Measurement::get_num_samples() const {
    return num_samples_;
}

void Measurement::print_summary() const {
    std::cout << "========== DQMC MEASUREMENT SUMMARY ==========\n";
    std::cout << " Total Measurement Sweeps      : " << num_samples_ << "\n";
    std::cout << " Average Particle Density      : " << get_avg_density() << "\n";
    std::cout << " Average Double Occupancy      : " << get_avg_double_occ() << "\n";
    std::cout << " Average Local Moment          : " << get_avg_local_moment() << "\n";
    std::cout << " Nearest-Neighbor Spin Corr    : " << get_nearest_neighbor_spin_corr() << "\n";
    std::cout << " AF Structure Factor S(pi,pi)  : " << get_af_structure_factor() << "\n";
    std::cout << "==============================================\n";
}