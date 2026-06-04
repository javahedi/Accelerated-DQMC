#include "Measurement.hpp"
#include <iostream>

Measurement::Measurement(int num_sites)
    : num_sites_(num_sites), num_samples_(0),
      density_up_(Eigen::VectorXd::Zero(num_sites)),
      density_down_(Eigen::VectorXd::Zero(num_sites)),
      double_occ_(Eigen::VectorXd::Zero(num_sites)),
      local_moment_(Eigen::VectorXd::Zero(num_sites)) {}

void Measurement::sample(const Eigen::MatrixXd& G_up, const Eigen::MatrixXd& G_down) {
    for (int i = 0; i < num_sites_; ++i) {
        // From Eq. 10: <n_i_sigma> = 1 - G_sigma(i, i)
        double n_up = 1.0 - G_up(i, i);
        double n_down = 1.0 - G_down(i, i);
        
        // From Eq. 11: <n_up n_down> = (1 - G_up(i,i)) * (1 - G_down(i,i))
        double d_occ = (1.0 - G_up(i, i)) * (1.0 - G_down(i, i));
        
        // From Eq. 12: <(n_up - n_down)^2> = <n_up + n_down> - 2<n_up n_down>
        double moment = (n_up + n_down) - 2.0 * d_occ;

        density_up_(i) += n_up;
        density_down_(i) += n_down;
        double_occ_(i) += d_occ;
        local_moment_(i) += moment;
    }
    num_samples_++;
}

void Measurement::normalize() {
    if (num_samples_ > 0) {
        density_up_ /= num_samples_;
        density_down_ /= num_samples_;
        double_occ_ /= num_samples_;
        local_moment_ /= num_samples_;
    }
}

double Measurement::get_avg_density() const {
    return (density_up_.mean() + density_down_.mean());
}

double Measurement::get_avg_double_occ() const {
    return double_occ_.mean();
}

double Measurement::get_avg_local_moment() const {
    return local_moment_.mean();
}

void Measurement::print_summary() const {
    std::cout << "========== DQMC MEASUREMENT SUMMARY ==========\n";
    std::cout << " Total Measurement Sweeps : " << num_samples_ << "\n";
    std::cout << " Average Particle Density : " << get_avg_density() << "\n";
    std::cout << " Average Double Occupancy : " << get_avg_double_occ() << "\n";
    std::cout << " Average Local Moment <m_z^2> : " << get_avg_local_moment() << "\n";
    std::cout << "==============================================\n";
}