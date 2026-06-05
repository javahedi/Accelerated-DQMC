#include "Measurement.hpp"
#include "SquareLattice.hpp" // Ensure this path points to your lattice definition
#include <iostream>
#include <cmath>

Measurement::Measurement(int num_sites)
    : num_sites_(num_sites), num_samples_(0),
      density_up_(Eigen::VectorXd::Zero(num_sites)),
      density_down_(Eigen::VectorXd::Zero(num_sites)),
      double_occ_(Eigen::VectorXd::Zero(num_sites)),
      local_moment_(Eigen::VectorXd::Zero(num_sites)),
      G_up_accum_(Eigen::MatrixXd::Zero(num_sites, num_sites)),
      G_down_accum_(Eigen::MatrixXd::Zero(num_sites, num_sites)) {}

void Measurement::sample(const Eigen::MatrixXd& G_up, const Eigen::MatrixXd& G_down) {
    // 1. Accumulate full non-local matrices for off-diagonal Wick evaluations later
    G_up_accum_ += G_up;
    G_down_accum_ += G_down;

    // 2. Local accumulations
    for (int i = 0; i < num_sites_; ++i) {
        double n_up = 1.0 - G_up(i, i);
        double n_down = 1.0 - G_down(i, i);
        double d_occ = (1.0 - G_up(i, i)) * (1.0 - G_down(i, i));
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
        G_up_accum_ /= num_samples_;
        G_down_accum_ /= num_samples_;
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

// ============================================================================
// 3. Compute Nearest Neighbor Spin Correlation: <S_i * S_{i+1}> = 3 * <S_i^z S_{j}^z>
// ============================================================================
double Measurement::get_nearest_neighbor_spin_corr(const SquareLattice& lattice) const {
    double total_corr = 0.0;
    int bond_count = 0;

    for (int i = 0; i < num_sites_; ++i) {
        // Query your lattice class for neighbors of site i
        std::vector<int> neighbors = lattice.get_neighbors(i); 
        
        for (int j : neighbors) {
            // Wick's contraction for Sz-Sz off-diagonal terms
            double sz_sz = 0.25 * ((1.0 - 2.0 * G_up_accum_(i, i)) * (1.0 - 2.0 * G_down_accum_(j, j)));
            
            total_corr += sz_sz;
            bond_count++;
        }
    }

    // Multiply by 3 for full isotropic spatial space: <S . S> = 3 * <Sz Sz>
    return (bond_count == 0) ? 0.0 : 3.0 * (total_corr / bond_count);
}

// ============================================================================
// 4. Compute Antiferromagnetic Structure Factor: S(pi,pi)
// ============================================================================
double Measurement::get_af_structure_factor(const SquareLattice& lattice) const {
    double sum = 0.0;

    for (int i = 0; i < num_sites_; ++i) {
        // Fetch spatial coordinates (x, y) to compute staggered phase
        int xi = lattice.get_x(i);
        int yi = lattice.get_y(i);

        for (int j = 0; j < num_sites_; ++j) {
            int xj = lattice.get_x(j);
            int yj = lattice.get_y(j);

            // Staggered factor (-1)^(|i-j|) = exp(i * Q * (r_i - r_j)) for Q=(pi,pi)
            int phase = ((xi - xj) + (yi - yj)) % 2 == 0 ? 1 : -1;

            double sz_sz = 0.0;
            if (i == j) {
                // Diagonal local moment calculation
                double n_up = 1.0 - G_up_accum_(i, i);
                double n_down = 1.0 - G_down_accum_(i, i);
                double d_occ = double_occ_(i); // Normalized local double occupancy buffer
                sz_sz = 0.25 * (n_up + n_down - 2.0 * d_occ);
            } else {
                // Non-local calculation
                sz_sz = 0.25 * ((1.0 - 2.0 * G_up_accum_(i, i)) * (1.0 - 2.0 * G_down_accum_(j, j)));
            }

            sum += phase * sz_sz;
        }
    }

    return sum / num_sites_;
}

void Measurement::print_summary() const {
    std::cout << "========== DQMC MEASUREMENT SUMMARY ==========\n";
    std::cout << " Total Measurement Sweeps : " << num_samples_ << "\n";
    std::cout << " Average Particle Density : " << get_avg_density() << "\n";
    std::cout << " Average Double Occupancy : " << get_avg_double_occ() << "\n";
    std::cout << " Average Local Moment <m_z^2> : " << get_avg_local_moment() << "\n";
    std::cout << "==============================================\n";
}

// #include "Measurement.hpp"
// #include <iostream>

// Measurement::Measurement(int num_sites)
//     : num_sites_(num_sites), num_samples_(0),
//       density_up_(Eigen::VectorXd::Zero(num_sites)),
//       density_down_(Eigen::VectorXd::Zero(num_sites)),
//       double_occ_(Eigen::VectorXd::Zero(num_sites)),
//       local_moment_(Eigen::VectorXd::Zero(num_sites)) {}

// void Measurement::sample(const Eigen::MatrixXd& G_up, const Eigen::MatrixXd& G_down) {
//     for (int i = 0; i < num_sites_; ++i) {
//         // From Eq. 10: <n_i_sigma> = 1 - G_sigma(i, i)
//         double n_up = 1.0 - G_up(i, i);
//         double n_down = 1.0 - G_down(i, i);
        
//         // From Eq. 11: <n_up n_down> = (1 - G_up(i,i)) * (1 - G_down(i,i))
//         double d_occ = (1.0 - G_up(i, i)) * (1.0 - G_down(i, i));
        
//         // From Eq. 12: <(n_up - n_down)^2> = <n_up + n_down> - 2<n_up n_down>
//         double moment = (n_up + n_down) - 2.0 * d_occ;

//         density_up_(i) += n_up;
//         density_down_(i) += n_down;
//         double_occ_(i) += d_occ;
//         local_moment_(i) += moment;
//     }
//     num_samples_++;
// }

// void Measurement::normalize() {
//     if (num_samples_ > 0) {
//         density_up_ /= num_samples_;
//         density_down_ /= num_samples_;
//         double_occ_ /= num_samples_;
//         local_moment_ /= num_samples_;
//     }
// }

// double Measurement::get_avg_density() const {
//     return (density_up_.mean() + density_down_.mean());
// }

// double Measurement::get_avg_double_occ() const {
//     return double_occ_.mean();
// }

// double Measurement::get_avg_local_moment() const {
//     return local_moment_.mean();
// }

// void Measurement::print_summary() const {
//     std::cout << "========== DQMC MEASUREMENT SUMMARY ==========\n";
//     std::cout << " Total Measurement Sweeps : " << num_samples_ << "\n";
//     std::cout << " Average Particle Density : " << get_avg_density() << "\n";
//     std::cout << " Average Double Occupancy : " << get_avg_double_occ() << "\n";
//     std::cout << " Average Local Moment <m_z^2> : " << get_avg_local_moment() << "\n";
//     std::cout << "==============================================\n";
// }