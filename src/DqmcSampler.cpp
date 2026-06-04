#include "DqmcSampler.hpp"
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>


DqmcSampler::DqmcSampler(int num_sites, int num_slices, const Eigen::MatrixXd& k, unsigned int seed)
    : num_sites_(num_sites), num_slices_(num_slices), rng_(seed), dist_(0.0, 1.0) {
    exp_k_ = k.exp();
    exp_neg_k_ = (-k).exp();
}

void DqmcSampler::perform_sweep(InteractionConfig& config, GreensEngine& engine) {
    Eigen::MatrixXd G_up = engine.G_up();
    Eigen::MatrixXd G_down = engine.G_down();

    // Loop backwards through imaginary time slices: L-1 down to 0
    for (int l = num_slices_ - 1; l >= 0; --l) {
        
        // Spatial local updates on slice l
        for (int i = 0; i < num_sites_; ++i) {
            // Fetch current spin entry
            int s = config.get_spin(i, l);
            
            // Fetch exponential diagonals for current slice configuration
            Eigen::VectorXd exp_v_up = config.get_exponential_diagonal(l, 1.0);
            
            // del_up = exp(-2*lambda*s) - 1, del_down = exp(+2*lambda*s) - 1
            double del_u = (1.0 / (exp_v_up(i) * exp_v_up(i))) - 1.0;
            double del_d = (exp_v_up(i) * exp_v_up(i)) - 1.0;

            // Metropolis selection criteria
            double d_up = 1.0 + (1.0 - G_up(i, i)) * del_u;
            double d_down = 1.0 + (1.0 - G_down(i, i)) * del_d;
            double d = d_up * d_down;

            // if (dist_(rng_) < std::abs(d)) {
            //     // Corrected: pulling from row(i) to accurately match [G]_{ij}
            //     Eigen::VectorXd c_up = (-del_u * G_up.row(i)).transpose();
            //     c_up(i) += del_u;

            //     Eigen::VectorXd c_down = (-del_d * G_down.row(i)).transpose();
            //     c_down(i) += del_d;

            //     // Apply Sherman-Morrison rank-1 formulation update
            //     G_up = G_up - (G_up.col(i) / (1.0 + c_up(i))) * c_up.transpose();
            //     G_down = G_down - (G_down.col(i) / (1.0 + c_down(i))) * c_down.transpose();

            //     // Flip configuration space array spin scalar
            //     config.flip_spin(i, l);
            // }

            if (dist_(rng_) < std::abs(d)) {
                Eigen::VectorXd e_i = Eigen::VectorXd::Zero(num_sites_);
                e_i(i) = 1.0;

                double r_up = 1.0 + del_u * (1.0 - G_up(i, i));
                double r_down = 1.0 + del_d * (1.0 - G_down(i, i));

                Eigen::VectorXd a_up = del_u * (e_i - G_up.col(i));
                Eigen::VectorXd a_down = del_d * (e_i - G_down.col(i));

                G_up = G_up - (a_up / r_up) * G_up.row(i);
                G_down = G_down - (a_down / r_down) * G_down.row(i);

                config.flip_spin(i, l);
            }
        }

        // Time Slice Wrapping Step: G = B_l * G * B_l^-1
        Eigen::VectorXd wrap_up = config.get_exponential_diagonal(l, 1.0);
        Eigen::VectorXd wrap_down = config.get_exponential_diagonal(l, -1.0);
        Eigen::VectorXd inv_wrap_up = wrap_up.cwiseInverse();
        Eigen::VectorXd inv_wrap_down = wrap_down.cwiseInverse();

        G_up = exp_k_ * (wrap_up.asDiagonal() * G_up * inv_wrap_up.asDiagonal()) * exp_neg_k_;
        G_down = exp_k_ * (wrap_down.asDiagonal() * G_down * inv_wrap_down.asDiagonal()) * exp_neg_k_;
    }

    // Save final updated state back into the engine
    engine.set_G_up(G_up);
    engine.set_G_down(G_down);
}