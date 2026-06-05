#include "DqmcSampler.hpp"

#include <unsupported/Eigen/MatrixFunctions>
#include <stdexcept>
#include <cmath>

DqmcSampler::DqmcSampler(
    int num_sites,
    int num_slices,
    const Eigen::MatrixXd& k,
    unsigned int seed
)
    : num_sites_(num_sites),
      num_slices_(num_slices),
      rng_(seed),
      dist_(0.0, 1.0) {

    if (num_sites_ <= 0) {
        throw std::invalid_argument("DqmcSampler: num_sites must be positive.");
    }

    if (num_slices_ <= 0) {
        throw std::invalid_argument("DqmcSampler: num_slices must be positive.");
    }

    if (k.rows() != num_sites_ || k.cols() != num_sites_) {
        throw std::invalid_argument("DqmcSampler: k has wrong dimensions.");
    }

    exp_k_ = k.exp();
    exp_neg_k_ = (-k).exp();
}

void DqmcSampler::perform_sweep(
    InteractionConfig& config,
    GreensEngine& engine
) {
    Eigen::MatrixXd G_up = engine.G_up();
    Eigen::MatrixXd G_down = engine.G_down();

    if (G_up.rows() != num_sites_ || G_up.cols() != num_sites_) {
        throw std::runtime_error("DqmcSampler: G_up has wrong dimensions.");
    }

    if (G_down.rows() != num_sites_ || G_down.cols() != num_sites_) {
        throw std::runtime_error("DqmcSampler: G_down has wrong dimensions.");
    }

    for (int l = num_slices_ - 1; l >= 0; --l) {
        for (int i = 0; i < num_sites_; ++i) {

            // Your tests use InteractionConfig as (slice, site).
            int s = config.get_spin(l, i);

            Eigen::VectorXd exp_v_up =
                config.get_exponential_diagonal(l, 1.0);

            double exp_lambda_s = exp_v_up(i);

            double delta_up =
                1.0 / (exp_lambda_s * exp_lambda_s) - 1.0;

            double delta_down =
                exp_lambda_s * exp_lambda_s - 1.0;

            double r_up =
                1.0 + delta_up * (1.0 - G_up(i, i));

            double r_down =
                1.0 + delta_down * (1.0 - G_down(i, i));

            double ratio = r_up * r_down;

            // For no-sign-problem cases, ratio should be positive.
            // If you later sample sign-problem cases, you need to track the sign separately.
            double accept_prob = std::min(1.0, std::abs(ratio));

            // if (dist_(rng_) < accept_prob) {

            //     Eigen::VectorXd e_i =
            //         Eigen::VectorXd::Zero(num_sites_);

            //     e_i(i) = 1.0;

            //     // Correct Sherman-Morrison update:
            //     //
            //     // G' = G - [ delta * (G_col_i - e_i) / r ] * G_row_i
            //     //
            //     Eigen::VectorXd u_up =
            //         delta_up * (G_up.col(i) - e_i) / r_up;

            //     Eigen::VectorXd u_down =
            //         delta_down * (G_down.col(i) - e_i) / r_down;

            //     G_up.noalias() -= u_up * G_up.row(i);
            //     G_down.noalias() -= u_down * G_down.row(i);

            //     config.flip_spin(l, i);
            // }

            if (dist_(rng_) < accept_prob) {

                Eigen::VectorXd c_up =
                    delta_up * (
                        Eigen::VectorXd::Unit(num_sites_, i)
                        - G_up.row(i).transpose()
                    );

                Eigen::VectorXd c_down =
                    delta_down * (
                        Eigen::VectorXd::Unit(num_sites_, i)
                        - G_down.row(i).transpose()
                    );

                G_up -= (G_up.col(i) / r_up) * c_up.transpose();
                G_down -= (G_down.col(i) / r_down) * c_down.transpose();

                config.flip_spin(l, i);
            }
            
        }

        Eigen::VectorXd wrap_up =
            config.get_exponential_diagonal(l, 1.0);

        Eigen::VectorXd wrap_down =
            config.get_exponential_diagonal(l, -1.0);

        Eigen::VectorXd inv_wrap_up =
            wrap_up.cwiseInverse();

        Eigen::VectorXd inv_wrap_down =
            wrap_down.cwiseInverse();

        // B_l = exp(k) exp(v_l)
        // G <- B_l G B_l^{-1}
        G_up =
            exp_k_
            * wrap_up.asDiagonal()
            * G_up
            * inv_wrap_up.asDiagonal()
            * exp_neg_k_;

        G_down =
            exp_k_
            * wrap_down.asDiagonal()
            * G_down
            * inv_wrap_down.asDiagonal()
            * exp_neg_k_;
    }

    engine.set_G_up(G_up);
    engine.set_G_down(G_down);
}