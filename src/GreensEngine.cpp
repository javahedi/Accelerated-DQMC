#include "GreensEngine.hpp"

#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/SVD>
#include <stdexcept>
#include <algorithm>

GreensEngine::GreensEngine(
    int num_sites,
    int num_slices,
    const Eigen::MatrixXd& k,
    bool stabilize,
    int stabilization_frequency
)
    : num_sites_(num_sites),
      num_slices_(num_slices),
      stabilize_(stabilize),
      stabilization_frequency_(stabilization_frequency) {

    if (num_sites_ <= 0) {
        throw std::invalid_argument("GreensEngine: num_sites must be positive.");
    }

    if (num_slices_ <= 0) {
        throw std::invalid_argument("GreensEngine: num_slices must be positive.");
    }

    if (k.rows() != num_sites_ || k.cols() != num_sites_) {
        throw std::invalid_argument("GreensEngine: k has wrong dimensions.");
    }

    if (stabilization_frequency_ <= 0) {
        stabilization_frequency_ = 5;
    }

    exp_k_ = k.exp();

    G_up_ = Eigen::MatrixXd::Identity(num_sites_, num_sites_);
    G_down_ = Eigen::MatrixXd::Identity(num_sites_, num_sites_);
}

Eigen::MatrixXd GreensEngine::make_B_slice(
    const InteractionConfig& config,
    int slice,
    double spin_factor
) const {
    Eigen::VectorXd exp_v_diag =
        config.get_exponential_diagonal(slice, spin_factor);

    if (exp_v_diag.size() != num_sites_) {
        throw std::runtime_error(
            "GreensEngine: interaction diagonal has wrong size."
        );
    }

    Eigen::MatrixXd B = exp_k_;

    for (int col = 0; col < num_sites_; ++col) {
        B.col(col) *= exp_v_diag(col);
    }

    return B;
}

Eigen::MatrixXd GreensEngine::compute_B_product(
    const InteractionConfig& config,
    double spin_factor
) const {
    Eigen::MatrixXd product =
        Eigen::MatrixXd::Identity(num_sites_, num_sites_);

    for (int l = 0; l < num_slices_; ++l) {
        Eigen::MatrixXd B_l = make_B_slice(config, l, spin_factor);
        product = B_l * product;
    }

    return product;
}

Eigen::MatrixXd GreensEngine::compute_stabilized_greens(
    const InteractionConfig& config,
    double spin_factor
) const {
    // Maintain:
    //
    //     P = B_{L-1} ... B_0 = U * D * Vt
    //
    // where U and Vt are well-conditioned orthogonal matrices,
    // and D stores all large/small scales.

    Eigen::MatrixXd U =
        Eigen::MatrixXd::Identity(num_sites_, num_sites_);

    Eigen::VectorXd D =
        Eigen::VectorXd::Ones(num_sites_);

    Eigen::MatrixXd Vt =
        Eigen::MatrixXd::Identity(num_sites_, num_sites_);

    Eigen::MatrixXd block =
        Eigen::MatrixXd::Identity(num_sites_, num_sites_);

    int block_count = 0;

    for (int l = 0; l < num_slices_; ++l) {
        Eigen::MatrixXd B_l = make_B_slice(config, l, spin_factor);

        // Accumulate current unstabilized block from the left.
        block = B_l * block;
        block_count++;

        if (
            block_count >= stabilization_frequency_ ||
            l == num_slices_ - 1
        ) {
            // New product:
            //
            //     P_new = block * U * D * Vt
            //
            // First form A = block * U * D.
            Eigen::MatrixXd A = block * U;

            for (int col = 0; col < num_sites_; ++col) {
                A.col(col) *= D(col);
            }

            Eigen::JacobiSVD<Eigen::MatrixXd> svd(
                A,
                Eigen::ComputeFullU | Eigen::ComputeFullV
            );

            U = svd.matrixU();
            D = svd.singularValues();

            // A = U_new * D_new * V_block^T
            // P_new = U_new * D_new * V_block^T * Vt_old
            Vt = svd.matrixV().transpose() * Vt;

            block.setIdentity();
            block_count = 0;
        }
    }

    // We need:
    //
    //     G = (I + U D Vt)^(-1)
    //
    // Use the stable identity:
    //
    //     I + U D Vt = U * (D + U^T Vt^T) * Vt
    //
    // Therefore:
    //
    //     G = Vt^T * (D + U^T Vt^T)^(-1) * U^T

    Eigen::MatrixXd Ut = U.transpose();
    Eigen::MatrixXd V = Vt.transpose();

    Eigen::MatrixXd middle = Ut * V;

    for (int i = 0; i < num_sites_; ++i) {
        middle(i, i) += D(i);
    }

    Eigen::MatrixXd solved =
        middle.fullPivLu().solve(Ut);

    Eigen::MatrixXd G =
        V * solved;

    return G;
}

void GreensEngine::recompute_all_from_scratch(
    const InteractionConfig& config
) {
    if (stabilize_) {
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                G_up_ = compute_stabilized_greens(config, 1.0);
            }

            #pragma omp section
            {
                G_down_ = compute_stabilized_greens(config, -1.0);
            }
        }
    } else {
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                Eigen::MatrixXd I =
                    Eigen::MatrixXd::Identity(num_sites_, num_sites_);

                Eigen::MatrixXd M =
                    I + compute_B_product(config, 1.0);

                G_up_ = M.partialPivLu().solve(I);
            }

            #pragma omp section
            {
                Eigen::MatrixXd I =
                    Eigen::MatrixXd::Identity(num_sites_, num_sites_);

                Eigen::MatrixXd M =
                    I + compute_B_product(config, -1.0);

                G_down_ = M.partialPivLu().solve(I);
            }
        }
    }
}

void GreensEngine::set_G_up(const Eigen::MatrixXd& G) {
    if (G.rows() != num_sites_ || G.cols() != num_sites_) {
        throw std::invalid_argument("GreensEngine: G_up has wrong dimensions.");
    }

    G_up_ = G;
}

void GreensEngine::set_G_down(const Eigen::MatrixXd& G) {
    if (G.rows() != num_sites_ || G.cols() != num_sites_) {
        throw std::invalid_argument("GreensEngine: G_down has wrong dimensions.");
    }

    G_down_ = G;
}