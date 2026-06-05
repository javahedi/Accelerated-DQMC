#include "GreensEngine.hpp"

#include <unsupported/Eigen/MatrixFunctions>
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
        stabilization_frequency_ = num_slices_;
    }

    // IMPORTANT:
    // If the DQMC formula uses exp(-Δτ K), then the caller must pass k = -Δτ K.
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

    // B_l = exp(k) * exp(v_l).
    // Since exp(v_l) is diagonal, multiply columns of exp(k).
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

    // Convention:
    // product = B_{L-1} ... B_1 B_0.
    //
    // This matches the common DQMC equal-time convention where new slices
    // multiply from the left.
    for (int l = 0; l < num_slices_; ++l) {
        Eigen::MatrixXd B_l = make_B_slice(config, l, spin_factor);
        // product.noalias() = B_l * product;
        // Do NOT use noalias here.
        // product is both input and output.
        product = B_l * product;
        
    }

    return product;
}

Eigen::MatrixXd GreensEngine::compute_stabilized_greens(
    const InteractionConfig& config,
    double spin_factor
) const {
    // Thread-safe local workspaces.
    Eigen::MatrixXd Q =
        Eigen::MatrixXd::Identity(num_sites_, num_sites_);

    Eigen::MatrixXd R_total =
        Eigen::MatrixXd::Identity(num_sites_, num_sites_);

    int since_last_qr = 0;

    for (int l = 0; l < num_slices_; ++l) {
        Eigen::MatrixXd B_l = make_B_slice(config, l, spin_factor);

        // Accumulate from the left:
        // P_new = B_l * P_old.
        Eigen::MatrixXd A = B_l * Q;

        ++since_last_qr;

        if (since_last_qr >= stabilization_frequency_ ||
            l == num_slices_ - 1) {

            Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);

            Eigen::MatrixXd Q_new =
                qr.householderQ() *
                Eigen::MatrixXd::Identity(num_sites_, num_sites_);

            Eigen::MatrixXd R_new =
                qr.matrixQR()
                    .topRows(num_sites_)
                    .template triangularView<Eigen::Upper>();

            Q = Q_new;
            R_total = R_new * R_total;

            since_last_qr = 0;
        } else {
            Q = A;
        }
    }

    // We have approximately:
    //
    //     P = Q * R_total
    //
    // Need:
    //
    //     G = (I + P)^(-1)
    //       = (I + Q R)^(-1)
    //       = (Q (Q^T + R))^(-1)
    //       = (Q^T + R)^(-1) Q^T
    //
    // This avoids explicitly forming I + Q R.
    Eigen::MatrixXd Qt = Q.transpose();
    Eigen::MatrixXd system = Qt + R_total;

    Eigen::MatrixXd G =
        system.partialPivLu().solve(Qt);

    return G;
}

void GreensEngine::recompute_all_from_scratch(
    const InteractionConfig& config
) {
    if (stabilize_) {
        // Safe: compute_stabilized_greens uses only local workspaces.
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