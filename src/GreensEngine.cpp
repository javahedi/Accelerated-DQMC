
#include "GreensEngine.hpp"
#include <unsupported/Eigen/MatrixFunctions>
#include <cmath>

GreensEngine::GreensEngine(int num_sites, int num_slices, const Eigen::MatrixXd& k, bool stabilize, int stabilization_frequency)
    : num_sites_(num_sites), num_slices_(num_slices), stabilize_(stabilize), stabilization_frequency_(stabilization_frequency) {
    exp_k_ = k.exp();
    G_up_ = Eigen::MatrixXd::Identity(num_sites_, num_sites_);
    G_down_ = Eigen::MatrixXd::Identity(num_sites_, num_sites_);

    // Allocate memory on the heap EXACTLY ONCE here
    U_workspace_.resize(num_sites_, num_sites_);
    D_workspace_.resize(num_sites_);
    V_workspace_.resize(num_sites_, num_sites_);
    M_workspace_.resize(num_sites_, num_sites_);
    target_workspace_.resize(num_sites_, num_sites_);
    V_inv_workspace_.resize(num_sites_, num_sites_);
    inner_workspace_.resize(num_sites_, num_sites_);
}

Eigen::MatrixXd GreensEngine::compute_B_product(const InteractionConfig& config, double spin_factor) const {
    Eigen::MatrixXd product = Eigen::MatrixXd::Identity(num_sites_, num_sites_);
    for (int l = 0; l < num_slices_; ++l) {
        Eigen::VectorXd exp_v_diag = config.get_exponential_diagonal(l, spin_factor);
        Eigen::MatrixXd B_l = exp_k_ * exp_v_diag.asDiagonal();
        product = product * B_l;
    }
    return product;
}

Eigen::MatrixXd GreensEngine::compute_stabilized_greens(const InteractionConfig& config, double spin_factor) const {
    // Re-initialize existing buffers without discarding their memory footprints
    U_workspace_.setIdentity();
    D_workspace_.setOnes();
    V_workspace_.setIdentity();
    M_workspace_.setIdentity();

    for (int l = 0; l < num_slices_; ++l) {
        Eigen::VectorXd exp_v_diag = config.get_exponential_diagonal(l, spin_factor);
        
        // Use noalias() to prevent Eigen from making invisible internal allocations
        M_workspace_ *= exp_k_ * exp_v_diag.asDiagonal();

        if ((l + 1) % stabilization_frequency_ == 0 || (l == num_slices_ - 1)) {
            target_workspace_.noalias() = M_workspace_ * U_workspace_ * D_workspace_.asDiagonal();
            
            Eigen::HouseholderQR<Eigen::MatrixXd> qr(target_workspace_);
            U_workspace_ = qr.householderQ();
            Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();

            for (int i = 0; i < num_sites_; ++i) {
                D_workspace_(i) = R(i, i);
                if (std::abs(D_workspace_(i)) < 1e-14) D_workspace_(i) = 1e-14;
                R.row(i) /= D_workspace_(i);
            }
            V_workspace_ = R * V_workspace_;
            M_workspace_.setIdentity();
        }
    }

    if (num_slices_ % stabilization_frequency_ != 0) {
        target_workspace_.noalias() = M_workspace_ * U_workspace_ * D_workspace_.asDiagonal();
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(target_workspace_);
        U_workspace_ = qr.householderQ();
        Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();
        for (int i = 0; i < num_sites_; ++i) {
            D_workspace_(i) = R(i, i);
            R.row(i) /= D_workspace_(i);
        }
        V_workspace_ = R * V_workspace_;
    }

    Eigen::MatrixXd U_inv = U_workspace_.transpose(); // orthogonal matrix transpose is free
    V_inv_workspace_ = V_workspace_.inverse();
    inner_workspace_.noalias() = U_inv * V_inv_workspace_;
    
    for (int i = 0; i < num_sites_; ++i) {
        inner_workspace_.diagonal()(i) += D_workspace_(i);
    }

    return V_inv_workspace_ * inner_workspace_.inverse() * U_inv;
}

void GreensEngine::recompute_all_from_scratch(const InteractionConfig& config) {
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
                Eigen::MatrixXd I = Eigen::MatrixXd::Identity(num_sites_, num_sites_);
                G_up_ = (I + compute_B_product(config, 1.0)).inverse();
            }
            #pragma omp section
            {
                Eigen::MatrixXd I = Eigen::MatrixXd::Identity(num_sites_, num_sites_);
                G_down_ = (I + compute_B_product(config, -1.0)).inverse();
            }
        }
    }
}

// #include "GreensEngine.hpp"
// #include <unsupported/Eigen/MatrixFunctions>
// #include <cmath>
// #include <algorithm>


// // GreensEngine::GreensEngine(int num_sites, int num_slices, const Eigen::MatrixXd& k, bool stabilize, int stabilization_frequency)
// //     : num_sites_(num_sites), num_slices_(num_slices), stabilize_(stabilize), stabilization_frequency_(stabilization_frequency) {
// //     exp_k_ = k.exp();
// //     G_up_ = Eigen::MatrixXd::Identity(num_sites_, num_sites_);
// //     G_down_ = Eigen::MatrixXd::Identity(num_sites_, num_sites_);
// // }

// GreensEngine::GreensEngine(
//     int num_sites,
//     int num_slices,
//     const Eigen::MatrixXd& k,
//     bool stabilize,
//     int stabilization_frequency
// )
//     : num_sites_(num_sites),
//       num_slices_(num_slices),
//       stabilize_(stabilize),
//       stabilization_frequency_(std::max(1, stabilization_frequency)) {
//     exp_k_ = k.exp();

//     G_up_ = Eigen::MatrixXd::Identity(num_sites_, num_sites_);
//     G_down_ = Eigen::MatrixXd::Identity(num_sites_, num_sites_);


//     // Allocate memory on the heap EXACTLY ONCE here
//     U_workspace_.resize(num_sites_, num_sites_);
//     D_workspace_.resize(num_sites_);
//     V_workspace_.resize(num_sites_, num_sites_);
//     M_workspace_.resize(num_sites_, num_sites_);
//     target_workspace_.resize(num_sites_, num_sites_);
//     V_inv_workspace_.resize(num_sites_, num_sites_);
//     inner_workspace_.resize(num_sites_, num_sites_);
// }

// // Your original working method (100% untouched)
// Eigen::MatrixXd GreensEngine::compute_B_product(const InteractionConfig& config, double spin_factor) const {
//     Eigen::MatrixXd product = Eigen::MatrixXd::Identity(num_sites_, num_sites_);
//     for (int l = 0; l < num_slices_; ++l) {
//         Eigen::VectorXd exp_v_diag = config.get_exponential_diagonal(l, spin_factor);
//         Eigen::MatrixXd B_l = exp_k_ * exp_v_diag.asDiagonal();
//         product = product * B_l;
//     }
//     return product;
// }

// // The new stabilized numerical backend
// // Eigen::MatrixXd GreensEngine::compute_stabilized_greens(const InteractionConfig& config, double spin_factor) const {
// //     Eigen::MatrixXd U = Eigen::MatrixXd::Identity(num_sites_, num_sites_);
// //     Eigen::VectorXd D = Eigen::VectorXd::Ones(num_sites_);
// //     Eigen::MatrixXd V = Eigen::MatrixXd::Identity(num_sites_, num_sites_);
// //     Eigen::MatrixXd M = Eigen::MatrixXd::Identity(num_sites_, num_sites_);

// //     for (int l = 0; l < num_slices_; ++l) {
// //         Eigen::VectorXd exp_v_diag = config.get_exponential_diagonal(l, spin_factor);
// //         Eigen::MatrixXd B_l = exp_k_ * exp_v_diag.asDiagonal();
// //         M = M * B_l;

// //         if ((l + 1) % stabilization_frequency_ == 0 || (l == num_slices_ - 1)) {
// //             Eigen::MatrixXd target = M * U * D.asDiagonal();
// //             Eigen::HouseholderQR<Eigen::MatrixXd> qr(target);
// //             U = qr.householderQ();
// //             Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();

// //             for (int i = 0; i < num_sites_; ++i) {
// //                 D(i) = R(i, i);
// //                 if (std::abs(D(i)) < 1e-14) D(i) = 1e-14;
// //                 R.row(i) /= D(i);
// //             }
// //             V = R * V;
// //             M.setIdentity();
// //         }
// //     }

// //     if (num_slices_ % stabilization_frequency_ != 0) {
// //         Eigen::MatrixXd target = M * U * D.asDiagonal();
// //         Eigen::HouseholderQR<Eigen::MatrixXd> qr(target);
// //         U = qr.householderQ();
// //         Eigen::MatrixXd R = qr.matrixQR().triangularView<Eigen::Upper>();
// //         for (int i = 0; i < num_sites_; ++i) {
// //             D(i) = R(i, i);
// //             R.row(i) /= D(i);
// //         }
// //         V = R * V;
// //     }

// //     Eigen::MatrixXd U_inv = U.transpose();
// //     Eigen::MatrixXd V_inv = V.inverse();
// //     Eigen::MatrixXd inner = U_inv * V_inv;
// //     for (int i = 0; i < num_sites_; ++i) {
// //         inner.diagonal()(i) += D(i);
// //     }

// //     return V_inv * inner.inverse() * U_inv;
// // }



// Eigen::MatrixXd GreensEngine::compute_stabilized_greens(
//     const InteractionConfig& config, double spin_factor) const {

//     Eigen::MatrixXd U = Eigen::MatrixXd::Identity(num_sites_, num_sites_);
//     Eigen::VectorXd D = Eigen::VectorXd::Ones(num_sites_);
//     Eigen::MatrixXd V = Eigen::MatrixXd::Identity(num_sites_, num_sites_);
//     Eigen::MatrixXd block =Eigen::MatrixXd::Identity(num_sites_, num_sites_);

//     for (int l = 0; l < num_slices_; ++l) {
//         Eigen::VectorXd exp_v_diag = config.get_exponential_diagonal(l, spin_factor);

//         Eigen::MatrixXd B_l = exp_k_ * exp_v_diag.asDiagonal();

//         block = block * B_l;

//         if ((l + 1) % stabilization_frequency_ == 0 ||
//             l == num_slices_ - 1) {

//             Eigen::MatrixXd target =
//                 block * U * D.asDiagonal();

//             Eigen::HouseholderQR<Eigen::MatrixXd> qr(target);

//             Eigen::MatrixXd Q = qr.householderQ();
//             Eigen::MatrixXd R =
//                 qr.matrixQR().template triangularView<Eigen::Upper>();

//             Eigen::VectorXd new_D(num_sites_);

//             for (int i = 0; i < num_sites_; ++i) {
//                 new_D(i) = R(i, i);

//                 if (std::abs(new_D(i)) > 0.0) {
//                     R.row(i) /= new_D(i);
//                 }
//             }

//             U = Q;
//             D = new_D;
//             V = R * V;

//             block.setIdentity();
//         }
//     }

//     Eigen::MatrixXd U_inv = U.transpose();
//     Eigen::MatrixXd V_inv = V.inverse();

//     Eigen::MatrixXd inner = U_inv * V_inv;

//     for (int i = 0; i < num_sites_; ++i) {
//         inner(i, i) += D(i);
//     }

//     return V_inv * inner.inverse() * U_inv;
// }


// // Traffic router based on the flag
// void GreensEngine::recompute_all_from_scratch(const InteractionConfig& config) {
//     if (stabilize_) {
//         // Run the stabilized UDV mode in parallel across two CPU cores
//         #pragma omp parallel sections
//         {
//             #pragma omp section
//             {
//                 G_up_ = compute_stabilized_greens(config, 1.0);
//             }
//             #pragma omp section
//             {
//                 G_down_ = compute_stabilized_greens(config, -1.0);
//             }
//         }
//     } else {
//         // Run your original code path in parallel across two CPU cores
//         #pragma omp parallel sections
//         {
//             #pragma omp section
//             {
//                 Eigen::MatrixXd I = Eigen::MatrixXd::Identity(num_sites_, num_sites_);
//                 G_up_ = (I + compute_B_product(config, 1.0)).inverse();
//             }
//             #pragma omp section
//             {
//                 Eigen::MatrixXd I = Eigen::MatrixXd::Identity(num_sites_, num_sites_);
//                 G_down_ = (I + compute_B_product(config, -1.0)).inverse();
//             }
//         }
//     }
// }