#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <cmath>

#include "SquareLattice.hpp"
#include "KineticBuilder.hpp"
#include "InteractionConfig.hpp"
#include "GreensEngine.hpp"
#include "DqmcSampler.hpp"
#include "Measurement.hpp"

struct Result {
    double U;
    double beta;
    double T;
    double density;
    double double_occ;
    double local_moment;
    double nn_spin_corr;
    double af_structure_factor;
};

Result run_one_simulation(
    int L_size,
    double U,
    double beta,
    double t,
    double target_delta_tau,
    int equilibration_sweeps,
    int measurement_sweeps,
    int reinversion_frequency,
    unsigned int seed
) {
    int num_sites = L_size * L_size;

    int num_slices =
        std::max(1, static_cast<int>(std::ceil(beta / target_delta_tau)));

    double delta_tau = beta / static_cast<double>(num_slices);

    // For shifted interaction U(n_up - 1/2)(n_down - 1/2),
    // half filling is mu = 0.
    double mu = 0.0;

    SquareLattice lattice(L_size, L_size, true);

    Eigen::MatrixXd k =
        KineticBuilder::buildKineticMatrix(
            lattice,
            t,
            mu,
            delta_tau
        );

    InteractionConfig config(num_sites, num_slices, U, delta_tau);
    config.randomize();

    bool use_stabilization = true;

    GreensEngine engine(
        num_sites,
        num_slices,
        k,
        use_stabilization,
        reinversion_frequency
    );

    engine.recompute_all_from_scratch(config);

    DqmcSampler sampler(num_sites, num_slices, k, seed);
    Measurement measurements(num_sites);

    for (int sweep = 0; sweep < equilibration_sweeps; ++sweep) {
        sampler.perform_sweep(config, engine);

        if ((sweep + 1) % reinversion_frequency == 0) {
            engine.recompute_all_from_scratch(config);
        }
    }

    engine.recompute_all_from_scratch(config);

    for (int sweep = 0; sweep < measurement_sweeps; ++sweep) {
        sampler.perform_sweep(config, engine);

        if ((sweep + 1) % reinversion_frequency == 0) {
            engine.recompute_all_from_scratch(config);
        }

        measurements.sample(
            engine.G_up(),
            engine.G_down(),
            lattice
        );
    }

    measurements.normalize();

    return Result{
        U,
        beta,
        1.0 / beta,
        measurements.get_avg_density(),
        measurements.get_avg_double_occ(),
        measurements.get_avg_local_moment(),
        measurements.get_nearest_neighbor_spin_corr(),
        measurements.get_af_structure_factor()
    };
}

int main() {
    int L_size = 6;
    double t = 1.0;
    double target_delta_tau = 0.05;

    int equilibration_sweeps = 1000;
    int measurement_sweeps = 5000;

    // Used both as Green's reinversion frequency and stabilization stride.
    int reinversion_frequency = 10;

    std::vector<double> U_values = {
        0.5, 1.0, 2.0, 4.0, 6.0, 8.0
    };

    std::vector<double> T_values = {
        100.0, 40.0, 20.0, 10.0, 5.0,
        2.0, 1.0, 0.5, 0.25
    };

    std::ofstream file("dqmc_scan.csv");

    if (!file.is_open()) {
        std::cerr << "Error: could not open dqmc_scan.csv\n";
        return 1;
    }

    file
        << "U,T,beta,L_tau,delta_tau,"
        << "density,double_occupancy,local_moment,"
        << "nn_spin_corr,af_structure_factor\n";

    std::cout << "Starting DQMC scan...\n";
    std::cout << "Lattice: " << L_size << "x" << L_size << "\n";
    std::cout << "t = " << t << "\n";
    std::cout << "target_delta_tau = " << target_delta_tau << "\n";
    std::cout << "equilibration_sweeps = " << equilibration_sweeps << "\n";
    std::cout << "measurement_sweeps = " << measurement_sweeps << "\n";
    std::cout << "reinversion_frequency = " << reinversion_frequency << "\n";

    unsigned int seed = 12345;

    for (double U : U_values) {
        std::cout << "\n=====================================\n";
        std::cout << "Scanning U = " << U << "\n";
        std::cout << "=====================================\n";

        for (double T : T_values) {
            double beta = 1.0 / T;

            int num_slices =
                std::max(
                    1,
                    static_cast<int>(std::ceil(beta / target_delta_tau))
                );

            double delta_tau =
                beta / static_cast<double>(num_slices);

            std::cout
                << "Running U=" << U
                << ", T=" << T
                << ", beta=" << beta
                << ", L_tau=" << num_slices
                << ", delta_tau=" << delta_tau
                << " ... "
                << std::flush;

            Result r = run_one_simulation(
                L_size,
                U,
                beta,
                t,
                target_delta_tau,
                equilibration_sweeps,
                measurement_sweeps,
                reinversion_frequency,
                seed++
            );

            file << std::setprecision(12)
                 << r.U << ","
                 << r.T << ","
                 << r.beta << ","
                 << num_slices << ","
                 << delta_tau << ","
                 << r.density << ","
                 << r.double_occ << ","
                 << r.local_moment << ","
                 << r.nn_spin_corr << ","
                 << r.af_structure_factor << "\n";

            file.flush();

            std::cout
                << "density=" << r.density
                << ", double_occ=" << r.double_occ
                << ", moment=" << r.local_moment
                << ", nn_spin=" << r.nn_spin_corr
                << ", S_pi_pi=" << r.af_structure_factor
                << "\n";
        }
    }

    file.close();

    std::cout << "\nDone. Data saved to dqmc_scan.csv\n";

    return 0;
}