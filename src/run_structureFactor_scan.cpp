#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <omp.h>

#include "SquareLattice.hpp"
#include "KineticBuilder.hpp"
#include "InteractionConfig.hpp"
#include "GreensEngine.hpp"
#include "DqmcSampler.hpp"
#include "Measurement.hpp"

struct Fig3Result {
    double T;
    double beta;
    int num_slices;
    double delta_tau;
    double nn_spin_corr;
    double structure_factor;
};

Fig3Result run_temperature_point(
    int L_size,
    double U,
    double T,
    int total_equilibration_sweeps,
    int total_measurement_sweeps,
    unsigned int base_seed
) {
    double beta = 1.0 / T;
    double t = 1.0;
    double mu = 0.0;

    double target_delta_tau = 0.05;

    int num_slices =
        std::max(
            2,
            static_cast<int>(std::ceil(beta / target_delta_tau))
        );

    double delta_tau =
        beta / static_cast<double>(num_slices);

    int num_sites = L_size * L_size;

    int num_threads = 1;

    #pragma omp parallel
    {
        #pragma omp single
        {
            num_threads = omp_get_num_threads();
        }
    }

    int eq_sweeps_per_thread =
        std::max(1, total_equilibration_sweeps / num_threads);

    int meas_sweeps_per_thread =
        std::max(1, total_measurement_sweeps / num_threads);

    std::vector<double> thread_nn(num_threads, 0.0);
    std::vector<double> thread_sf(num_threads, 0.0);

    #pragma omp parallel for schedule(static)
    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
        SquareLattice lattice(L_size, L_size, true);

        Eigen::MatrixXd k =
            KineticBuilder::buildKineticMatrix(
                lattice,
                t,
                mu,
                delta_tau
            );

        InteractionConfig config(
            num_sites,
            num_slices,
            U,
            delta_tau
        );

        config.randomize();

        //GreensEngine engine(num_sites,num_slices,k,true,1);
        GreensEngine engine(num_sites,num_slices,k,false); // No stabilization for this scan, to save time. We will recompute from scratch every time, so it should be fine.

        engine.recompute_all_from_scratch(config);

        DqmcSampler sampler(
            num_sites,
            num_slices,
            k,
            base_seed + static_cast<unsigned int>(thread_id * 1701)
        );

        Measurement measurements(num_sites);

        for (int sweep = 0; sweep < eq_sweeps_per_thread; ++sweep) {
            sampler.perform_sweep(config, engine);
            engine.recompute_all_from_scratch(config);
        }

        for (int sweep = 0; sweep < meas_sweeps_per_thread; ++sweep) {
            sampler.perform_sweep(config, engine);

            // Critical: measure only a freshly reconstructed Green's function.
            engine.recompute_all_from_scratch(config);

            measurements.sample(
                engine.G_up(),
                engine.G_down(),
                lattice
            );
        }

        measurements.normalize();

        thread_nn[thread_id] =
            measurements.get_nearest_neighbor_spin_corr();

        thread_sf[thread_id] =
            measurements.get_af_structure_factor();
    }

    double final_nn = 0.0;
    double final_sf = 0.0;

    for (int i = 0; i < num_threads; ++i) {
        final_nn += thread_nn[i];
        final_sf += thread_sf[i];
    }

    final_nn /= static_cast<double>(num_threads);
    final_sf /= static_cast<double>(num_threads);

    return Fig3Result{
        T,
        beta,
        num_slices,
        delta_tau,
        final_nn,
        final_sf
    };
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Starting DQMC magnetic observable scan\n";
    std::cout << "Fixed parameters: U = 2.0, lattice = 6x6\n";
    std::cout << "========================================================\n\n";

    int L_size = 6;
    double U = 2.0;

    int equilibration_sweeps = 500;
    int measurement_sweeps = 2000;

    unsigned int base_seed = 42;

    std::vector<double> temperatures = {
        100.0, 50.0, 20.0, 10.0, 5.0,
        2.0, 1.0, 0.5, 0.3, 0.2,
        0.15, 0.1, 0.08, 0.06, 0.05,
        0.04, 0.03
    };

    std::ofstream csv_file("run_structureFactor_scan.csv");

    if (!csv_file.is_open()) {
        std::cerr << "Error: could not open run_structureFactor_scan.csv\n";
        return 1;
    }

    csv_file
        << "T,beta,L_tau,delta_tau,"
        << "NN_Spin_Correlation,AF_Structure_Factor\n";

    std::cout
        << std::setw(8)  << "T"
        << std::setw(10) << "Beta"
        << std::setw(12) << "L_tau"
        << std::setw(14) << "DeltaTau"
        << std::setw(20) << "<S_i S_j>"
        << std::setw(20) << "S(pi,pi)"
        << "\n";

    std::cout
        << "--------------------------------------------------------------------------\n";

    for (double T : temperatures) {
        Fig3Result res =
            run_temperature_point(
                L_size,
                U,
                T,
                equilibration_sweeps,
                measurement_sweeps,
                base_seed
            );

        base_seed += 7919;

        std::cout
            << std::setw(8)  << std::fixed << std::setprecision(3) << res.T
            << std::setw(10) << std::setprecision(2) << res.beta
            << std::setw(12) << res.num_slices
            << std::setw(14) << std::setprecision(5) << res.delta_tau
            << std::setw(20) << std::setprecision(8) << res.nn_spin_corr
            << std::setw(20) << std::setprecision(8) << res.structure_factor
            << "\n";

        csv_file
            << std::setprecision(12)
            << res.T << ","
            << res.beta << ","
            << res.num_slices << ","
            << res.delta_tau << ","
            << res.nn_spin_corr << ","
            << res.structure_factor << "\n";

        csv_file.flush();
    }

    csv_file.close();

    std::cout
        << "\n[Success] Data file written to "
        << "'run_structureFactor_scan.csv'.\n";

    return 0;
}