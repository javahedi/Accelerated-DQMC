#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cmath>
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
    double nn_spin_corr;
    double structure_factor;
};

Fig3Result run_temperature_point(
    int L_size,
    double U,
    double T,
    int total_equilibration_sweeps,
    int total_measurement_sweeps,
    int stabilization_frequency,
    unsigned int base_seed
) {
    double beta = 1.0 / T;
    double t = 1.0; // Hopping scale reference
    double mu = 0.0; // Half-filling boundary

    // Dynamically adjust time slices so delta_tau doesn't get unphysically large
    double target_delta_tau = 0.10;
    int num_slices = std::max(2, static_cast<int>(std::ceil(beta / target_delta_tau)));
    double delta_tau = beta / static_cast<double>(num_slices);

    int num_sites = L_size * L_size;
    int num_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    int eq_sweeps_per_thread = total_equilibration_sweeps / num_threads;
    int meas_sweeps_per_thread = total_measurement_sweeps / num_threads;

    std::vector<double> thread_nn(num_threads, 0.0);
    std::vector<double> thread_sf(num_threads, 0.0);

    #pragma omp parallel for schedule(static)
    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
        SquareLattice lattice(L_size, L_size, true); // Periodic boundary conditions
        Eigen::MatrixXd k = KineticBuilder::buildKineticMatrix(lattice, t, mu, delta_tau);
        
        InteractionConfig config(num_sites, num_slices, U, delta_tau);
        config.randomize();

        // Optimized zero-allocation workspace engine
        GreensEngine engine(num_sites, num_slices, k, true, stabilization_frequency);
        engine.compute_stabilized_greens(config, num_slices - 1);

        DqmcSampler sampler(num_sites, num_slices, k, base_seed + thread_id * 1701);
        Measurement measurements(num_sites);

        // Warm up chain
        for (int sweep = 0; sweep < eq_sweeps_per_thread; ++sweep) {
            sampler.perform_sweep(config, engine);
        }

        // Measure chain
        for (int sweep = 0; sweep < meas_sweeps_per_thread; ++sweep) {
            sampler.perform_sweep(config, engine);
            measurements.sample(engine.G_up(), engine.G_down());
        }

        measurements.normalize();

        // Extract structural lattice correlations from your measurement engine
        thread_nn[thread_id] = measurements.get_nearest_neighbor_spin_corr(lattice);
        thread_sf[thread_id] = measurements.get_af_structure_factor(lattice);
    }

    double final_nn = 0.0;
    double final_sf = 0.0;
    for (int i = 0; i < num_threads; ++i) {
        final_nn += thread_nn[i];
        final_sf += thread_sf[i];
    }

    return Fig3Result{
        T,
        beta,
        final_nn / num_threads,
        final_sf / num_threads
    };
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Starting DQMC Production Sweep for Magnetic Observables (Fig 3)\n";
    std::cout << "Fixed Parameters: U = 2.0, Lattice = 6x6 (N=36)\n";
    std::cout << "========================================================\n\n";

    int L_size = 6;
    double U = 2.0;
    int equilibration_sweeps = 500;
    int measurement_sweeps = 2000;
    int stabilization_frequency = 10; // Checkpoint every 10 time slice wraps
    unsigned int base_seed = 42;

    // Generate log-spaced temperature values matching the reference figure's x-axis
    std::vector<double> temperatures = {
        100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 
        0.5, 0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.04, 0.03
    };

    std::ofstream csv_file("run_structureFactor_scan.csv");
    csv_file << "T,beta,NN_Spin_Correlation,AF_Structure_Factor\n";

    std::cout << std::setw(8) << "T" 
              << std::setw(10) << "Beta" 
              << std::setw(15) << "L Slices"
              << std::setw(18) << "<S_i S_{i+1}>" 
              << std::setw(18) << "S(pi,pi)" << "\n";
    std::cout << "------------------------------------------------------------------------\n";

    for (double T : temperatures) {
        double beta = 1.0 / T;
        int current_slices = std::max(2, static_cast<int>(std::ceil(beta / 0.10)));

        // Run the optimized multi-chain simulation point
        Fig3Result res = run_temperature_point(
            L_size, U, T, 
            equilibration_sweeps, measurement_sweeps, 
            stabilization_frequency, base_seed
        );

        std::cout << std::setw(8) << std::fixed << std::setprecision(3) << res.T
                  << std::setw(10) << std::setprecision(2) << res.beta
                  << std::setw(15) << current_slices
                  << std::setw(18) << std::setprecision(6) << res.nn_spin_corr
                  << std::setw(18) << std::setprecision(6) << res.structure_factor << std::endl;

        csv_file << res.T << "," << res.beta << "," << res.nn_spin_corr << "," << res.structure_factor << "\n";
    }

    csv_file.close();
    std::cout << "\n[Success] Data file written to 'run_structureFactor_scan.csv'.\n";
    return 0;
}