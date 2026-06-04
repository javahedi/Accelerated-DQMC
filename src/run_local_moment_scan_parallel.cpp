#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <omp.h> // Include OpenMP header to track threads if needed

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
};

Result run_one_simulation(
    int L_size,
    double U,
    double beta,
    double t,
    double target_delta_tau,
    int total_equilibration_sweeps,
    int total_measurement_sweeps,
    int reinversion_frequency,
    unsigned int base_seed
) {
    int num_sites = L_size * L_size;
    int num_slices = std::max(1, static_cast<int>(std::ceil(beta / target_delta_tau)));
    double delta_tau = beta / static_cast<double>(num_slices);
    double mu = 0.0; // Half-filling boundary

    // Determine how many parallel threads OpenMP has available
    int num_threads = 1;
    #pragma omp parallel
    {
        #pragma omp single
        num_threads = omp_get_num_threads();
    }

    // Distribute total sweep requirements evenly across all threads
    int eq_sweeps_per_thread = total_equilibration_sweeps / num_threads;
    int meas_sweeps_per_thread = total_measurement_sweeps / num_threads;

    // Shared arrays to safely gather thread-isolated results
    std::vector<double> thread_densities(num_threads, 0.0);
    std::vector<double> thread_double_occs(num_threads, 0.0);
    std::vector<double> thread_local_moments(num_threads, 0.0);

    // Launch independent Markov Chains in parallel
    #pragma omp parallel for schedule(static)
    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
        // Every thread gets its own isolated lattice, engines, configurations, and UNIQUE seed
        SquareLattice lattice(L_size, L_size, true);
        Eigen::MatrixXd k = KineticBuilder::buildKineticMatrix(lattice, t, mu, delta_tau);
        
        InteractionConfig config(num_sites, num_slices, U, delta_tau);
        config.randomize();

        // Use the preallocated internal workspaces from Step 2
        GreensEngine engine(num_sites, num_slices, k, true, reinversion_frequency);
        //GreensEngine engine(num_sites, num_slices, k, false); // For comparison, run without stabilization
        engine.recompute_all_from_scratch(config);

        // Offset the random seed so threads explore completely different paths in phase space
        DqmcSampler sampler(num_sites, num_slices, k, base_seed + thread_id * 1000);
        Measurement measurements(num_sites);

        // Thread-isolated equilibration phase
        for (int sweep = 0; sweep < eq_sweeps_per_thread; ++sweep) {
            sampler.perform_sweep(config, engine);
            if ((sweep + 1) % reinversion_frequency == 0) {
                engine.recompute_all_from_scratch(config);
            }
        }

        engine.recompute_all_from_scratch(config);

        // Thread-isolated measurement accumulation phase
        for (int sweep = 0; sweep < meas_sweeps_per_thread; ++sweep) {
            sampler.perform_sweep(config, engine);
            if ((sweep + 1) % reinversion_frequency == 0) {
                engine.recompute_all_from_scratch(config);
            }
            measurements.sample(engine.G_up(), engine.G_down());
        }

        measurements.normalize();

        // Stash the normalized statistical averages into the thread-safe array
        thread_densities[thread_id] = measurements.get_avg_density();
        thread_double_occs[thread_id] = measurements.get_avg_double_occ();
        thread_local_moments[thread_id] = measurements.get_avg_local_moment();
    }

    // Combine the independent measurements from all threads
    double final_density = 0.0;
    double final_double_occ = 0.0;
    double final_local_moment = 0.0;

    for (int i = 0; i < num_threads; ++i) {
        final_density += thread_densities[i];
        final_double_occ += thread_double_occs[i];
        final_local_moment += thread_local_moments[i];
    }

    return Result{
        U,
        beta,
        1.0 / beta,
        final_density / num_threads,
        final_double_occ / num_threads,
        final_local_moment / num_threads
    };
}


int main() {
    int L_size = 6;
    double t = 1.0;
    // Discretization adjusted to stay safe within the Suzuki-Trotter boundaries
    double target_delta_tau = 0.05; 

    int equilibration_sweeps = 1000;
    int measurement_sweeps = 5000;
    int reinversion_frequency = 1; // High frequency matches our QR pivot stabilization intervals

    std::vector<double> U_values = {
        0.5, 1.0, 2.0, 4.0, 6.0, 8.0
    };

    std::vector<double> T_values = { 
        100.0, 40.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.25
    };

    std::ofstream file("local_moment_scan.csv");

    if (!file.is_open()) {
        std::cerr << "Error: could not open local_moment_scan.csv\n";
        return 1;
    }

    file << "U,T,beta,density,double_occupancy,local_moment\n";

    std::cout << "Starting stabilized local moment scan...\n";
    std::cout << "Lattice: " << L_size << "x" << L_size << "\n";
    std::cout << "t = " << t << "\n";
    std::cout << "target_delta_tau = " << target_delta_tau << "\n";

    unsigned int seed = 12345;

    for (double U : U_values) {
        std::cout << "\n=====================================\n";
        std::cout << "Scanning U = " << U << "\n";
        std::cout << "=====================================\n";

        for (double T : T_values) {
            double beta = 1.0 / T;

            int num_slices =
                std::max(1, static_cast<int>(std::ceil(beta / target_delta_tau)));

            double delta_tau = beta / static_cast<double>(num_slices);

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
                 << r.density << ","
                 << r.double_occ << ","
                 << r.local_moment << "\n";

            file.flush();

            std::cout
                << "moment=" << r.local_moment
                << ", density=" << r.density
                << ", double_occ=" << r.double_occ
                << "\n";
        }
    }

    file.close();

    std::cout << "\nDone. Data saved to local_moment_scan.csv\n";

    return 0;
}