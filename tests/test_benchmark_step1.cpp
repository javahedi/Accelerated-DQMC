#include <iostream>
#include <chrono>
#include "SquareLattice.hpp"
#include "KineticBuilder.hpp"
#include "InteractionConfig.hpp"
#include "GreensEngine.hpp"
#include "DqmcSampler.hpp"

int main() {
    std::cout << "--- DQMC Optimization Benchmark: Step 1 ---\n";

    // Use moderately heavy parameters to get a distinct time measurement
    int L_size = 6; 
    int num_sites = L_size * L_size;
    int num_slices = 80; // High number of slices to make the matrix ops heavy
    double beta = 4.0;
    double delta_tau = beta / num_slices;
    double t = 1.0;
    double U = 4.0;
    double mu = 0.0;

    SquareLattice lattice(L_size, L_size, true);
    Eigen::MatrixXd k = KineticBuilder::buildKineticMatrix(lattice, t, mu, delta_tau);
    InteractionConfig config(num_sites, num_slices, U, delta_tau);
    config.randomize();

    // Initialize with stabilization turned ON
    GreensEngine engine(num_sites, num_slices, k, true, 5);
    DqmcSampler sampler(num_sites, num_slices, k, 12345);

    int warmups = 5;
    int timed_sweeps = 50;
    int reinversion_frequency = 2; // Forces frequent scratch recalculations

    std::cout << "Warming up system...\n";
    for(int s = 0; s < warmups; ++s) {
        sampler.perform_sweep(config, engine);
        engine.recompute_all_from_scratch(config);
    }

    std::cout << "Running " << timed_sweeps << " timed sweeps with frequent re-inversions...\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int sweep = 0; sweep < timed_sweeps; ++sweep) {
        sampler.perform_sweep(config, engine);
        if ((sweep + 1) % reinversion_frequency == 0) {
            engine.recompute_all_from_scratch(config);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "-------------------------------------------\n";
    std::cout << "Total Elapsed Time: " << elapsed.count() << " seconds.\n";
    std::cout << "Average Time per Sweep: " << (elapsed.count() / timed_sweeps) * 1000.0 << " ms.\n";
    std::cout << "-------------------------------------------\n";

    return 0;
}