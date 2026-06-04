// src/main.cpp
#include <iostream>
#include "SquareLattice.hpp"
#include "KineticBuilder.hpp"
#include "InteractionConfig.hpp"
#include "GreensEngine.hpp"
#include "DqmcSampler.hpp"
#include "Measurement.hpp"

int main() {
    std::cout << "Starting Determinant Quantum Monte Carlo (DQMC) Simulation...\n";

    // 1. Simulation Parameters
    int L_size = 4;          // 4x4 Square Lattice
    int num_sites = L_size * L_size;
    int num_slices = 10;     // Imaginary time slices (L)
    double beta = 1.0;       // Inverse Temperature
    double delta_tau = beta / num_slices;
    double t = 1.0;          // Hopping parameter
    double U = 4.0;          // On-site Coulomb repulsion
    double mu = 0.0;         // Half-filling chemical potential (for repulsive model)

    int equilibration_sweeps = 200;
    int measurement_sweeps = 1000;
    int reinversion_frequency = 5; // Recompute from scratch every 5 sweeps to avoid numerical drift

    // 2. Initialize Core Geometry & Hamiltonian Elements
    SquareLattice lattice(L_size, L_size, true); // 2D periodic lattice
    Eigen::MatrixXd k = KineticBuilder::buildKineticMatrix(lattice, t, mu, delta_tau);

    InteractionConfig config(num_sites, num_slices, U, delta_tau);
    GreensEngine engine(num_sites, num_slices, k);
    
    // Initial calculation from scratch
    engine.recompute_all_from_scratch(config);

    DqmcSampler sampler(num_sites, num_slices, k, 42); // Seeded random sampler
    Measurement measurements(num_sites);

    // 3. Equilibration Phase (Warm-up sweeps without sampling)
    std::cout << "Running " << equilibration_sweeps << " equilibration sweeps...\n";
    for (int sweep = 0; sweep < equilibration_sweeps; ++sweep) {
        sampler.perform_sweep(config, engine);
        
        // Periodically refresh Green's functions from scratch to eliminate rounding errors
        if (sweep % reinversion_frequency == 0) {
            engine.recompute_all_from_scratch(config);
        }
    }

    // 4. Production Phase (Sweeps with structural physical sampling)
    std::cout << "Running " << measurement_sweeps << " measurement sweeps...\n";
    for (int sweep = 0; sweep < measurement_sweeps; ++sweep) {
        sampler.perform_sweep(config, engine);
        
        if (sweep % reinversion_frequency == 0) {
            engine.recompute_all_from_scratch(config);
        }

        // Always sample properties at equal-time frame (tau = 0)
        measurements.sample(engine.G_up(), engine.G_down());
    }

    // 5. Normalize and Display Data Findings
    measurements.normalize();
    measurements.print_summary();

    return 0;
}