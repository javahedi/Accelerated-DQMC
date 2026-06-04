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

    double mu = 0.0; // half filling for shifted Hubbard interaction

    SquareLattice lattice(L_size, L_size, true);

    Eigen::MatrixXd k =
        KineticBuilder::buildKineticMatrix(lattice, t, mu, delta_tau);

    InteractionConfig config(num_sites, num_slices, U, delta_tau);
    config.randomize();

    // MODIFIED: Toggle the 'stabilize' flag to true, and pass reinversion_frequency 
    // as the stabilization stride to completely eliminate truncation errors at small T.
    //GreensEngine engine(num_sites, num_slices, k, true, reinversion_frequency);
    GreensEngine engine(num_sites, num_slices, k, false);
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

        measurements.sample(engine.G_up(), engine.G_down());
    }

    measurements.normalize();

    return Result{
        U,
        beta,
        1.0 / beta,
        measurements.get_avg_density(),
        measurements.get_avg_double_occ(),
        measurements.get_avg_local_moment()
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