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

struct ChainDiagnostics {
    int accepted_samples = 0;
    int rejected_bad_green = 0;

    double min_n = 1.0e100;
    double max_n = -1.0e100;
    double max_abs_G = 0.0;
};

struct Fig3Result {
    double T;
    double beta;
    int num_slices;
    double delta_tau;

    double density;
    double double_occ;
    double local_moment;

    double nn_spin_corr;
    double structure_factor;

    int accepted_samples;
    int rejected_bad_green;

    double min_n;
    double max_n;
    double max_abs_G;
};

bool diagnose_green(
    const Eigen::MatrixXd& G_up,
    const Eigen::MatrixXd& G_down,
    ChainDiagnostics& diag
) {
    int num_sites = static_cast<int>(G_up.rows());

    bool bad = false;

    double local_min_n = 1.0e100;
    double local_max_n = -1.0e100;

    for (int i = 0; i < num_sites; ++i) {
        double n_up = 1.0 - G_up(i, i);
        double n_down = 1.0 - G_down(i, i);

        local_min_n = std::min(local_min_n, std::min(n_up, n_down));
        local_max_n = std::max(local_max_n, std::max(n_up, n_down));

        if (!std::isfinite(n_up) || !std::isfinite(n_down)) {
            bad = true;
        }
    }

    double local_max_abs_G =
        std::max(
            G_up.cwiseAbs().maxCoeff(),
            G_down.cwiseAbs().maxCoeff()
        );

    if (!std::isfinite(local_max_abs_G)) {
        bad = true;
    }

    diag.min_n = std::min(diag.min_n, local_min_n);
    diag.max_n = std::max(diag.max_n, local_max_n);
    diag.max_abs_G = std::max(diag.max_abs_G, local_max_abs_G);

    // Loose sanity bounds. Any large violation means Green's function is corrupt.
    if (
        local_min_n < -1.0e-6 ||
        local_max_n > 1.0 + 1.0e-6 ||
        local_max_abs_G > 10.0
    ) {
        bad = true;
    }

    return !bad;
}

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

    std::vector<double> thread_density(num_threads, 0.0);
    std::vector<double> thread_double_occ(num_threads, 0.0);
    std::vector<double> thread_local_moment(num_threads, 0.0);
    std::vector<double> thread_nn(num_threads, 0.0);
    std::vector<double> thread_sf(num_threads, 0.0);

    std::vector<int> thread_accepted(num_threads, 0);
    std::vector<int> thread_rejected_bad_green(num_threads, 0);

    std::vector<double> thread_min_n(num_threads, 1.0e100);
    std::vector<double> thread_max_n(num_threads, -1.0e100);
    std::vector<double> thread_max_abs_G(num_threads, 0.0);

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

        // GreensEngine engine(
        //     num_sites,
        //     num_slices,
        //     k,
        //     false
        // );
        GreensEngine engine(
            num_sites,
            num_slices,
            k,
            true,  // stabilize
            5      // stabilization frequency
        );

        engine.recompute_all_from_scratch(config);

        DqmcSampler sampler(
            num_sites,
            num_slices,
            k,
            base_seed + static_cast<unsigned int>(thread_id * 1701)
        );

        Measurement measurements(num_sites);
        ChainDiagnostics diag;

        for (int sweep = 0; sweep < eq_sweeps_per_thread; ++sweep) {
            sampler.perform_sweep(config, engine);
            engine.recompute_all_from_scratch(config);
        }

        for (int sweep = 0; sweep < meas_sweeps_per_thread; ++sweep) {
            sampler.perform_sweep(config, engine);

            engine.recompute_all_from_scratch(config);

            bool green_ok =
                diagnose_green(
                    engine.G_up(),
                    engine.G_down(),
                    diag
                );

            if (!green_ok) {
                diag.rejected_bad_green++;

                if (diag.rejected_bad_green <= 5) {
                    #pragma omp critical
                    {
                        std::cerr
                            << "\nBAD GREEN"
                            << " | T=" << T
                            << " | thread=" << thread_id
                            << " | sweep=" << sweep
                            << " | min_n=" << diag.min_n
                            << " | max_n=" << diag.max_n
                            << " | max_abs_G=" << diag.max_abs_G
                            << "\n";
                    }
                }

                continue;
            }

            measurements.sample(
                engine.G_up(),
                engine.G_down(),
                lattice
            );

            diag.accepted_samples++;
        }

        if (diag.accepted_samples > 0) {
            measurements.normalize();

            thread_density[thread_id] =
                measurements.get_avg_density();

            thread_double_occ[thread_id] =
                measurements.get_avg_double_occ();

            thread_local_moment[thread_id] =
                measurements.get_avg_local_moment();

            thread_nn[thread_id] =
                measurements.get_nearest_neighbor_spin_corr();

            thread_sf[thread_id] =
                measurements.get_af_structure_factor();
        }

        thread_accepted[thread_id] =
            diag.accepted_samples;

        thread_rejected_bad_green[thread_id] =
            diag.rejected_bad_green;

        thread_min_n[thread_id] =
            diag.min_n;

        thread_max_n[thread_id] =
            diag.max_n;

        thread_max_abs_G[thread_id] =
            diag.max_abs_G;
    }

    double final_density = 0.0;
    double final_double_occ = 0.0;
    double final_local_moment = 0.0;
    double final_nn = 0.0;
    double final_sf = 0.0;

    int valid_threads = 0;
    int total_accepted = 0;
    int total_rejected_bad_green = 0;

    double global_min_n = 1.0e100;
    double global_max_n = -1.0e100;
    double global_max_abs_G = 0.0;

    for (int i = 0; i < num_threads; ++i) {
        total_accepted += thread_accepted[i];
        total_rejected_bad_green += thread_rejected_bad_green[i];

        global_min_n = std::min(global_min_n, thread_min_n[i]);
        global_max_n = std::max(global_max_n, thread_max_n[i]);
        global_max_abs_G = std::max(global_max_abs_G, thread_max_abs_G[i]);

        if (thread_accepted[i] > 0) {
            final_density += thread_density[i];
            final_double_occ += thread_double_occ[i];
            final_local_moment += thread_local_moment[i];
            final_nn += thread_nn[i];
            final_sf += thread_sf[i];
            valid_threads++;
        }
    }

    if (valid_threads > 0) {
        final_density /= static_cast<double>(valid_threads);
        final_double_occ /= static_cast<double>(valid_threads);
        final_local_moment /= static_cast<double>(valid_threads);
        final_nn /= static_cast<double>(valid_threads);
        final_sf /= static_cast<double>(valid_threads);
    }

    return Fig3Result{
        T,
        beta,
        num_slices,
        delta_tau,
        final_density,
        final_double_occ,
        final_local_moment,
        final_nn,
        final_sf,
        total_accepted,
        total_rejected_bad_green,
        global_min_n,
        global_max_n,
        global_max_abs_G
    };
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "Starting DQMC magnetic observable scan with diagnostics\n";
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
        << "density,double_occupancy,local_moment,"
        << "NN_Spin_Correlation,AF_Structure_Factor,"
        << "accepted_samples,rejected_bad_green,"
        << "min_n,max_n,max_abs_G\n";

    std::cout
        << std::setw(8)  << "T"
        << std::setw(10) << "Beta"
        << std::setw(8)  << "L_tau"
        << std::setw(12) << "dtau"
        << std::setw(12) << "dens"
        << std::setw(12) << "docc"
        << std::setw(12) << "moment"
        << std::setw(16) << "<SiSj>"
        << std::setw(16) << "S(pi,pi)"
        << std::setw(10) << "badG"
        << std::setw(14) << "max|G|"
        << "\n";

    std::cout
        << "------------------------------------------------------------------------------------------------\n";

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
            << std::setw(8)  << res.num_slices
            << std::setw(12) << std::setprecision(5) << res.delta_tau
            << std::setw(12) << std::setprecision(6) << res.density
            << std::setw(12) << std::setprecision(6) << res.double_occ
            << std::setw(12) << std::setprecision(6) << res.local_moment
            << std::setw(16) << std::setprecision(8) << res.nn_spin_corr
            << std::setw(16) << std::setprecision(8) << res.structure_factor
            << std::setw(10) << res.rejected_bad_green
            << std::setw(14) << std::setprecision(5) << res.max_abs_G
            << "\n";

        csv_file
            << std::setprecision(12)
            << res.T << ","
            << res.beta << ","
            << res.num_slices << ","
            << res.delta_tau << ","
            << res.density << ","
            << res.double_occ << ","
            << res.local_moment << ","
            << res.nn_spin_corr << ","
            << res.structure_factor << ","
            << res.accepted_samples << ","
            << res.rejected_bad_green << ","
            << res.min_n << ","
            << res.max_n << ","
            << res.max_abs_G << "\n";

        csv_file.flush();
    }

    csv_file.close();

    std::cout
        << "\n[Success] Data file written to "
        << "'run_structureFactor_scan.csv'.\n";

    return 0;
}