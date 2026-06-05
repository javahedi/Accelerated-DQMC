#include <iostream>
#include <cassert>
#include <cmath>

#include <Eigen/Dense>

#include "SquareLattice.hpp"
#include "Measurement.hpp"

bool is_close(double a, double b, double tol = 1e-9) {
    return std::abs(a - b) < tol;
}

void test_local_observables() {
    std::cout << "Running test_local_observables... ";

    int width = 2;
    int height = 1;
    int N = width * height;

    SquareLattice lattice(width, height, false);
    Measurement meas(N);

    Eigen::MatrixXd G_up = Eigen::MatrixXd::Zero(N, N);
    Eigen::MatrixXd G_down = Eigen::MatrixXd::Zero(N, N);

    // n_up = 1 - G_up(ii)
    // n_down = 1 - G_down(ii)
    //
    // Site 0: n_up = 0.8, n_down = 0.4
    // Site 1: n_up = 0.2, n_down = 0.6
    G_up(0, 0) = 0.2;
    G_up(1, 1) = 0.8;

    G_down(0, 0) = 0.6;
    G_down(1, 1) = 0.4;

    meas.sample(G_up, G_down, lattice);
    meas.normalize();

    double expected_density =
        ((0.8 + 0.4) + (0.2 + 0.6)) / 2.0;

    double expected_double_occ =
        ((0.8 * 0.4) + (0.2 * 0.6)) / 2.0;

    double expected_moment_site0 =
        0.8 + 0.4 - 2.0 * 0.8 * 0.4;

    double expected_moment_site1 =
        0.2 + 0.6 - 2.0 * 0.2 * 0.6;

    double expected_local_moment =
        (expected_moment_site0 + expected_moment_site1) / 2.0;

    assert(is_close(meas.get_avg_density(), expected_density));
    assert(is_close(meas.get_avg_double_occ(), expected_double_occ));
    assert(is_close(meas.get_avg_local_moment(), expected_local_moment));
    assert(meas.get_num_samples() == 1);

    std::cout << "PASSED!" << std::endl;
}

void test_multiple_samples_average() {
    std::cout << "Running test_multiple_samples_average... ";

    int N = 2;
    SquareLattice lattice(2, 1, false);
    Measurement meas(N);

    Eigen::MatrixXd G_up_1 = Eigen::MatrixXd::Zero(N, N);
    Eigen::MatrixXd G_down_1 = Eigen::MatrixXd::Zero(N, N);

    G_up_1(0, 0) = 0.2;
    G_up_1(1, 1) = 0.8;
    G_down_1(0, 0) = 0.6;
    G_down_1(1, 1) = 0.4;

    Eigen::MatrixXd G_up_2 = Eigen::MatrixXd::Zero(N, N);
    Eigen::MatrixXd G_down_2 = Eigen::MatrixXd::Zero(N, N);

    G_up_2(0, 0) = 0.5;
    G_up_2(1, 1) = 0.5;
    G_down_2(0, 0) = 0.5;
    G_down_2(1, 1) = 0.5;

    meas.sample(G_up_1, G_down_1, lattice);
    meas.sample(G_up_2, G_down_2, lattice);
    meas.normalize();

    double density_sample_1 =
        ((0.8 + 0.4) + (0.2 + 0.6)) / 2.0;

    double density_sample_2 =
        ((0.5 + 0.5) + (0.5 + 0.5)) / 2.0;

    double expected_density =
        (density_sample_1 + density_sample_2) / 2.0;

    assert(is_close(meas.get_avg_density(), expected_density));
    assert(meas.get_num_samples() == 2);

    std::cout << "PASSED!" << std::endl;
}

void test_nearest_neighbor_spin_corr() {
    std::cout << "Running test_nearest_neighbor_spin_corr... ";

    int N = 2;
    SquareLattice lattice(2, 1, false);
    Measurement meas(N);

    Eigen::MatrixXd G_up = Eigen::MatrixXd::Zero(N, N);
    Eigen::MatrixXd G_down = Eigen::MatrixXd::Zero(N, N);

    G_up(1, 0) = 0.3;
    G_down(0, 1) = 0.2;

    meas.sample(G_up, G_down, lattice);
    meas.normalize();

    double expected =
        -G_up(1, 0) * G_down(0, 1);

    assert(is_close(meas.get_nearest_neighbor_spin_corr(), expected));

    std::cout << "PASSED!" << std::endl;
}

void test_af_structure_factor_two_sites() {
    std::cout << "Running test_af_structure_factor_two_sites... ";

    int N = 2;
    SquareLattice lattice(2, 1, false);
    Measurement meas(N);

    Eigen::MatrixXd G_up = Eigen::MatrixXd::Zero(N, N);
    Eigen::MatrixXd G_down = Eigen::MatrixXd::Zero(N, N);

    // Occupations:
    // n_up = n_down = 0.5 on both sites
    // local moment per site = 0.5 + 0.5 - 2 * 0.25 = 0.5
    G_up(0, 0) = 0.5;
    G_up(1, 1) = 0.5;
    G_down(0, 0) = 0.5;
    G_down(1, 1) = 0.5;

    // Off-diagonal transverse correlations.
    G_up(1, 0) = 0.3;
    G_down(0, 1) = 0.2;

    G_up(0, 1) = 0.4;
    G_down(1, 0) = 0.1;

    meas.sample(G_up, G_down, lattice);
    meas.normalize();

    double corr_00 = 0.5;
    double corr_11 = 0.5;
    double corr_01 = -G_up(1, 0) * G_down(0, 1);
    double corr_10 = -G_up(0, 1) * G_down(1, 0);

    // For 2x1 lattice, phases:
    // phase(0,0)=+1
    // phase(1,1)=+1
    // phase(0,1)=-1
    // phase(1,0)=-1
    double expected =
        (corr_00 + corr_11 - corr_01 - corr_10) / 2.0;

    assert(is_close(meas.get_af_structure_factor(), expected));

    std::cout << "PASSED!" << std::endl;
}

int main() {
    test_local_observables();
    test_multiple_samples_average();
    test_nearest_neighbor_spin_corr();
    test_af_structure_factor_two_sites();

    std::cout << "Measurement tests completed successfully." << std::endl;

    return 0;
}