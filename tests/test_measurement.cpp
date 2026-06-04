#include <iostream>
#include <cassert>
#include <cmath>
#include <Eigen/Dense>

#include "Measurement.hpp"

void test_measurement_basic() {
    std::cout << "Running test_measurement_basic... ";

    int N = 2;

    Eigen::MatrixXd G_up(N, N);
    Eigen::MatrixXd G_down(N, N);

    G_up << 0.25, 0.0,
            0.0, 0.75;

    G_down << 0.50, 0.0,
              0.0, 0.25;

    Measurement measurement(N);

    measurement.sample(G_up, G_down);
    measurement.normalize();

    // Site 0:
    // n_up = 1 - 0.25 = 0.75
    // n_down = 1 - 0.50 = 0.50
    // double_occ = 0.75 * 0.50 = 0.375
    // local_moment = 0.75 + 0.50 - 2*0.375 = 0.50
    //
    // Site 1:
    // n_up = 1 - 0.75 = 0.25
    // n_down = 1 - 0.25 = 0.75
    // double_occ = 0.25 * 0.75 = 0.1875
    // local_moment = 0.25 + 0.75 - 2*0.1875 = 0.625

    double expected_avg_density =
        ((0.75 + 0.50) + (0.25 + 0.75)) / 2.0;

    double expected_avg_double_occ =
        (0.375 + 0.1875) / 2.0;

    double expected_avg_local_moment =
        (0.50 + 0.625) / 2.0;

    assert(std::abs(measurement.get_avg_density() - expected_avg_density) < 1e-12);
    assert(std::abs(measurement.get_avg_double_occ() - expected_avg_double_occ) < 1e-12);
    assert(std::abs(measurement.get_avg_local_moment() - expected_avg_local_moment) < 1e-12);

    std::cout << "PASSED!" << std::endl;
}

void test_measurement_multiple_samples() {
    std::cout << "Running test_measurement_multiple_samples... ";

    int N = 1;

    Eigen::MatrixXd G_up_1(1, 1);
    Eigen::MatrixXd G_down_1(1, 1);
    G_up_1(0, 0) = 0.2;
    G_down_1(0, 0) = 0.4;

    Eigen::MatrixXd G_up_2(1, 1);
    Eigen::MatrixXd G_down_2(1, 1);
    G_up_2(0, 0) = 0.6;
    G_down_2(0, 0) = 0.8;

    Measurement measurement(N);

    measurement.sample(G_up_1, G_down_1);
    measurement.sample(G_up_2, G_down_2);
    measurement.normalize();

    // Sample 1:
    // n_up = 0.8, n_down = 0.6
    // density = 1.4
    // double_occ = 0.48
    // moment = 1.4 - 2*0.48 = 0.44
    //
    // Sample 2:
    // n_up = 0.4, n_down = 0.2
    // density = 0.6
    // double_occ = 0.08
    // moment = 0.6 - 2*0.08 = 0.44

    double expected_avg_density = (1.4 + 0.6) / 2.0;
    double expected_avg_double_occ = (0.48 + 0.08) / 2.0;
    double expected_avg_local_moment = (0.44 + 0.44) / 2.0;

    assert(std::abs(measurement.get_avg_density() - expected_avg_density) < 1e-12);
    assert(std::abs(measurement.get_avg_double_occ() - expected_avg_double_occ) < 1e-12);
    assert(std::abs(measurement.get_avg_local_moment() - expected_avg_local_moment) < 1e-12);

    std::cout << "PASSED!" << std::endl;
}

int main() {
    test_measurement_basic();
    test_measurement_multiple_samples();

    std::cout << "Measurement tests completed successfully." << std::endl;
    return 0;
}