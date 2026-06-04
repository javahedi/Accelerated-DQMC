#pragma once
#include "ILattice.hpp"
#include <Eigen/Dense>

class KineticBuilder
{
public:
    static Eigen::MatrixXd buildKineticMatrix(
        const ILattice& lattice,
        double t,
        double mu,
        double delta_tau);
};