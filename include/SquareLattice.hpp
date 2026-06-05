#pragma once
#include <vector>
#include "ILattice.hpp"


class SquareLattice : public ILattice {
private:
    int width_;
    int height_;
    bool periodic_;
    int num_sites_;
    std::vector<std::vector<int>> neighbors_;

    int xy_to_index(int x, int y) const;
    void compute_neighbors();

public:
    SquareLattice(int width, int height, bool periodic);
    int num_sites() const override;
    std::vector<int> get_neighbors(int site_index) const override;

    int get_x(int site_index) const;
    int get_y(int site_index) const;
};