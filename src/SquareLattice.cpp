#include "SquareLattice.hpp"
#include <stdexcept> // For std::invalid_argument

SquareLattice::SquareLattice(int width, int height, bool periodic) 
: width_(width), height_(height), periodic_(periodic) {
    if (width <= 0 || height <= 0) {
        throw std::invalid_argument("Lattice dimensions must be positive integers.");
    }
    num_sites_ = width_ * height_;
    compute_neighbors();
}


int SquareLattice::xy_to_index(int x, int y) const {
    return x + (y * width_);
}

void SquareLattice::compute_neighbors() {
    neighbors_.resize(num_sites_);
    for (int y = 0; y < height_; ++y){
        for (int x = 0; x < width_; ++x){
            int current_site = xy_to_index(x,y);
            int dx[] = {1, -1, 0, 0};
            int dy[] = {0, 0, 1, -1};

            for (int d =0; d < 4; ++d) {
                int nx = x + dx[d];
                int ny = y + dy[d];

                if (periodic_) {
                    nx = (nx % width_  + width_) % width_;
                    ny = (ny % height_ + height_) % height_;
                    neighbors_[current_site].push_back(xy_to_index(nx,ny));
                }else {
                    if (nx >= 0 && nx < width_ && ny >=0 && ny < height_) {
                        neighbors_[current_site].push_back(xy_to_index(nx,ny));
                    }
                }
            }
        }
    }
}


int SquareLattice::num_sites() const{
    return num_sites_;
}

std::vector<int> SquareLattice::get_neighbors(int site_index) const {
    return neighbors_[site_index];
}