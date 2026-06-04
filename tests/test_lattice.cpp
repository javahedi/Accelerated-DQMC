#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>

#include "ILattice.hpp"
#include "SquareLattice.hpp"

bool has_neighbor(const std::vector<int>& neighbors, int target) {
    return std::find(neighbors.begin(), neighbors.end(), target) != neighbors.end();
}

void test_square_lattice_2x2() {
    std::cout << "Running test_square_lattice_2x2... ";

    // Instantiate a 2x2 periodic square lattice
    ILattice* lattice = new SquareLattice(2, 2, true); 
    
    // Assert total site count N = 4
    assert(lattice->num_sites() == 4);

    // Get neighbors for site 0 (which is at x=0, y=0)
    std::vector<int> n0 = lattice->get_neighbors(0);

    std::cout << "\nneighbors of 0:";
    for (int x : n0) 
        std::cout << " " << x;
        std::cout << std::endl;
    
    // Under periodic boundary conditions on a 2x2 grid:
    // Right neighbor is x=1, y=0 -> index 1
    // Up neighbor is x=0, y=1 -> index 2
    assert(has_neighbor(n0, 1));
    assert(has_neighbor(n0, 2));
    
    delete lattice;
    std::cout << "PASSED!" << std::endl;
}

int main() {
    test_square_lattice_2x2();
    std::cout << "All tests completed successfully." << std::endl;
    return 0;
}