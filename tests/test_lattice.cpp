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
    std::cout << "Running test_square_lattice_2x2... \n";

    ILattice* lattice = new SquareLattice(2, 2, true); 
    
    assert(lattice->num_sites() == 4);

    std::vector<int> n0 = lattice->get_neighbors(0);

    std::cout << "neighbors of 0:";
    for (int x : n0) std::cout << " " << x;
    std::cout << std::endl;
    
    // In a 2x2 periodic grid, every site is adjacent to exactly 2 distinct sites.
    // (Site 0 is connected to Site 1 and Site 2. Site 3 is diagonally opposite)
    assert(n0.size() == 2); 
    assert(has_neighbor(n0, 1));
    assert(has_neighbor(n0, 2));
    assert(!has_neighbor(n0, 0)); // No self-loops
    assert(!has_neighbor(n0, 3)); // No diagonal hopping
    
    delete lattice;
    std::cout << "PASSED!" << std::endl;
}

int main() {
    test_square_lattice_2x2();
    std::cout << "All tests completed successfully." << std::endl;
    return 0;
}