#pragma once
#include <vector>

class ILattice
{
public:
    virtual ~ILattice() = default;  
    virtual int num_sites() const = 0;
    virtual std::vector<int> get_neighbors(int site) const = 0;
};