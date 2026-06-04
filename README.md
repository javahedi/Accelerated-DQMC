# Determinant Quantum Monte Carlo (DQMC) Engine

A high-performance, numerically stabilized C++ implementation of the Determinant Quantum Monte Carlo (DQMC) algorithm for the two-dimensional, single-band Hubbard model on a square lattice.

This framework is optimized for modern hardware—leveraging hardware-accelerated linear algebra backends, zero-allocation loops, and multi-core thread parallelization—to investigate strongly correlated electronic systems at low temperatures without numerical underflow or floating-point degradation.

---

## 🌌 Physics Overview & Methodology

This software simulates the **Hubbard Model**, a fundamental Hamiltonian used to describe the transition between conducting metals and insulating materials (Mott insulators), magnetic ordering, and unconventional superconductivity in solid-state systems.

### The Hamiltonian

$$H = -t \sum_{\langle i,j \rangle, \sigma} \left( c^\dagger_{i\sigma} c_{j\sigma} + \text{h.c.} \right) + U \sum_{i} \left(n_{i\uparrow} - \frac{1}{2}\right)\left(n_{i\downarrow} - \frac{1}{2}\right) - \mu \sum_{i, \sigma} n_{i\sigma}$$

Where:

* $t$ is the hopping parameter (kinetic energy of electrons jumping between nearest-neighbor sites $\langle i,j \rangle$).
* $U$ is the on-site Coulomb repulsion energy. The interaction term is symmetrically shifted around $\mu=0$ to ensure **half-filling** (particle-hole symmetry).
* $\mu$ is the chemical potential controlling system density.

### The DQMC Algorithm

Because the interaction term contains products of four creation/annihilation operators ($n_{i\uparrow}n_{i\downarrow}$), the system cannot be solved exactly. DQMC maps this interacting quantum problem in $D$ dimensions into a non-interacting quantum problem coupled to a classical fluctuating auxiliary field in $D+1$ dimensions:

1. **Trotter-Suzuki Decomposition:** The imaginary time interval $\beta = 1/T$ is sliced into $L$ discrete intervals $\Delta\tau = \beta/L$. The exponential partition function is decoupled: $e^{-\beta H} \approx \prod^L e^{-\Delta\tau K} e^{-\Delta\tau V}$.
2. **Discrete Hubbard-Stratonovich Transformation:** The on-site interaction is replaced by a coupling to a classical auxiliary Ising spin field $s_{i,l} = \pm 1$ at every spatial site $i$ and time slice $l$.
3. **Fermionic Determinants:** Integrating out the fermionic degrees of freedom yields a weight for sampling the classical field configuration based on the product of up and down spin determinants:

$$w(s) = \det[I + B_L B_{L-1} \dots B_1]_\uparrow \det[I + B_L B_{L-1} \dots B_1]_\downarrow$$


4. **Metropolis Sampling:** Auxiliary spins are flipped locally. The accept/reject ratio is evaluated efficiently through the equal-time Green's function matrix $G^\sigma = (I + B_L \dots B_1)^{-1}$ via Sherman-Morrison rank-1 updates without fully recomputing the determinants from scratch.

---

## 🛠️ Software Architecture & Optimization

From a software engineering perspective, DQMC scales as $\mathcal{O}(L \cdot N^3)$, where $N$ is the total number of lattice sites and $L$ is the number of imaginary time slices. At large coupling $U$ or low temperatures (large $\beta$), the matrices $B_l$ develop exponential scales, leading to severe numerical underflow. This engine is highly optimized to handle these challenges.

### Core Optimization Stack

1. **UDV Matrix Stabilization Loop (Numerical Invariance):**
To calculate $G$ without catastrophic floating-point truncation, the product chain $B_L \dots B_1$ is periodically re-orthogonalized via pivoted Householder QR decompositions, factoring the scales into a diagonal matrix $D$:

$$B_L \dots B_1 = U \cdot D \cdot V$$



This preserves a dynamic numerical range spanning hundreds of orders of magnitude within standard IEEE double-precision limits.

2. **Zero-Allocation Workspace Performance:**
Memory allocations (`new` heap cycles) inside hot loops are completely eliminated. The `GreensEngine` manages persistent, mutable scratchpad matrices (`U_workspace_`, `V_workspace_`, etc.) allocated exactly once during initialization and reused continuously.

3. **Hardware-Accelerated Linear Algebra (BLAS Integration):**
By compiling with the `EIGEN_USE_BLAS` preprocessor flag, Eigen's default C++ evaluation loops are bypassed. Heavy $\mathcal{O}(N^3)$ operations are offloaded directly to vendor-optimized BLAS backends. On macOS, this links directly to the **Apple Accelerate Framework (vecLib)**, utilizing hardware matrix coprocessors (AMX).
    
4. **Embarrassingly Parallel Markov Chains:**
Instead of introducing thread synchronization overhead within the strict sequential updates of a single configuration, the engine implements parallel Markov chains using **OpenMP**. The simulation forks independent, non-interacting paths across all available CPU threads using offset random seed arrays, combining normalized physical observables at the end of execution.

---

## 📦 Repository Structure

```text
├── CMakeLists.txt              # Build configuration with automatic toolchain targeting
├── include/                    # Header files
│   ├── SquareLattice.hpp       # Geometry boundary configurations
│   ├── KineticBuilder.hpp      # Sparse-to-dense kinetic matrix compilers
│   ├── InteractionConfig.hpp   # Aux-field state tracking and exponential diagonals
│   ├── GreensEngine.hpp        # Matrix product chains and UDV decomposition loops
│   ├── DqmcSampler.hpp         # Metropolis sweep execution & Sherman-Morrison solver
│   └── Measurement.hpp         # Observable accumulator (density, double-occ, moments)
├── src/                        # Implementation files
│   ├── main.cpp                # Standard validation executable
│   └── run_local_moment_scan.cpp # Multi-parameter sweeping pipeline
├── tests/                      # Unit-testing and performance validation binaries
│   ├── test_greens.cpp         # Compares stabilized paths to analytic non-interacting limits
│   └── benchmark_step1.cpp     # Execution clocking and micro-benchmarks
└── external/
    └── eigen-3.4.0/            # Header-only Eigen linear algebra dependency

```

---

## 🚀 Quick Start & Building

### Prerequisites

* A C++17 compliant compiler
* CMake (v3.12 or higher)
* **macOS users:** Homebrew LLVM is recommended for native OpenMP support (`brew install llvm`).

### Build Instructions

The `CMakeLists.txt` is pre-configured to automatically intercept Apple Silicon setups, route paths to the Homebrew LLVM compiler, and activate OpenMP and Accelerate backends without requiring manually passed environment variables:

```bash
# Clone the repository
git clone https://github.com/yourusername/project-DQMC.git
cd project-DQMC

# Create and enter build folder
mkdir build && cd build

# Configure and compile using the optimized profiles
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .

```

### Running the Sweeping Scan Pipeline

To execute the multi-parameter scan across varying interaction strengths $U$ and target temperatures $T$ to measure the formation of the local magnetic moment:

```bash
./run_local_moment_scan

```

This generates a structured data file `local_moment_scan.csv` tracking:


$$\langle n \rangle, \quad \langle n_\uparrow n_\downarrow \rangle, \quad \langle m_z^2 \rangle = \langle (n_\uparrow - n_\downarrow)^2 \rangle$$

---

## 📊 Verification Benchmarks

You can verify that both the direct product engine and the stabilized UDV engine match exact analytical thermodynamic limits by running the test suite:

```bash
./test_greens

```

To check system performance scaling:

```bash
./benchmark_step1

```