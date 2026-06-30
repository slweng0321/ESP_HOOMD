// Copyright (c) 2024-2026 ESP Plugin Contributors.
// Released under the BSD 3-Clause License.
//
// ESPForceCompute.cc
//
// CPU implementation of the Ewald Summation with Prolates (ESP) method for
// HOOMD-blue as an independent custom plugin.
//
// Architecture mirrors PPPMForceCompute (HOOMD-blue md component) while
// replacing the Gaussian / B-spline–specific pieces with PSWF-based
// coefficient builders and a real-space piecewise-polynomial lookup table
// for the short-range correction potential L(r).
//
// Reference:
//   Liang, Lu, Barnett, Greengard, Jiang,
//   "Accelerating molecular dynamics simulations using fast Ewald summation
//    with prolates", Nat. Commun. 2026.
//   https://doi.org/10.1038/s41467-026-73232-8

#include "ESPForceCompute.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "hoomd/BoxDim.h"
#include "hoomd/Index1D.h"

namespace hoomd
{
namespace md
{

// ============================================================================
// Anonymous namespace: file-local mathematical helpers
// ============================================================================
namespace
{

constexpr Scalar ESP_PI     = Scalar(3.14159265358979323846264338327950288);
constexpr Scalar ESP_TWO_PI = Scalar(2.0) * ESP_PI;

inline bool is_pow2(unsigned int n)
    {
    while (n && n % 2 == 0)
        n /= 2;
    return (n == 1);
    }

inline Scalar sqr(Scalar x) { return x * x; }

// Numerically stable sinc(x) = sin(x)/x
inline Scalar sinc(Scalar x)
    {
    if (std::fabs(x) < Scalar(1.0e-8))
        {
        Scalar x2 = x * x;
        return Scalar(1.0) - x2 / Scalar(6.0) + x2 * x2 / Scalar(120.0);
        }
    return std::sin(x) / x;
    }

inline unsigned int index_3d(unsigned int i,
                              unsigned int j,
                              unsigned int k,
                              const uint3& dim)
    {
    return (k * dim.y + j) * dim.x + i;
    }

inline int wrap_index(int idx, int dim)
    {
    int out = idx % dim;
    if (out < 0)
        out += dim;
    return out;
    }

inline Scalar safe_div(Scalar a, Scalar b, Scalar fallback = Scalar(0.0))
    {
    return (std::fabs(b) > Scalar(1.0e-12)) ? (a / b) : fallback;
    }

// ---------------------------------------------------------------------------
// Gauss–Legendre quadrature nodes and weights on [-1, 1] for n = 16
// (sufficient for smooth integrands of the PSWF self-energy)
// ---------------------------------------------------------------------------
static const Scalar GL16_nodes[16] = {
    Scalar(-0.9894009349916499),  Scalar(-0.9445750230732326),
    Scalar(-0.8656312023341870),  Scalar(-0.7554044083550030),
    Scalar(-0.6178762444026438),  Scalar(-0.4580167776572274),
    Scalar(-0.2816035507792589),  Scalar(-0.0950125098360223),
    Scalar( 0.0950125098360223),  Scalar( 0.2816035507792589),
    Scalar( 0.4580167776572274),  Scalar( 0.6178762444026438),
    Scalar( 0.7554044083550030),  Scalar( 0.8656312023341870),
    Scalar( 0.9445750230732326),  Scalar( 0.9894009349916499)
};
static const Scalar GL16_weights[16] = {
    Scalar(0.0271524594117541),  Scalar(0.0622535239386479),
    Scalar(0.0951585116824928),  Scalar(0.1246289712555339),
    Scalar(0.1495959888165767),  Scalar(0.1691565193950025),
    Scalar(0.1826034150449236),  Scalar(0.1894506104550685),
    Scalar(0.1894506104550685),  Scalar(0.1826034150449236),
    Scalar(0.1691565193950025),  Scalar(0.1495959888165767),
    Scalar(0.1246289712555339),  Scalar(0.0951585116824928),
    Scalar(0.0622535239386479),  Scalar(0.0271524594117541)
};

// Composite Gauss-Legendre integration over [a, b] using n_panels sub-panels
Scalar integrate_gl(const std::function<Scalar(Scalar)>& f,
                    Scalar a,
                    Scalar b,
                    unsigned int n_panels = 8)
    {
    const Scalar half = Scalar(0.5) * (b - a);
    const Scalar mid  = Scalar(0.5) * (b + a);
    const Scalar panel_width = (b - a) / Scalar(n_panels);
    Scalar total = Scalar(0.0);

    for (unsigned int p = 0; p < n_panels; ++p)
        {
        const Scalar pa = a + Scalar(p) * panel_width;
        const Scalar pb = pa + panel_width;
        const Scalar lh = Scalar(0.5) * (pb - pa);
        const Scalar lm = Scalar(0.5) * (pb + pa);
        for (unsigned int q = 0; q < 16; ++q)
            total += GL16_weights[q] * f(lm + lh * GL16_nodes[q]) * lh;
        }
    (void)half; (void)mid;
    return total;
    }

// Trapezoidal rule fallback (used in table building for L(r))
Scalar integrate_trapezoid(const std::function<Scalar(Scalar)>& f,
                           Scalar a,
                           Scalar b,
                           unsigned int n)
    {
    if (n < 2)
        n = 2;
    const Scalar h   = (b - a) / Scalar(n - 1);
    Scalar       sum = Scalar(0.5) * (f(a) + f(b));
    for (unsigned int i = 1; i < n - 1; ++i)
        sum += f(a + Scalar(i) * h);
    return sum * h;
    }

// ---------------------------------------------------------------------------
// Dense linear system solver (Gaussian elimination with partial pivoting)
// Used for fitting degree-(n-1) polynomial through n points.
// ---------------------------------------------------------------------------
std::vector<Scalar> solve_linear_system(std::vector<Scalar> A,
                                         std::vector<Scalar> b,
                                         unsigned int        n)
    {
    for (unsigned int col = 0; col < n; ++col)
        {
        // find pivot
        unsigned int pivot     = col;
        Scalar       pivot_abs = std::fabs(A[col * n + col]);
        for (unsigned int row = col + 1; row < n; ++row)
            {
            Scalar cand = std::fabs(A[row * n + col]);
            if (cand > pivot_abs)
                {
                pivot     = row;
                pivot_abs = cand;
                }
            }
        if (pivot_abs < Scalar(1.0e-14))
            throw std::runtime_error(
                "ESPForceCompute: Singular linear system in polynomial fit.");

        if (pivot != col)
            {
            for (unsigned int j = 0; j < n; ++j)
                std::swap(A[col * n + j], A[pivot * n + j]);
            std::swap(b[col], b[pivot]);
            }

        const Scalar diag = A[col * n + col];
        for (unsigned int j = col; j < n; ++j)
            A[col * n + j] /= diag;
        b[col] /= diag;

        for (unsigned int row = 0; row < n; ++row)
            {
            if (row == col)
                continue;
            const Scalar factor = A[row * n + col];
            if (std::fabs(factor) < Scalar(1.0e-20))
                continue;
            for (unsigned int j = col; j < n; ++j)
                A[row * n + j] -= factor * A[col * n + j];
            b[row] -= factor * b[col];
            }
        }
    return b;
    }

// Fit a degree-(n-1) polynomial through n (x, y) points.
// Returns coefficients [c0, c1, ..., c_{n-1}] such that
//   p(x) = c0 + c1*x + c2*x^2 + ... evaluated in Horner form.
std::vector<Scalar> fit_power_polynomial(const std::vector<Scalar>& x,
                                          const std::vector<Scalar>& y)
    {
    const unsigned int n = static_cast<unsigned int>(x.size());
    if (n != y.size() || n == 0)
        throw std::runtime_error(
            "ESPForceCompute: Invalid input to fit_power_polynomial().");

    std::vector<Scalar> A(n * n, Scalar(0.0));
    for (unsigned int i = 0; i < n; ++i)
        {
        Scalar xp = Scalar(1.0);
        for (unsigned int j = 0; j < n; ++j, xp *= x[i])
            A[i * n + j] = xp;
        }
    return solve_linear_system(A, y, n);
    }

// Evaluate polynomial with coefficients `coeffs` at `x` using Horner's rule.
template<class T>
T horner_eval(const std::vector<T>& coeffs, T x)
    {
    T val = T(0);
    for (int i = static_cast<int>(coeffs.size()) - 1; i >= 0; --i)
        val = coeffs[static_cast<size_t>(i)] + val * x;
    return val;
    }

} // anonymous namespace
// ============================================================================

// ----------------------------------------------------------------------------
// Constructor
// ----------------------------------------------------------------------------
ESPForceCompute::ESPForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                 std::shared_ptr<NeighborList>      nlist,
                                 std::shared_ptr<ParticleGroup>     group)
    : ForceCompute(sysdef),
      m_nlist(nlist),
      m_group(group),
      m_mesh_points(make_uint3(0, 0, 0)),
      m_global_dim(make_uint3(0, 0, 0)),
      m_n_ghost_cells(make_uint3(0, 0, 0)),
      m_grid_dim(make_uint3(0, 0, 0)),
      m_ghost_width(make_scalar3(Scalar(0), Scalar(0), Scalar(0))),
      m_ghost_offset(0),
      m_n_cells(0),
      m_n_inner_cells(0),
      m_radius(0),
      m_kappa(Scalar(0.0)),
      m_rcut(Scalar(0.0)),
      m_order(0),
      m_alpha(Scalar(0.0)),
    m_pswf_c(Scalar(0.0)),
      m_q(Scalar(0.0)),
      m_q2(Scalar(0.0)),
      m_need_initialize(true),
      m_params_set(false),
      m_box_changed(false),
      m_ptls_added_removed(false),
      m_body_energy(Scalar(0.0)),
      m_qstarsq(Scalar(0.0)),
      m_n_table_segments(ESP_TABLE_SEGMENTS),
      m_pswf_self_energy_const(Scalar(0.0)),
      m_kiss_fft(nullptr),
      m_kiss_ifft(nullptr),
      m_kiss_fft_initialized(false),
#ifdef ENABLE_MPI
      m_dfft_initialized(false),
#endif
      m_kiss_fft_initialized_priv(false)
    {
    if (!m_nlist)
        throw std::runtime_error("ESPForceCompute: NeighborList pointer is null.");
    if (!m_group)
        throw std::runtime_error("ESPForceCompute: ParticleGroup pointer is null.");

    m_pdata->getBoxChangeSignal()
        .connect<ESPForceCompute, &ESPForceCompute::setBoxChange>(this);
    m_pdata->getGlobalParticleNumberChangeSignal()
        .connect<ESPForceCompute, &ESPForceCompute::slotGlobalParticleNumberChange>(this);

    m_force.zeroFill();
    m_virial.zeroFill();
    }

// ----------------------------------------------------------------------------
// Destructor
// ----------------------------------------------------------------------------
ESPForceCompute::~ESPForceCompute()
    {
    m_pdata->getGlobalParticleNumberChangeSignal()
        .disconnect<ESPForceCompute, &ESPForceCompute::slotGlobalParticleNumberChange>(this);
    m_pdata->getBoxChangeSignal()
        .disconnect<ESPForceCompute, &ESPForceCompute::setBoxChange>(this);

    if (m_kiss_fft_initialized)
        {
        kiss_fft_free(m_kiss_fft);
        kiss_fft_free(m_kiss_ifft);
        kiss_fft_cleanup();
        m_kiss_fft             = nullptr;
        m_kiss_ifft            = nullptr;
        m_kiss_fft_initialized = false;
        }

#ifdef ENABLE_MPI
    if (m_dfft_initialized)
        {
        dfft_destroy_plan(m_dfft_plan_forward);
        dfft_destroy_plan(m_dfft_plan_inverse);
        m_dfft_initialized = false;
        }
#endif
    }

// ----------------------------------------------------------------------------
// setParams — validate and store all ESP algorithm parameters
// ----------------------------------------------------------------------------
void ESPForceCompute::setParams(unsigned int nx,
                                unsigned int ny,
                                unsigned int nz,
                                unsigned int order,
                                Scalar       kappa,
                                Scalar       rcut,
                                Scalar       alpha,
                                unsigned int n_table)
    {
    if (nx == 0 || ny == 0 || nz == 0)
        throw std::runtime_error("ESPForceCompute: Mesh dimensions must be positive.");

    if (order < 1 || order > ESP_MAX_ORDER)
        {
        std::ostringstream oss;
        oss << "ESPForceCompute: Invalid PSWF interpolation order " << order
            << ". Valid range is [1, " << ESP_MAX_ORDER << "].";
        throw std::runtime_error(oss.str());
        }

    if (kappa <= Scalar(0.0))
        throw std::runtime_error("ESPForceCompute: kappa must be positive.");
    if (rcut <= Scalar(0.0))
        throw std::runtime_error("ESPForceCompute: rcut must be positive.");
    if (n_table < 16)
        throw std::runtime_error(
            "ESPForceCompute: lookup table must contain at least 16 segments.");

    m_kappa            = kappa;
    m_rcut             = rcut;
    m_alpha            = alpha;
    m_order            = static_cast<int>(order);
    m_radius           = static_cast<unsigned int>((m_order + 1) / 2);
    m_n_table_segments = n_table;
    m_global_dim       = make_uint3(nx, ny, nz);
    m_mesh_points      = m_global_dim;

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        const Index3D& didx = m_pdata->getDomainDecomposition()->getDomainIndexer();
        if (!is_pow2(nx) || !is_pow2(ny) || !is_pow2(nz))
            throw std::runtime_error(
                "ESPForceCompute: distributed FFT mesh dimensions must be powers of two.");
        if (nx % didx.getW())
            throw std::runtime_error(
                "ESPForceCompute: nx must be divisible by processor-grid width.");
        if (ny % didx.getH())
            throw std::runtime_error(
                "ESPForceCompute: ny must be divisible by processor-grid height.");
        if (nz % didx.getD())
            throw std::runtime_error(
                "ESPForceCompute: nz must be divisible by processor-grid depth.");

        m_mesh_points.x = nx / didx.getW();
        m_mesh_points.y = ny / didx.getH();
        m_mesh_points.z = nz / didx.getD();
        }
#endif

    GPUArray<Scalar> rho_coeff(static_cast<size_t>(order)
                                   * static_cast<size_t>(2 * order + 1),
                               m_exec_conf);
    m_rho_coeff.swap(rho_coeff);

    GPUArray<Scalar> gf_b(order, m_exec_conf);
    m_gf_b.swap(gf_b);

    m_need_initialize = true;
    m_params_set      = true;
    m_box_changed     = true;
    }

// ----------------------------------------------------------------------------
// computeForces — main entry point called by HOOMD integrator each step
// ----------------------------------------------------------------------------
void ESPForceCompute::computeForces(uint64_t timestep)
    {
    if (!m_params_set)
        throw std::runtime_error(
            "ESPForceCompute: call setParams() before computeForces().");

    if (m_box_changed || m_ptls_added_removed)
        m_need_initialize = true;

    if (m_need_initialize)
        {
        setupMesh();
        initializeFFT();
        setupCoeffs();
        computeInfluenceFunction();
        m_need_initialize     = false;
        m_box_changed         = false;
        m_ptls_added_removed  = false;
        }

    // Zero force and virial buffers
    m_force.zeroFill();
    m_virial.zeroFill();

    {
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    for (unsigned int idx = 0; idx < m_pdata->getN(); ++idx)
        h_force.data[idx] = make_scalar4(Scalar(0), Scalar(0), Scalar(0), Scalar(0));
    }

    // Accumulate total and squared charges (needed for self-energy correction)
    m_q  = Scalar(0.0);
    m_q2 = Scalar(0.0);
    {
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(),
                                 access_location::host, access_mode::read);
    for (unsigned int idx = 0; idx < m_pdata->getN(); ++idx)
        {
        const Scalar q = h_charge.data[idx];
        m_q  += q;
        m_q2 += q * q;
        }
    }

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE, &m_q,  1, MPI_HOOMD_SCALAR,
                      MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &m_q2, 1, MPI_HOOMD_SCALAR,
                      MPI_SUM, m_exec_conf->getMPICommunicator());
        }
#endif

    // Five-step ESP pipeline
    assignParticles();
    updateMeshes();
    interpolateForces();
    computeBodyCorrection();
    const Scalar pe = computePE();
    computeVirial();
    fixExclusions();

    // Distribute reciprocal-space energy equally across group members
    const unsigned int group_size = m_group->getNumMembers();
    if (group_size > 0)
        {
        ArrayHandle<Scalar4> h_force(m_force,
                                     access_location::host, access_mode::readwrite);
        const Scalar pe_share = pe / Scalar(group_size);
        for (unsigned int g = 0; g < group_size; ++g)
            {
            const unsigned int idx = m_group->getMemberIndex(g);
            h_force.data[idx].w += pe_share;
            }
        }

    (void)timestep;
    }

// ============================================================================
// Mesh initialisation
// ============================================================================

void ESPForceCompute::setupMesh()
    {
    m_n_ghost_cells = computeGhostCellNum();

    m_grid_dim.x = m_mesh_points.x + 2 * m_n_ghost_cells.x;
    m_grid_dim.y = m_mesh_points.y + 2 * m_n_ghost_cells.y;
    m_grid_dim.z = m_mesh_points.z + 2 * m_n_ghost_cells.z;

    m_n_inner_cells = m_mesh_points.x * m_mesh_points.y * m_mesh_points.z;
    m_n_cells       = m_grid_dim.x * m_grid_dim.y * m_grid_dim.z;

    m_ghost_offset  = index_3d(m_n_ghost_cells.x,
                                m_n_ghost_cells.y,
                                m_n_ghost_cells.z,
                                m_grid_dim);

    const BoxDim  box = m_pdata->getBox();
    const Scalar3 L   = box.getL();
    m_ghost_width = make_scalar3(
        safe_div(L.x * Scalar(m_n_ghost_cells.x), Scalar(m_global_dim.x)),
        safe_div(L.y * Scalar(m_n_ghost_cells.y), Scalar(m_global_dim.y)),
        safe_div(L.z * Scalar(m_n_ghost_cells.z), Scalar(m_global_dim.z)));
    }

// ============================================================================
// FFT plan allocation and mesh buffer allocation
// ============================================================================

void ESPForceCompute::initializeFFT()
    {
    // Teardown any previously allocated plans
    if (m_kiss_fft_initialized)
        {
        kiss_fft_free(m_kiss_fft);
        kiss_fft_free(m_kiss_ifft);
        kiss_fft_cleanup();
        m_kiss_fft             = nullptr;
        m_kiss_ifft            = nullptr;
        m_kiss_fft_initialized = false;
        }
#ifdef ENABLE_MPI
    if (m_dfft_initialized)
        {
        dfft_destroy_plan(m_dfft_plan_forward);
        dfft_destroy_plan(m_dfft_plan_inverse);
        m_dfft_initialized = false;
        }
#endif

    // ------------------------------------------------------------------
    // Allocate mesh GPUArrays
    // ------------------------------------------------------------------
    { GPUArray<kiss_fft_cpx> tmp(m_n_cells,       m_exec_conf); m_mesh.swap(tmp);                }
    { GPUArray<kiss_fft_cpx> tmp(m_n_inner_cells,  m_exec_conf); m_fourier_mesh.swap(tmp);        }
    { GPUArray<kiss_fft_cpx> tmp(m_n_inner_cells,  m_exec_conf); m_fourier_mesh_G_x.swap(tmp);    }
    { GPUArray<kiss_fft_cpx> tmp(m_n_inner_cells,  m_exec_conf); m_fourier_mesh_G_y.swap(tmp);    }
    { GPUArray<kiss_fft_cpx> tmp(m_n_inner_cells,  m_exec_conf); m_fourier_mesh_G_z.swap(tmp);    }
    { GPUArray<kiss_fft_cpx> tmp(m_n_cells,        m_exec_conf); m_inv_fourier_mesh_x.swap(tmp);  }
    { GPUArray<kiss_fft_cpx> tmp(m_n_cells,        m_exec_conf); m_inv_fourier_mesh_y.swap(tmp);  }
    { GPUArray<kiss_fft_cpx> tmp(m_n_cells,        m_exec_conf); m_inv_fourier_mesh_z.swap(tmp);  }
    { GPUArray<Scalar>       tmp(m_n_inner_cells,  m_exec_conf); m_inf_f.swap(tmp);               }
    { GPUArray<Scalar3>      tmp(m_n_inner_cells,  m_exec_conf); m_k.swap(tmp);                   }
    { GPUArray<Scalar>       tmp(6*m_n_inner_cells, m_exec_conf); m_virial_mesh.swap(tmp);         }
    {
    GPUArray<Scalar> tmp(static_cast<size_t>(m_n_table_segments) * 12, m_exec_conf);
    m_pswf_table_gpu.swap(tmp);
    }

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        const Index3D decomp_idx =
            m_pdata->getDomainDecomposition()->getDomainIndexer();

        int gdim[3], pdim[3], pidx[3], embed[3];

        pdim[0] = static_cast<int>(decomp_idx.getD());
        pdim[1] = static_cast<int>(decomp_idx.getH());
        pdim[2] = static_cast<int>(decomp_idx.getW());

        gdim[0] = static_cast<int>(m_mesh_points.z * decomp_idx.getD());
        gdim[1] = static_cast<int>(m_mesh_points.y * decomp_idx.getH());
        gdim[2] = static_cast<int>(m_mesh_points.x * decomp_idx.getW());

        embed[0] = static_cast<int>(m_grid_dim.z);
        embed[1] = static_cast<int>(m_grid_dim.y);
        embed[2] = static_cast<int>(m_grid_dim.x);

        const uint3 pcoord =
            m_pdata->getDomainDecomposition()->getGridPos();
        pidx[0] = static_cast<int>(pcoord.z);
        pidx[1] = static_cast<int>(pcoord.y);
        pidx[2] = static_cast<int>(pcoord.x);

        int row_m = 0;
        ArrayHandle<unsigned int> h_cart_ranks(
            m_pdata->getDomainDecomposition()->getCartRanks(),
            access_location::host, access_mode::read);

        dfft_create_plan(&m_dfft_plan_forward, 3, gdim, embed, nullptr,
                         pdim, pidx, row_m, 0, 1,
                         m_exec_conf->getMPICommunicator(),
                         reinterpret_cast<int*>(h_cart_ranks.data));

        dfft_create_plan(&m_dfft_plan_inverse, 3, gdim, nullptr, embed,
                         pdim, pidx, row_m, 0, 1,
                         m_exec_conf->getMPICommunicator(),
                         reinterpret_cast<int*>(h_cart_ranks.data));

        m_grid_comm_forward.reset(new CommunicatorGrid<kiss_fft_cpx>(
            m_sysdef,
            make_uint3(m_mesh_points.x, m_mesh_points.y, m_mesh_points.z),
            make_uint3(m_grid_dim.x,    m_grid_dim.y,    m_grid_dim.z),
            m_n_ghost_cells, true));

        m_grid_comm_reverse.reset(new CommunicatorGrid<kiss_fft_cpx>(
            m_sysdef,
            make_uint3(m_mesh_points.x, m_mesh_points.y, m_mesh_points.z),
            make_uint3(m_grid_dim.x,    m_grid_dim.y,    m_grid_dim.z),
            m_n_ghost_cells, false));

        m_dfft_initialized = true;
        }
    else
#endif
        {
        int dims[3] = {static_cast<int>(m_mesh_points.z),
                       static_cast<int>(m_mesh_points.y),
                       static_cast<int>(m_mesh_points.x)};

        m_kiss_fft  = kiss_fftnd_alloc(dims, 3, 0, nullptr, nullptr);
        m_kiss_ifft = kiss_fftnd_alloc(dims, 3, 1, nullptr, nullptr);

        if (!m_kiss_fft || !m_kiss_ifft)
            throw std::runtime_error(
                "ESPForceCompute: failed to allocate KISS FFT plans.");

        m_kiss_fft_initialized      = true;
        m_kiss_fft_initialized_priv = true;
        }
    }

// ============================================================================
// PSWF coefficient setup pipeline
// ============================================================================

void ESPForceCompute::setupCoeffs()
    {
    compute_pswf_rho_coeff();
    compute_pswf_gf_denom();
    buildPSWFTable();
    uploadPSWFTable();
    m_pswf_self_energy_const = computePSWFSelfEnergyConst();
    }

// ----------------------------------------------------------------------------
// compute_pswf_rho_coeff
//   Replaces the B-spline recurrence of PPPMForceCompute::compute_rho_coeff().
//   The PSWF kernel is sampled at m_order interpolation nodes per stencil
//   offset, then a degree-(m_order-1) polynomial is fit to those samples.
//   The resulting Horner coefficients are stored in m_rho_coeff with the
//   same (offset, degree) layout used by the GPU charge-assignment kernel.
// ----------------------------------------------------------------------------
void ESPForceCompute::compute_pswf_rho_coeff()
    {
    ArrayHandle<Scalar> h_rho_coeff(m_rho_coeff,
                                    access_location::host, access_mode::overwrite);

    const int    mult_fact = 2 * m_order + 1;
    const int    nlower    = -(m_order - 1) / 2;
    const int    nupper    =   m_order / 2;
    const Scalar radius    = std::max(Scalar(1.0), Scalar(m_radius));

    for (int offset = nlower; offset <= nupper; ++offset)
        {
        std::vector<Scalar> xs(static_cast<size_t>(m_order));
        std::vector<Scalar> ys(static_cast<size_t>(m_order));

        for (int n = 0; n < m_order; ++n)
            {
            Scalar xi;
            if (m_order == 1)
                xi = Scalar(0.0);
            else
                xi = Scalar(-0.5) + Scalar(n) / Scalar(m_order - 1);

            const Scalar arg = std::fabs((Scalar(offset) - xi) / radius);
            xs[static_cast<size_t>(n)] = xi;
            ys[static_cast<size_t>(n)] = evalPSWFKernel(arg);
            }

        std::vector<Scalar> coeffs = fit_power_polynomial(xs, ys);

        for (int deg = 0; deg < m_order; ++deg)
            h_rho_coeff.data[(offset - nlower)
                             + deg * mult_fact] = coeffs[static_cast<size_t>(deg)];
        }

    // Partition-of-unity normalisation: each polynomial degree must sum to 1
    // across all offsets so that Σ_l W_l(r) = 1 for any fractional position.
    for (int deg = 0; deg < m_order; ++deg)
        {
        Scalar sum = Scalar(0.0);
        for (int offset = nlower; offset <= nupper; ++offset)
            sum += h_rho_coeff.data[(offset - nlower) + deg * mult_fact];

        if (std::fabs(sum) > Scalar(1.0e-12))
            for (int offset = nlower; offset <= nupper; ++offset)
                h_rho_coeff.data[(offset - nlower) + deg * mult_fact] /= sum;
        }
    }

// ----------------------------------------------------------------------------
// compute_pswf_gf_denom
//   Builds coefficients for the aliasing denominator G_denom(k) ≡
//   |Σ_m ψ̂((k+2πm/h)r_c/σ_0)|^2.  We use a parametric approximation of
//   the PSWF Fourier transform via a modified exponential series whose
//   bandwidth parameter β captures the compactness of the kernel.
// ----------------------------------------------------------------------------
void ESPForceCompute::compute_pswf_gf_denom()
    {
    ArrayHandle<Scalar> h_gf_b(m_gf_b, access_location::host, access_mode::overwrite);

    const Scalar beta = m_pswf_c;

    for (int l = 0; l < m_order; ++l)
        {
        // (-1)^l * β^l / (2l)!
        Scalar coeff = Scalar((l % 2) ? -1.0 : 1.0);
        Scalar denom = Scalar(1.0);
        for (int n = 2; n <= 2 * l; ++n)
            denom *= Scalar(n);
        coeff *= std::pow(beta, Scalar(l)) / denom;
        h_gf_b.data[l] = coeff;
        }
    }

// Helper: evaluate the aliasing denominator at (x, y, z) = (kx*h/2)^2, etc.
Scalar ESPForceCompute::gf_denom_pswf(Scalar x, Scalar y, Scalar z) const
    {
    ArrayHandle<Scalar> h_gf_b(m_gf_b, access_location::host, access_mode::read);

    Scalar sx = Scalar(0.0), sy = Scalar(0.0), sz = Scalar(0.0);
    for (int l = m_order - 1; l >= 0; --l)
        {
        sx = h_gf_b.data[l] + sx * x;
        sy = h_gf_b.data[l] + sy * y;
        sz = h_gf_b.data[l] + sz * z;
        }
    const Scalar s = sx * sy * sz;
    return std::max(s * s, Scalar(1.0e-12));
    }

// ============================================================================
// Optimal influence function G̃(k)
// ============================================================================
//
// ESP replaces the Gaussian-based PPPM influence function
//   G_PPPM(k) = e^{-k^2/(4κ^2)} / k^2
// with the PSWF-based
//   G_ESP(k) = |ψ̂(|k|)|^2 / (k^2 · G_denom(k))
// where ψ̂ is approximated by the separable product of sinc^P × compact
// rational factor (Fourier transform of the separable PSWF stencil).
// This ensures the same G̃ is used for both the reciprocal-space summation
// and the aliasing correction denominator.
// ============================================================================

void ESPForceCompute::computeInfluenceFunction()
    {
    ArrayHandle<Scalar>  h_inf_f(m_inf_f, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar3> h_k    (m_k,     access_location::host, access_mode::overwrite);

    const BoxDim  box = m_pdata->getGlobalBox();
    const Scalar3 L   = box.getL();

    unsigned int kx_offset = 0, ky_offset = 0, kz_offset = 0;
#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        const uint3 pcoord = m_pdata->getDomainDecomposition()->getGridPos();
        kx_offset = pcoord.x * m_mesh_points.x;
        ky_offset = pcoord.y * m_mesh_points.y;
        kz_offset = pcoord.z * m_mesh_points.z;
        }
#endif

    const Scalar hx = safe_div(L.x, Scalar(m_global_dim.x));
    const Scalar hy = safe_div(L.y, Scalar(m_global_dim.y));
    const Scalar hz = safe_div(L.z, Scalar(m_global_dim.z));

    for (unsigned int kz = 0; kz < m_mesh_points.z; ++kz)
        {
        const unsigned int kz_g = kz + kz_offset;
        const int nz = (kz_g <= m_global_dim.z / 2)
                         ? int(kz_g) : int(kz_g) - int(m_global_dim.z);
        const Scalar kz_val = ESP_TWO_PI * Scalar(nz) / L.z;

        for (unsigned int ky = 0; ky < m_mesh_points.y; ++ky)
            {
            const unsigned int ky_g = ky + ky_offset;
            const int ny = (ky_g <= m_global_dim.y / 2)
                             ? int(ky_g) : int(ky_g) - int(m_global_dim.y);
            const Scalar ky_val = ESP_TWO_PI * Scalar(ny) / L.y;

            for (unsigned int kx = 0; kx < m_mesh_points.x; ++kx)
                {
                const unsigned int kx_g = kx + kx_offset;
                const int nx = (kx_g <= m_global_dim.x / 2)
                                 ? int(kx_g) : int(kx_g) - int(m_global_dim.x);
                const Scalar kx_val = ESP_TWO_PI * Scalar(nx) / L.x;

                const unsigned int idx = index_3d(kx, ky, kz, m_mesh_points);
                h_k.data[idx] = make_scalar3(kx_val, ky_val, kz_val);

                const Scalar k2 = kx_val*kx_val + ky_val*ky_val + kz_val*kz_val;
                if (k2 < Scalar(1.0e-16))
                    {
                    h_inf_f.data[idx] = Scalar(0.0);
                    continue;
                    }

                // Dimensionless half-wavevectors (argument of sinc in SI)
                const Scalar sx = Scalar(0.5) * kx_val * hx;
                const Scalar sy = Scalar(0.5) * ky_val * hy;
                const Scalar sz = Scalar(0.5) * kz_val * hz;

                // Separable approximation of ψ̂:
                //   sinc^P factor captures the polynomial spread stencil
                //   compact rational factor encodes PSWF spectral decay
                const Scalar wx = std::pow(std::fabs(sinc(sx)), m_order);
                const Scalar wy = std::pow(std::fabs(sinc(sy)), m_order);
                const Scalar wz = std::pow(std::fabs(sinc(sz)), m_order);
                const Scalar cx = Scalar(1.0) / (Scalar(1.0) + sqr(sx) / Scalar(m_order + 1));
                const Scalar cy = Scalar(1.0) / (Scalar(1.0) + sqr(sy) / Scalar(m_order + 1));
                const Scalar cz = Scalar(1.0) / (Scalar(1.0) + sqr(sz) / Scalar(m_order + 1));

                const Scalar pswf_hat = wx * wy * wz * cx * cy * cz;

                // Aliasing denominator (PPPM-like, built from PSWF coefficients)
                const Scalar denom      = std::max(
                    gf_denom_pswf(sqr(sx), sqr(sy), sqr(sz)), Scalar(1.0e-12));
                // Optionally screened k^2 for Debye–Hückel systems
                const Scalar screened_k2 = k2 + sqr(m_alpha);

                h_inf_f.data[idx] = (pswf_hat * pswf_hat) / (screened_k2 * denom);
                }
            }
        }

    // Store Nyquist squared for energy DC guard
    const Scalar qstar = Scalar(0.25)
        * (sqr(ESP_TWO_PI / hx) + sqr(ESP_TWO_PI / hy) + sqr(ESP_TWO_PI / hz));
    m_qstarsq = qstar;
    }

// ============================================================================
// Charge assignment (spread particles onto mesh)
// ============================================================================

void ESPForceCompute::assignParticles()
    {
    ArrayHandle<kiss_fft_cpx> h_mesh(m_mesh,
                                     access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4>      h_pos (m_pdata->getPositions(),
                                     access_location::host, access_mode::read);
    ArrayHandle<Scalar>       h_chg (m_pdata->getCharges(),
                                     access_location::host, access_mode::read);
    ArrayHandle<Scalar>       h_rho (m_rho_coeff,
                                     access_location::host, access_mode::read);

    for (unsigned int i = 0; i < m_n_cells; ++i)
        { h_mesh.data[i].r = 0; h_mesh.data[i].i = 0; }

    const BoxDim box      = m_pdata->getBox();
    const int    mf       = 2 * m_order + 1;
    const int    nlower   = -(m_order - 1) / 2;
    const int    nupper   =   m_order / 2;

    for (unsigned int gi = 0; gi < m_group->getNumMembers(); ++gi)
        {
        const unsigned int idx     = m_group->getMemberIndex(gi);
        const Scalar4      postype = h_pos.data[idx];
        const Scalar       qi      = h_chg.data[idx];

        const Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        if (std::isnan(pos.x) || std::isnan(pos.y) || std::isnan(pos.z))
            continue;

        Scalar3 frac = box.makeFraction(pos);
        Scalar3 rp   = make_scalar3(
            frac.x * Scalar(m_mesh_points.x) + Scalar(m_n_ghost_cells.x),
            frac.y * Scalar(m_mesh_points.y) + Scalar(m_n_ghost_cells.y),
            frac.z * Scalar(m_mesh_points.z) + Scalar(m_n_ghost_cells.z));

        const Scalar shift    = (m_order % 2) ? Scalar(0.5) : Scalar(0.0);
        const Scalar shiftone = (m_order % 2) ? Scalar(0.0) : Scalar(0.5);

        int ix = int(rp.x + shift);
        int iy = int(rp.y + shift);
        int iz = int(rp.z + shift);

        const Scalar dx = shiftone + Scalar(ix) - rp.x;
        const Scalar dy = shiftone + Scalar(iy) - rp.y;
        const Scalar dz = shiftone + Scalar(iz) - rp.z;

        if (ix == int(m_grid_dim.x) && !m_n_ghost_cells.x) ix = 0;
        if (iy == int(m_grid_dim.y) && !m_n_ghost_cells.y) iy = 0;
        if (iz == int(m_grid_dim.z) && !m_n_ghost_cells.z) iz = 0;

        for (int ii = nlower; ii <= nupper; ++ii)
            {
            Scalar Wx = Scalar(0.0);
            for (int p = m_order - 1; p >= 0; --p)
                Wx = h_rho.data[(ii - nlower) + p * mf] + Wx * dx;

            int ni = ix + ii;
            if (!m_n_ghost_cells.x) ni = wrap_index(ni, int(m_grid_dim.x));
            if (ni < 0 || ni >= int(m_grid_dim.x)) continue;

            for (int jj = nlower; jj <= nupper; ++jj)
                {
                Scalar Wy = Scalar(0.0);
                for (int p = m_order - 1; p >= 0; --p)
                    Wy = h_rho.data[(jj - nlower) + p * mf] + Wy * dy;

                int nj = iy + jj;
                if (!m_n_ghost_cells.y) nj = wrap_index(nj, int(m_grid_dim.y));
                if (nj < 0 || nj >= int(m_grid_dim.y)) continue;

                for (int kk = nlower; kk <= nupper; ++kk)
                    {
                    Scalar Wz = Scalar(0.0);
                    for (int p = m_order - 1; p >= 0; --p)
                        Wz = h_rho.data[(kk - nlower) + p * mf] + Wz * dz;

                    int nk = iz + kk;
                    if (!m_n_ghost_cells.z) nk = wrap_index(nk, int(m_grid_dim.z));
                    if (nk < 0 || nk >= int(m_grid_dim.z)) continue;

                    h_mesh.data[index_3d(static_cast<unsigned>(ni),
                                         static_cast<unsigned>(nj),
                                         static_cast<unsigned>(nk),
                                         m_grid_dim)].r
                        += static_cast<kiss_fft_scalar>(qi * Wx * Wy * Wz);
                    }
                }
            }
        }
    }

// ============================================================================
// Forward FFT → scale → inverse FFT
// ============================================================================

void ESPForceCompute::updateMeshes()
    {
    // ------------------------------------------------------------------
    // Step 1: ghost-cell exchange + forward FFT  ρ → ρ̃
    // ------------------------------------------------------------------
#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        m_grid_comm_forward->communicate(m_mesh);
        ArrayHandle<kiss_fft_cpx> h_mesh(m_mesh,
                                          access_location::host, access_mode::read);
        ArrayHandle<kiss_fft_cpx> h_fmesh(m_fourier_mesh,
                                           access_location::host, access_mode::overwrite);
        dfft_execute(
            reinterpret_cast<cpx_t*>(h_mesh.data  + m_ghost_offset),
            reinterpret_cast<cpx_t*>(h_fmesh.data), 0, m_dfft_plan_forward);
        }
    else
#endif
        {
        ArrayHandle<kiss_fft_cpx> h_mesh(m_mesh,
                                          access_location::host, access_mode::read);
        ArrayHandle<kiss_fft_cpx> h_fmesh(m_fourier_mesh,
                                           access_location::host, access_mode::overwrite);
        std::vector<kiss_fft_cpx> in(m_n_inner_cells), out(m_n_inner_cells);
        for (size_t i = 0; i < m_n_inner_cells; ++i)
            in[i] = h_mesh.data[i + m_ghost_offset];
        kiss_fftnd(m_kiss_fft, in.data(), out.data());
        for (size_t i = 0; i < m_n_inner_cells; ++i)
            h_fmesh.data[i] = out[i];
        }

    // ------------------------------------------------------------------
    // Step 2: multiply by influence function and build force meshes in k
    //         F̃_α(k) = i k_α · G̃(k) · ρ̃(k)
    // ------------------------------------------------------------------
    {
    ArrayHandle<Scalar>       h_inf(m_inf_f,          access_location::host, access_mode::read);
    ArrayHandle<Scalar3>      h_k  (m_k,              access_location::host, access_mode::read);
    ArrayHandle<kiss_fft_cpx> h_fm (m_fourier_mesh,   access_location::host, access_mode::read);
    ArrayHandle<kiss_fft_cpx> h_Gx (m_fourier_mesh_G_x, access_location::host, access_mode::overwrite);
    ArrayHandle<kiss_fft_cpx> h_Gy (m_fourier_mesh_G_y, access_location::host, access_mode::overwrite);
    ArrayHandle<kiss_fft_cpx> h_Gz (m_fourier_mesh_G_z, access_location::host, access_mode::overwrite);

    const Scalar inv_NNN =
        Scalar(1.0) / Scalar(m_global_dim.x * m_global_dim.y * m_global_dim.z);

    for (unsigned int idx = 0; idx < m_n_inner_cells; ++idx)
        {
        const Scalar re    = Scalar(h_fm.data[idx].r);
        const Scalar im    = Scalar(h_fm.data[idx].i);
        const Scalar scale = h_inf.data[idx] * inv_NNN;
        const Scalar3 kv   = h_k.data[idx];

        // Multiply by ik: (a + ib) * ik = -b*k + i*a*k
        h_Gx.data[idx].r = static_cast<kiss_fft_scalar>( im * kv.x * scale);
        h_Gx.data[idx].i = static_cast<kiss_fft_scalar>(-re * kv.x * scale);

        h_Gy.data[idx].r = static_cast<kiss_fft_scalar>( im * kv.y * scale);
        h_Gy.data[idx].i = static_cast<kiss_fft_scalar>(-re * kv.y * scale);

        h_Gz.data[idx].r = static_cast<kiss_fft_scalar>( im * kv.z * scale);
        h_Gz.data[idx].i = static_cast<kiss_fft_scalar>(-re * kv.z * scale);
        }
    }

    // ------------------------------------------------------------------
    // Step 3: inverse FFT of three force components + ghost reverse comms
    // ------------------------------------------------------------------
#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        ArrayHandle<kiss_fft_cpx> hGx(m_fourier_mesh_G_x,
                                       access_location::host, access_mode::read);
        ArrayHandle<kiss_fft_cpx> hGy(m_fourier_mesh_G_y,
                                       access_location::host, access_mode::read);
        ArrayHandle<kiss_fft_cpx> hGz(m_fourier_mesh_G_z,
                                       access_location::host, access_mode::read);
        ArrayHandle<kiss_fft_cpx> hRx(m_inv_fourier_mesh_x,
                                       access_location::host, access_mode::overwrite);
        ArrayHandle<kiss_fft_cpx> hRy(m_inv_fourier_mesh_y,
                                       access_location::host, access_mode::overwrite);
        ArrayHandle<kiss_fft_cpx> hRz(m_inv_fourier_mesh_z,
                                       access_location::host, access_mode::overwrite);

        dfft_execute(reinterpret_cast<cpx_t*>(hGx.data),
                     reinterpret_cast<cpx_t*>(hRx.data + m_ghost_offset),
                     1, m_dfft_plan_inverse);
        dfft_execute(reinterpret_cast<cpx_t*>(hGy.data),
                     reinterpret_cast<cpx_t*>(hRy.data + m_ghost_offset),
                     1, m_dfft_plan_inverse);
        dfft_execute(reinterpret_cast<cpx_t*>(hGz.data),
                     reinterpret_cast<cpx_t*>(hRz.data + m_ghost_offset),
                     1, m_dfft_plan_inverse);

        m_grid_comm_reverse->communicate(m_inv_fourier_mesh_x);
        m_grid_comm_reverse->communicate(m_inv_fourier_mesh_y);
        m_grid_comm_reverse->communicate(m_inv_fourier_mesh_z);
        }
    else
#endif
        {
        const size_t n = m_n_inner_cells;
        std::vector<kiss_fft_cpx> in(n), out(n);

        auto do_ifft = [&](GPUArray<kiss_fft_cpx>& G,
                           GPUArray<kiss_fft_cpx>& R)
            {
            ArrayHandle<kiss_fft_cpx> hG(G, access_location::host, access_mode::read);
            ArrayHandle<kiss_fft_cpx> hR(R, access_location::host, access_mode::overwrite);
            for (size_t i = 0; i < n; ++i) in[i]  = hG.data[i];
            kiss_fftnd(m_kiss_ifft, in.data(), out.data());
            for (size_t i = 0; i < n; ++i) hR.data[i + m_ghost_offset] = out[i];
            };

        do_ifft(m_fourier_mesh_G_x, m_inv_fourier_mesh_x);
        do_ifft(m_fourier_mesh_G_y, m_inv_fourier_mesh_y);
        do_ifft(m_fourier_mesh_G_z, m_inv_fourier_mesh_z);
        }
    }

// ============================================================================
// Force interpolation (gather from mesh onto particles)
// ============================================================================

void ESPForceCompute::interpolateForces()
    {
    ArrayHandle<Scalar4>      h_pos(m_pdata->getPositions(),
                                    access_location::host, access_mode::read);
    ArrayHandle<Scalar>       h_chg(m_pdata->getCharges(),
                                    access_location::host, access_mode::read);
    ArrayHandle<kiss_fft_cpx> hRx  (m_inv_fourier_mesh_x,
                                    access_location::host, access_mode::read);
    ArrayHandle<kiss_fft_cpx> hRy  (m_inv_fourier_mesh_y,
                                    access_location::host, access_mode::read);
    ArrayHandle<kiss_fft_cpx> hRz  (m_inv_fourier_mesh_z,
                                    access_location::host, access_mode::read);
    ArrayHandle<Scalar>       h_rho(m_rho_coeff,
                                    access_location::host, access_mode::read);
    ArrayHandle<Scalar4>      h_f  (m_force,
                                    access_location::host, access_mode::readwrite);

    const BoxDim box    = m_pdata->getBox();
    const int    mf     = 2 * m_order + 1;
    const int    nlower = -(m_order - 1) / 2;
    const int    nupper =   m_order / 2;

    for (unsigned int gi = 0; gi < m_group->getNumMembers(); ++gi)
        {
        const unsigned int idx     = m_group->getMemberIndex(gi);
        const Scalar4      postype = h_pos.data[idx];
        const Scalar       qi      = h_chg.data[idx];

        const Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        if (std::isnan(pos.x) || std::isnan(pos.y) || std::isnan(pos.z))
            continue;

        Scalar3 frac = box.makeFraction(pos);
        Scalar3 rp   = make_scalar3(
            frac.x * Scalar(m_mesh_points.x) + Scalar(m_n_ghost_cells.x),
            frac.y * Scalar(m_mesh_points.y) + Scalar(m_n_ghost_cells.y),
            frac.z * Scalar(m_mesh_points.z) + Scalar(m_n_ghost_cells.z));

        const Scalar shift    = (m_order % 2) ? Scalar(0.5) : Scalar(0.0);
        const Scalar shiftone = (m_order % 2) ? Scalar(0.0) : Scalar(0.5);

        int ix = int(rp.x + shift);
        int iy = int(rp.y + shift);
        int iz = int(rp.z + shift);

        const Scalar dx = shiftone + Scalar(ix) - rp.x;
        const Scalar dy = shiftone + Scalar(iy) - rp.y;
        const Scalar dz = shiftone + Scalar(iz) - rp.z;

        if (ix == int(m_grid_dim.x) && !m_n_ghost_cells.x) ix = 0;
        if (iy == int(m_grid_dim.y) && !m_n_ghost_cells.y) iy = 0;
        if (iz == int(m_grid_dim.z) && !m_n_ghost_cells.z) iz = 0;

        Scalar3 force = make_scalar3(Scalar(0), Scalar(0), Scalar(0));

        for (int ii = nlower; ii <= nupper; ++ii)
            {
            Scalar Wx = Scalar(0.0);
            for (int p = m_order - 1; p >= 0; --p)
                Wx = h_rho.data[(ii - nlower) + p * mf] + Wx * dx;

            int ni = ix + ii;
            if (!m_n_ghost_cells.x) ni = wrap_index(ni, int(m_grid_dim.x));
            if (ni < 0 || ni >= int(m_grid_dim.x)) continue;

            for (int jj = nlower; jj <= nupper; ++jj)
                {
                Scalar Wy = Scalar(0.0);
                for (int p = m_order - 1; p >= 0; --p)
                    Wy = h_rho.data[(jj - nlower) + p * mf] + Wy * dy;

                int nj = iy + jj;
                if (!m_n_ghost_cells.y) nj = wrap_index(nj, int(m_grid_dim.y));
                if (nj < 0 || nj >= int(m_grid_dim.y)) continue;

                for (int kk = nlower; kk <= nupper; ++kk)
                    {
                    Scalar Wz = Scalar(0.0);
                    for (int p = m_order - 1; p >= 0; --p)
                        Wz = h_rho.data[(kk - nlower) + p * mf] + Wz * dz;

                    int nk = iz + kk;
                    if (!m_n_ghost_cells.z) nk = wrap_index(nk, int(m_grid_dim.z));
                    if (nk < 0 || nk >= int(m_grid_dim.z)) continue;

                    const unsigned int mid = index_3d(
                        static_cast<unsigned>(ni),
                        static_cast<unsigned>(nj),
                        static_cast<unsigned>(nk), m_grid_dim);

                    const Scalar w = Wx * Wy * Wz;
                    force.x += qi * w * Scalar(hRx.data[mid].r);
                    force.y += qi * w * Scalar(hRy.data[mid].r);
                    force.z += qi * w * Scalar(hRz.data[mid].r);
                    }
                }
            }

        h_f.data[idx].x += force.x;
        h_f.data[idx].y += force.y;
        h_f.data[idx].z += force.z;
        }
    }

// ============================================================================
// Reciprocal-space energy
//   E_k = (1/2) Σ_k G̃(k)|ρ̃(k)|^2 / N^3  -  E_self(PSWF)
//   E_self = m_pswf_self_energy_const · Σ_i q_i^2
// ============================================================================

Scalar ESPForceCompute::computePE()
    {
    ArrayHandle<kiss_fft_cpx> h_fm (m_fourier_mesh,
                                    access_location::host, access_mode::read);
    ArrayHandle<Scalar>       h_inf(m_inf_f,
                                    access_location::host, access_mode::read);

    const Scalar inv_NNN =
        Scalar(1.0) / Scalar(m_global_dim.x * m_global_dim.y * m_global_dim.z);

    Scalar pe = Scalar(0.0);
    for (unsigned int idx = 0; idx < m_n_inner_cells; ++idx)
        {
        const Scalar re   = Scalar(h_fm.data[idx].r);
        const Scalar im   = Scalar(h_fm.data[idx].i);
        pe += Scalar(0.5) * h_inf.data[idx] * (re*re + im*im) * inv_NNN;
        }

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        MPI_Allreduce(MPI_IN_PLACE, &pe, 1, MPI_HOOMD_SCALAR,
                      MPI_SUM, m_exec_conf->getMPICommunicator());
#endif

    // Subtract PSWF self-energy correction:
    //   E_self = I_pswf · Σ q_i^2,  I_pswf = ∫_0^1 ψ^2(u) du / r_c
    pe -= m_pswf_self_energy_const * m_q2;

    // Add rigid-body correction (zero in the default implementation)
    pe += m_body_energy;

    return pe;
    }

// ============================================================================
// Virial from reciprocal space
//   W_αβ = Σ_k (2 G̃(k)/k^2) · k_α k_β · |ρ̃(k)|^2 / N^3
// ============================================================================

void ESPForceCompute::computeVirialMesh()
    {
    ArrayHandle<kiss_fft_cpx> h_fm(m_fourier_mesh,
                                   access_location::host, access_mode::read);
    ArrayHandle<Scalar>       h_inf(m_inf_f,
                                    access_location::host, access_mode::read);
    ArrayHandle<Scalar3>      h_k  (m_k,
                                    access_location::host, access_mode::read);
    ArrayHandle<Scalar>       h_vm (m_virial_mesh,
                                    access_location::host, access_mode::overwrite);

    const Scalar inv_NNN =
        Scalar(1.0) / Scalar(m_global_dim.x * m_global_dim.y * m_global_dim.z);

    for (unsigned int idx = 0; idx < m_n_inner_cells; ++idx)
        {
        const Scalar re   = Scalar(h_fm.data[idx].r);
        const Scalar im   = Scalar(h_fm.data[idx].i);
        const Scalar mod2 = (re*re + im*im) * h_inf.data[idx] * inv_NNN;
        const Scalar3 kv  = h_k.data[idx];
        const Scalar  k2  = kv.x*kv.x + kv.y*kv.y + kv.z*kv.z;
        const Scalar  sc  = (k2 > Scalar(1.0e-14)) ? (mod2 / k2) : Scalar(0.0);

        h_vm.data[0*m_n_inner_cells + idx] = sc * kv.x * kv.x;
        h_vm.data[1*m_n_inner_cells + idx] = sc * kv.x * kv.y;
        h_vm.data[2*m_n_inner_cells + idx] = sc * kv.x * kv.z;
        h_vm.data[3*m_n_inner_cells + idx] = sc * kv.y * kv.y;
        h_vm.data[4*m_n_inner_cells + idx] = sc * kv.y * kv.z;
        h_vm.data[5*m_n_inner_cells + idx] = sc * kv.z * kv.z;
        }
    }

void ESPForceCompute::computeVirial()
    {
    computeVirialMesh();

    ArrayHandle<Scalar> h_vm(m_virial_mesh,
                              access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_v (m_virial,
                              access_location::host, access_mode::overwrite);

    Scalar totals[6] = {};
    for (unsigned int comp = 0; comp < 6; ++comp)
        for (unsigned int idx = 0; idx < m_n_inner_cells; ++idx)
            totals[comp] += h_vm.data[comp * m_n_inner_cells + idx];

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        MPI_Allreduce(MPI_IN_PLACE, totals, 6, MPI_HOOMD_SCALAR,
                      MPI_SUM, m_exec_conf->getMPICommunicator());
#endif

    const size_t        pitch      = m_virial.getPitch();
    const unsigned int  group_size = m_group->getNumMembers();

    for (unsigned int comp = 0; comp < 6; ++comp)
        for (unsigned int idx = 0; idx < m_pdata->getN(); ++idx)
            h_v.data[comp * pitch + idx] = Scalar(0.0);

    if (group_size > 0)
        for (unsigned int g = 0; g < group_size; ++g)
            {
            const unsigned int idx = m_group->getMemberIndex(g);
            for (unsigned int comp = 0; comp < 6; ++comp)
                h_v.data[comp * pitch + idx] = totals[comp] / Scalar(group_size);
            }
    }

// ============================================================================
// Exclusion correction
//   For each excluded pair (i, j) the mesh over-counts the interaction.
//   We subtract the mesh contribution (full 1/r) and add back the correct
//   direct-space remainder:
//     ΔF_{ij} = q_i q_j [ (1/r^2 - (-dL/dr)) r̂ ]
//     ΔE_{ij} = q_i q_j (1/r - L(r))
//   where L(r) is evaluated from the precomputed lookup table.
// ============================================================================

void ESPForceCompute::fixExclusions()
    {
    if (!m_nlist->getExclusionsSet())
        return;

    ArrayHandle<Scalar4>       h_pos  (m_pdata->getPositions(),
                                       access_location::host, access_mode::read);
    ArrayHandle<Scalar>        h_chg  (m_pdata->getCharges(),
                                       access_location::host, access_mode::read);
    ArrayHandle<Scalar4>       h_f    (m_force,
                                       access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar>        h_v    (m_virial,
                                       access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int>  h_nex  (m_nlist->getNExArray(),
                                       access_location::host, access_mode::read);
    ArrayHandle<unsigned int>  h_exls (m_nlist->getExListArray(),
                                       access_location::host, access_mode::read);

    const Index2D   ex_idx     = m_nlist->getExListIndexer();
    const size_t    vpit       = m_virial.getPitch();

    // Lambda: evaluate L(r) and -dL/dr from the piecewise-polynomial table.
    // Falls back to direct quadrature outside [0, r_c].
    auto eval_table = [this](Scalar r, Scalar& Lr, Scalar& mdLdr)
        {
        if (r >= m_rcut)
            {
            Lr    = evalL_direct(r);
            mdLdr = evalDL_direct(r);
            return;
            }
        const Scalar s   = std::max(Scalar(0.0),
                           std::min(Scalar(1.0) - Scalar(1.0e-9),
                                    r / m_rcut));
        const unsigned int seg = std::min(
            static_cast<unsigned int>(s * Scalar(m_n_table_segments)),
            m_n_table_segments - 1u);

        const ESPTableEntry& e = m_pswf_table_cpu[seg];
        const Scalar t = (r - e.r_lo) * e.dr_inv;

        const std::vector<Scalar> pc = {
            e.potential_coeffs.x, e.potential_coeffs.y,
            e.potential_coeffs.z, e.potential_coeffs.w, e.potential_coeff4 };
        const std::vector<Scalar> fc = {
            e.force_coeffs.x, e.force_coeffs.y,
            e.force_coeffs.z, e.force_coeffs.w, e.force_coeff4 };

        Lr    = horner_eval(pc, t);
        mdLdr = horner_eval(fc, t);
        };

    for (unsigned int i = 0; i < m_pdata->getN(); ++i)
        {
        const Scalar4 pi4 = h_pos.data[i];
        const Scalar3 pi  = make_scalar3(pi4.x, pi4.y, pi4.z);
        const Scalar  qi  = h_chg.data[i];

        for (unsigned int ex = 0; ex < h_nex.data[i]; ++ex)
            {
            const unsigned int j = h_exls.data[ex_idx(ex, i)];
            if (j <= i || j >= m_pdata->getN()) continue;

            const Scalar4 pj4 = h_pos.data[j];
            const Scalar3 pj  = make_scalar3(pj4.x, pj4.y, pj4.z);
            const Scalar  qj  = h_chg.data[j];

            Scalar3 rij = pj - pi;
            rij = m_pdata->getBox().minImage(rij);
            const Scalar rsq = dot(rij, rij);
            if (rsq < Scalar(1.0e-20)) continue;

            const Scalar r    = std::sqrt(rsq);
            const Scalar invr = Scalar(1.0) / r;
            const Scalar qiqj = qi * qj;

            Scalar Lr = Scalar(0.0), mdLdr = Scalar(0.0);
            eval_table(r, Lr, mdLdr);

            // Net radial force: d/dr(1/r) - (-dL/dr) = -1/r^2 + mdLdr
            // Convention: positive → repulsive (i pushed away from j)
            const Scalar F_radial = qiqj * (-invr*invr + mdLdr) * invr;
            const Scalar3 fij     = make_scalar3(F_radial * rij.x,
                                                  F_radial * rij.y,
                                                  F_radial * rij.z);

            h_f.data[i].x += fij.x;
            h_f.data[i].y += fij.y;
            h_f.data[i].z += fij.z;
            h_f.data[j].x -= fij.x;
            h_f.data[j].y -= fij.y;
            h_f.data[j].z -= fij.z;

            // Energy correction: 1/r - L(r)
            const Scalar dU = qiqj * (invr - Lr);
            h_f.data[i].w += Scalar(0.5) * dU;
            h_f.data[j].w += Scalar(0.5) * dU;

            // Virial (factor of 1/2 from symmetric assignment)
            const Scalar vxx = Scalar(0.5) * rij.x * fij.x;
            const Scalar vxy = Scalar(0.5) * rij.x * fij.y;
            const Scalar vxz = Scalar(0.5) * rij.x * fij.z;
            const Scalar vyy = Scalar(0.5) * rij.y * fij.y;
            const Scalar vyz = Scalar(0.5) * rij.y * fij.z;
            const Scalar vzz = Scalar(0.5) * rij.z * fij.z;

            h_v.data[0*vpit + i] += vxx; h_v.data[0*vpit + j] += vxx;
            h_v.data[1*vpit + i] += vxy; h_v.data[1*vpit + j] += vxy;
            h_v.data[2*vpit + i] += vxz; h_v.data[2*vpit + j] += vxz;
            h_v.data[3*vpit + i] += vyy; h_v.data[3*vpit + j] += vyy;
            h_v.data[4*vpit + i] += vyz; h_v.data[4*vpit + j] += vyz;
            h_v.data[5*vpit + i] += vzz; h_v.data[5*vpit + j] += vzz;
            }
        }
    }

// ============================================================================
// Rigid-body self-energy correction (stub — extend for molecular systems)
// ============================================================================

void ESPForceCompute::computeBodyCorrection()
    {
    m_body_energy = Scalar(0.0);
    }

// ============================================================================
// PSWF lookup table construction
// ============================================================================

// ----------------------------------------------------------------------------
// evalPSWFKernel(x)
//   Returns χ_α(x) ≡ (1-x^2)^P · exp(β(1-x^2))  at x ∈ [0, 1].
// ----------------------------------------------------------------------------
Scalar ESPForceCompute::evalPSWFKernel(Scalar x) const
    {
    if (x >= Scalar(1.0)) return Scalar(0.0);
    if (x <  Scalar(0.0)) x = Scalar(0.0);

    const Scalar s    = Scalar(1.0) - x * x;
    const Scalar beta = m_pswf_c;
    const Scalar bs   = beta * s;
    return std::pow(s, Scalar(m_order)) * std::exp(bs);
    }

// ----------------------------------------------------------------------------
// evalL_direct(r)
//   L(r) = (1 - CDF_ψ(r/r_c)) / (4π r)
//   where CDF_ψ(x) = ∫_0^x ψ(u) du / ∫_0^1 ψ(u) du.
//   Both integrals are evaluated by composite Gauss–Legendre quadrature.
// ----------------------------------------------------------------------------
Scalar ESPForceCompute::evalL_direct(Scalar r) const
    {
    const Scalar rr = std::max(r, Scalar(1.0e-8) * m_rcut);
    const Scalar x  = std::min(rr / m_rcut, Scalar(1.0));

    auto kernel = [this](Scalar u) -> Scalar { return evalPSWFKernel(u); };

    const Scalar norm = integrate_gl(kernel, Scalar(0.0), Scalar(1.0), 8);
    if (x >= Scalar(1.0))
        return Scalar(0.0);

    const Scalar cdf = integrate_gl(kernel, Scalar(0.0), x, 8)
                     / std::max(norm, Scalar(1.0e-12));

    return (Scalar(1.0) - cdf) / (Scalar(4.0) * ESP_PI * rr);
    }

// ----------------------------------------------------------------------------
// evalDL_direct(r)
//   Returns -dL/dr via centred finite difference with step size h = 1e-5 r_c.
//   The table force_coeffs stores this value (positive = repulsive direction).
// ----------------------------------------------------------------------------
Scalar ESPForceCompute::evalDL_direct(Scalar r) const
    {
    const Scalar h  = std::max(Scalar(1.0e-5) * m_rcut, Scalar(1.0e-7));
    const Scalar rp = std::min(r + h, m_rcut);
    const Scalar rm = std::max(r - h, Scalar(1.0e-8) * m_rcut);
    return -(evalL_direct(rp) - evalL_direct(rm))
           / std::max(rp - rm, Scalar(1.0e-20));
    }

// ----------------------------------------------------------------------------
// buildPSWFTable
//   Fills m_pswf_table_cpu and uploads to m_pswf_table_gpu.
//   Each segment [r0, r1] is sampled at 5 Chebyshev-spaced points,
//   then a degree-4 polynomial is fit in the normalised variable t ∈ [0,1].
// ----------------------------------------------------------------------------
void ESPForceCompute::buildPSWFTable()
    {
    m_pswf_table_cpu.clear();
    m_pswf_table_cpu.resize(m_n_table_segments);

    const Scalar dr = m_rcut / Scalar(m_n_table_segments);

    for (unsigned int seg = 0; seg < m_n_table_segments; ++seg)
        {
        const Scalar r0 = Scalar(seg)     * dr;
        const Scalar r1 = Scalar(seg + 1) * dr;

        // 5 Chebyshev nodes on [0, 1] for degree-4 polynomial
        const std::vector<Scalar> t_nodes = {
            Scalar(0.0), Scalar(0.25), Scalar(0.5), Scalar(0.75), Scalar(1.0) };
        std::vector<Scalar> vv(5), ff(5);

        for (unsigned int p = 0; p < 5; ++p)
            {
            const Scalar r = std::max(r0 + (r1 - r0) * t_nodes[p],
                                      Scalar(1.0e-8) * m_rcut);
            vv[p] = evalL_direct(r);
            ff[p] = evalDL_direct(r);
            }

        std::vector<Scalar> vc = fit_power_polynomial(t_nodes, vv);
        std::vector<Scalar> fc = fit_power_polynomial(t_nodes, ff);

        ESPTableEntry e;
        e.potential_coeffs = make_scalar4(vc[0], vc[1], vc[2], vc[3]);
        e.potential_coeff4 = vc[4];
        e.force_coeffs     = make_scalar4(fc[0], fc[1], fc[2], fc[3]);
        e.force_coeff4     = fc[4];
        e.r_lo             = r0;
        e.dr_inv           = Scalar(1.0) / std::max(r1 - r0, Scalar(1.0e-20));
        m_pswf_table_cpu[seg] = e;
        }

    uploadPSWFTable();
    }

void ESPForceCompute::uploadPSWFTable()
    {
    ArrayHandle<Scalar> h_gpu(m_pswf_table_gpu,
                              access_location::host, access_mode::overwrite);
    for (unsigned int seg = 0; seg < m_n_table_segments; ++seg)
        {
        const ESPTableEntry& e = m_pswf_table_cpu[seg];
        const unsigned int   b = 12 * seg;
        h_gpu.data[b + 0]  = e.potential_coeffs.x;
        h_gpu.data[b + 1]  = e.potential_coeffs.y;
        h_gpu.data[b + 2]  = e.potential_coeffs.z;
        h_gpu.data[b + 3]  = e.potential_coeffs.w;
        h_gpu.data[b + 4]  = e.potential_coeff4;
        h_gpu.data[b + 5]  = e.force_coeffs.x;
        h_gpu.data[b + 6]  = e.force_coeffs.y;
        h_gpu.data[b + 7]  = e.force_coeffs.z;
        h_gpu.data[b + 8]  = e.force_coeffs.w;
        h_gpu.data[b + 9]  = e.force_coeff4;
        h_gpu.data[b + 10] = e.r_lo;
        h_gpu.data[b + 11] = e.dr_inv;
        }
    }

// ----------------------------------------------------------------------------
// computePSWFSelfEnergyConst
//   Returns I = ∫_0^1 ψ^2(u) du / r_c.
//   This replaces κ/√π in the PPPM self-energy formula.
// ----------------------------------------------------------------------------
Scalar ESPForceCompute::computePSWFSelfEnergyConst() const
    {
    auto integrand = [this](Scalar u) -> Scalar
        {
        const Scalar psi = evalPSWFKernel(u);
        return psi * psi;
        };
    const Scalar integral = integrate_gl(integrand, Scalar(0.0), Scalar(1.0), 8);
    return integral / std::max(m_rcut, Scalar(1.0e-20));
    }

// ============================================================================
// Ghost cell and error estimate helpers
// ============================================================================

uint3 ESPForceCompute::computeGhostCellNum() const
    {
#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        return make_uint3(m_radius, m_radius, m_radius);
#endif
    return make_uint3(0, 0, 0);
    }

Scalar ESPForceCompute::rms(Scalar h, Scalar prd, Scalar natoms) const
    {
    const Scalar x = h / std::max(prd, Scalar(1.0e-12));
    return std::sqrt(std::max(natoms, Scalar(1.0)))
         * std::pow(x, Scalar(m_order + 1));
    }

// ============================================================================
// Python binding export
// ============================================================================

void export_ESPForceCompute(pybind11::module& m)
    {
    pybind11::class_<ESPForceCompute,
                     ForceCompute,
                     std::shared_ptr<ESPForceCompute>>(m, "ESPForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<NeighborList>,
                            std::shared_ptr<ParticleGroup>>())
        .def("setParams",     &ESPForceCompute::setParams,
             pybind11::arg("nx"),
             pybind11::arg("ny"),
             pybind11::arg("nz"),
             pybind11::arg("order"),
             pybind11::arg("kappa"),
             pybind11::arg("rcut"),
             pybind11::arg("alpha")   = Scalar(0.0),
             pybind11::arg("n_table") = ESP_TABLE_SEGMENTS)
        .def("invalidate",    &ESPForceCompute::invalidate)
        .def("getResolution", &ESPForceCompute::getResolution)
        .def("getOrder",      &ESPForceCompute::getOrder)
        .def("getKappa",      &ESPForceCompute::getKappa)
        .def("getRCut",       &ESPForceCompute::getRCut)
        .def("getAlpha",      &ESPForceCompute::getAlpha)
        .def("getQSum",       &ESPForceCompute::getQSum)
        .def("getQ2Sum",      &ESPForceCompute::getQ2Sum)
        .def("getTableSize",  &ESPForceCompute::getTableSize)
        .def("getTablePtr",   &ESPForceCompute::getTablePtr);
    }

} // namespace md
} // namespace hoomd
