// Copyright (c) 2024-2026 ESP Plugin Contributors.
// Released under the BSD 3-Clause License.
//
// ESPForceCompute.cc
//
// CPU implementation of a custom HOOMD-blue ForceCompute plugin for
// Ewald Summation with Prolates (ESP).
//
// The implementation mirrors the structure of HOOMD-blue's PPPMForceCompute
// while replacing the Gaussian / B-spline specific pieces by PSWF-inspired
// coefficient builders and a real-space piecewise-polynomial lookup table.

#include "ESPForceCompute.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
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
namespace
{

constexpr Scalar ESP_PI = Scalar(3.1415926535897932384626433832795);
constexpr Scalar ESP_TWO_PI = Scalar(2.0) * ESP_PI;

inline bool is_pow2(unsigned int n)
    {
    while (n && n % 2 == 0)
        n /= 2;
    return (n == 1);
    }

inline Scalar sqr(Scalar x)
    {
    return x * x;
    }

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

std::vector<Scalar> solve_linear_system(std::vector<Scalar> A,
                                        std::vector<Scalar> b,
                                        unsigned int n)
    {
    for (unsigned int col = 0; col < n; ++col)
        {
        unsigned int pivot = col;
        Scalar pivot_abs = std::fabs(A[col * n + col]);

        for (unsigned int row = col + 1; row < n; ++row)
            {
            Scalar cand = std::fabs(A[row * n + col]);
            if (cand > pivot_abs)
                {
                pivot = row;
                pivot_abs = cand;
                }
            }

        if (pivot_abs < Scalar(1.0e-14))
            {
            throw std::runtime_error("ESPForceCompute: Singular linear system in polynomial fit.");
            }

        if (pivot != col)
            {
            for (unsigned int j = 0; j < n; ++j)
                std::swap(A[col * n + j], A[pivot * n + j]);
            std::swap(b[col], b[pivot]);
            }

        Scalar diag = A[col * n + col];
        for (unsigned int j = col; j < n; ++j)
            A[col * n + j] /= diag;
        b[col] /= diag;

        for (unsigned int row = 0; row < n; ++row)
            {
            if (row == col)
                continue;

            Scalar factor = A[row * n + col];
            if (std::fabs(factor) < Scalar(1.0e-20))
                continue;

            for (unsigned int j = col; j < n; ++j)
                A[row * n + j] -= factor * A[col * n + j];
            b[row] -= factor * b[col];
            }
        }

    return b;
    }

std::vector<Scalar> fit_power_polynomial(const std::vector<Scalar>& x,
                                         const std::vector<Scalar>& y)
    {
    const unsigned int n = static_cast<unsigned int>(x.size());
    if (n != y.size() || n == 0)
        {
        throw std::runtime_error("ESPForceCompute: Invalid input to fit_power_polynomial().");
        }

    std::vector<Scalar> A(n * n, Scalar(0.0));
    for (unsigned int i = 0; i < n; ++i)
        {
        Scalar xp = Scalar(1.0);
        for (unsigned int j = 0; j < n; ++j)
            {
            A[i * n + j] = xp;
            xp *= x[i];
            }
        }

    return solve_linear_system(A, y, n);
    }

template<class T> T horner_eval(const std::vector<T>& coeffs, T x)
    {
    T val = T(0);
    for (int i = static_cast<int>(coeffs.size()) - 1; i >= 0; --i)
        val = coeffs[static_cast<size_t>(i)] + val * x;
    return val;
    }

Scalar integrate_trapezoid(const std::function<Scalar(Scalar)>& f,
                           Scalar a,
                           Scalar b,
                           unsigned int n)
    {
    if (n < 2)
        n = 2;

    Scalar h = (b - a) / Scalar(n - 1);
    Scalar sum = Scalar(0.5) * (f(a) + f(b));
    for (unsigned int i = 1; i < n - 1; ++i)
        {
        Scalar x = a + Scalar(i) * h;
        sum += f(x);
        }
    return sum * h;
    }

} // end anonymous namespace

ESPForceCompute::ESPForceCompute(std::shared_ptr<SystemDefinition> sysdef,
                                 std::shared_ptr<NeighborList> nlist,
                                 std::shared_ptr<ParticleGroup> group)
    : ForceCompute(sysdef), m_nlist(nlist), m_group(group), m_mesh_points(make_uint3(0, 0, 0)),
      m_global_dim(make_uint3(0, 0, 0)), m_n_ghost_cells(make_uint3(0, 0, 0)),
      m_grid_dim(make_uint3(0, 0, 0)), m_ghost_width(make_scalar3(0.0, 0.0, 0.0)),
      m_ghost_offset(0), m_n_cells(0), m_n_inner_cells(0), m_radius(0), m_kappa(Scalar(0.0)),
      m_rcut(Scalar(0.0)), m_order(0), m_alpha(Scalar(0.0)), m_q(Scalar(0.0)),
      m_q2(Scalar(0.0)), m_need_initialize(true), m_params_set(false), m_box_changed(false),
      m_ptls_added_removed(false), m_body_energy(Scalar(0.0)), m_qstarsq(Scalar(0.0)),
      m_n_table_segments(ESP_TABLE_SEGMENTS), m_pswf_self_energy_const(Scalar(0.0)),
      m_kiss_fft(nullptr), m_kiss_ifft(nullptr), m_kiss_fft_initialized(false),
#ifdef ENABLE_MPI
      m_dfft_initialized(false),
#endif
      m_kiss_fft_initialized_priv(false)
    {
    if (!m_nlist)
        {
        throw std::runtime_error("ESPForceCompute: NeighborList pointer is null.");
        }

    if (!m_group)
        {
        throw std::runtime_error("ESPForceCompute: ParticleGroup pointer is null.");
        }

    m_pdata->getBoxChangeSignal().connect<ESPForceCompute, &ESPForceCompute::setBoxChange>(this);
    m_pdata->getGlobalParticleNumberChangeSignal()
        .connect<ESPForceCompute, &ESPForceCompute::slotGlobalParticleNumberChange>(this);

    m_force.zeroFill();
    m_virial.zeroFill();
    }

ESPForceCompute::~ESPForceCompute()
    {
    m_pdata->getGlobalParticleNumberChangeSignal().disconnect<ESPForceCompute,
                                                              &ESPForceCompute::slotGlobalParticleNumberChange>(this);
    m_pdata->getBoxChangeSignal().disconnect<ESPForceCompute, &ESPForceCompute::setBoxChange>(this);

    if (m_kiss_fft_initialized)
        {
        kiss_fft_free(m_kiss_fft);
        kiss_fft_free(m_kiss_ifft);
        kiss_fft_cleanup();
        m_kiss_fft = nullptr;
        m_kiss_ifft = nullptr;
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

void ESPForceCompute::setParams(unsigned int nx,
                                unsigned int ny,
                                unsigned int nz,
                                unsigned int order,
                                Scalar kappa,
                                Scalar rcut,
                                Scalar alpha,
                                unsigned int n_table)
    {
    if (nx == 0 || ny == 0 || nz == 0)
        {
        throw std::runtime_error("ESPForceCompute: Mesh dimensions must be positive.");
        }

    if (order < 1 || order > ESP_MAX_ORDER)
        {
        std::ostringstream oss;
        oss << "ESPForceCompute: Invalid PSWF interpolation order " << order
            << ". Valid range is [1, " << ESP_MAX_ORDER << "].";
        throw std::runtime_error(oss.str());
        }

    if (kappa <= Scalar(0.0))
        {
        throw std::runtime_error("ESPForceCompute: kappa must be positive.");
        }

    if (rcut <= Scalar(0.0))
        {
        throw std::runtime_error("ESPForceCompute: rcut must be positive.");
        }

    if (n_table < 16)
        {
        throw std::runtime_error("ESPForceCompute: lookup table must contain at least 16 segments.");
        }

    m_kappa = kappa;
    m_rcut = rcut;
    m_alpha = alpha;
    m_order = static_cast<int>(order);
    m_radius = static_cast<unsigned int>((m_order + 1) / 2);
    m_n_table_segments = n_table;

    m_global_dim = make_uint3(nx, ny, nz);
    m_mesh_points = m_global_dim;

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        const Index3D& didx = m_pdata->getDomainDecomposition()->getDomainIndexer();

        if (!is_pow2(nx) || !is_pow2(ny) || !is_pow2(nz))
            {
            throw std::runtime_error("ESPForceCompute: distributed FFT mesh dimensions must be powers of two.");
            }

        if (nx % didx.getW())
            {
            throw std::runtime_error("ESPForceCompute: nx must be divisible by processor-grid width.");
            }
        if (ny % didx.getH())
            {
            throw std::runtime_error("ESPForceCompute: ny must be divisible by processor-grid height.");
            }
        if (nz % didx.getD())
            {
            throw std::runtime_error("ESPForceCompute: nz must be divisible by processor-grid depth.");
            }

        m_mesh_points.x = nx / didx.getW();
        m_mesh_points.y = ny / didx.getH();
        m_mesh_points.z = nz / didx.getD();
        }
#endif

    GPUArray<Scalar> rho_coeff(static_cast<size_t>(order) * static_cast<size_t>(2 * order + 1), m_exec_conf);
    m_rho_coeff.swap(rho_coeff);

    GPUArray<Scalar> gf_b(order, m_exec_conf);
    m_gf_b.swap(gf_b);

    m_need_initialize = true;
    m_params_set = true;
    m_box_changed = true;
    }

void ESPForceCompute::computeForces(uint64_t timestep)
    {
    if (!m_params_set)
        {
        throw std::runtime_error("ESPForceCompute: call setParams() before computeForces().");
        }

    if (m_box_changed || m_ptls_added_removed)
        {
        m_need_initialize = true;
        }

    if (m_need_initialize)
        {
        setupMesh();
        initializeFFT();
        setupCoeffs();
        computeInfluenceFunction();
        m_need_initialize = false;
        m_box_changed = false;
        m_ptls_added_removed = false;
        }

    m_force.zeroFill();
    m_virial.zeroFill();

    {
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    for (unsigned int idx = 0; idx < m_pdata->getN(); ++idx)
        {
        h_force.data[idx] = make_scalar4(Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0));
        }
    }

    m_q = Scalar(0.0);
    m_q2 = Scalar(0.0);
    {
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
    for (unsigned int idx = 0; idx < m_pdata->getN(); ++idx)
        {
        const Scalar q = h_charge.data[idx];
        m_q += q;
        m_q2 += q * q;
        }
    }

    assignParticles();
    updateMeshes();
    interpolateForces();
    computeBodyCorrection();
    const Scalar pe = computePE();
    computeVirial();
    fixExclusions();

    const unsigned int group_size = m_group->getNumMembers();
    if (group_size > 0)
        {
        ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::readwrite);
        const Scalar pe_share = pe / Scalar(group_size);
        for (unsigned int group_idx = 0; group_idx < group_size; ++group_idx)
            {
            const unsigned int idx = m_group->getMemberIndex(group_idx);
            h_force.data[idx].w += pe_share;
            }
        }

    (void)timestep;
    }

void ESPForceCompute::setupMesh()
    {
    m_n_ghost_cells = computeGhostCellNum();

    m_grid_dim.x = m_mesh_points.x + 2 * m_n_ghost_cells.x;
    m_grid_dim.y = m_mesh_points.y + 2 * m_n_ghost_cells.y;
    m_grid_dim.z = m_mesh_points.z + 2 * m_n_ghost_cells.z;

    m_n_inner_cells = m_mesh_points.x * m_mesh_points.y * m_mesh_points.z;
    m_n_cells = m_grid_dim.x * m_grid_dim.y * m_grid_dim.z;

    m_ghost_offset = index_3d(m_n_ghost_cells.x, m_n_ghost_cells.y, m_n_ghost_cells.z, m_grid_dim);

    const BoxDim box = m_pdata->getBox();
    const Scalar3 L = box.getL();
    m_ghost_width = make_scalar3(safe_div(L.x * Scalar(m_n_ghost_cells.x), Scalar(m_global_dim.x)),
                                 safe_div(L.y * Scalar(m_n_ghost_cells.y), Scalar(m_global_dim.y)),
                                 safe_div(L.z * Scalar(m_n_ghost_cells.z), Scalar(m_global_dim.z)));
    }

void ESPForceCompute::initializeFFT()
    {
    if (m_kiss_fft_initialized)
        {
        kiss_fft_free(m_kiss_fft);
        kiss_fft_free(m_kiss_ifft);
        kiss_fft_cleanup();
        m_kiss_fft = nullptr;
        m_kiss_ifft = nullptr;
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

    GPUArray<kiss_fft_cpx> mesh(m_n_cells, m_exec_conf);
    m_mesh.swap(mesh);

    GPUArray<kiss_fft_cpx> fourier_mesh(m_n_inner_cells, m_exec_conf);
    m_fourier_mesh.swap(fourier_mesh);

    GPUArray<kiss_fft_cpx> fourier_mesh_G_x(m_n_inner_cells, m_exec_conf);
    m_fourier_mesh_G_x.swap(fourier_mesh_G_x);

    GPUArray<kiss_fft_cpx> fourier_mesh_G_y(m_n_inner_cells, m_exec_conf);
    m_fourier_mesh_G_y.swap(fourier_mesh_G_y);

    GPUArray<kiss_fft_cpx> fourier_mesh_G_z(m_n_inner_cells, m_exec_conf);
    m_fourier_mesh_G_z.swap(fourier_mesh_G_z);

    GPUArray<kiss_fft_cpx> inv_fourier_mesh_x(m_n_cells, m_exec_conf);
    m_inv_fourier_mesh_x.swap(inv_fourier_mesh_x);

    GPUArray<kiss_fft_cpx> inv_fourier_mesh_y(m_n_cells, m_exec_conf);
    m_inv_fourier_mesh_y.swap(inv_fourier_mesh_y);

    GPUArray<kiss_fft_cpx> inv_fourier_mesh_z(m_n_cells, m_exec_conf);
    m_inv_fourier_mesh_z.swap(inv_fourier_mesh_z);

    GPUArray<Scalar> inf_f(m_n_inner_cells, m_exec_conf);
    m_inf_f.swap(inf_f);

    GPUArray<Scalar3> kmesh(m_n_inner_cells, m_exec_conf);
    m_k.swap(kmesh);

    GPUArray<Scalar> virial_mesh(6 * m_n_inner_cells, m_exec_conf);
    m_virial_mesh.swap(virial_mesh);

    GPUArray<Scalar> pswf_table_gpu(static_cast<size_t>(m_n_table_segments) * 12, m_exec_conf);
    m_pswf_table_gpu.swap(pswf_table_gpu);

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        const Index3D decomp_idx = m_pdata->getDomainDecomposition()->getDomainIndexer();
        int gdim[3];
        int pdim[3];
        int pidx[3];
        int embed[3];

        pdim[0] = static_cast<int>(decomp_idx.getD());
        pdim[1] = static_cast<int>(decomp_idx.getH());
        pdim[2] = static_cast<int>(decomp_idx.getW());

        gdim[0] = static_cast<int>(m_mesh_points.z * decomp_idx.getD());
        gdim[1] = static_cast<int>(m_mesh_points.y * decomp_idx.getH());
        gdim[2] = static_cast<int>(m_mesh_points.x * decomp_idx.getW());

        embed[0] = static_cast<int>(m_grid_dim.z);
        embed[1] = static_cast<int>(m_grid_dim.y);
        embed[2] = static_cast<int>(m_grid_dim.x);

        const uint3 pcoord = m_pdata->getDomainDecomposition()->getGridPos();
        pidx[0] = static_cast<int>(pcoord.z);
        pidx[1] = static_cast<int>(pcoord.y);
        pidx[2] = static_cast<int>(pcoord.x);

        int row_m = 0;

        ArrayHandle<unsigned int> h_cart_ranks(m_pdata->getDomainDecomposition()->getCartRanks(),
                                               access_location::host,
                                               access_mode::read);

        dfft_create_plan(&m_dfft_plan_forward,
                         3,
                         gdim,
                         embed,
                         nullptr,
                         pdim,
                         pidx,
                         row_m,
                         0,
                         1,
                         m_exec_conf->getMPICommunicator(),
                         reinterpret_cast<int*>(h_cart_ranks.data));

        dfft_create_plan(&m_dfft_plan_inverse,
                         3,
                         gdim,
                         nullptr,
                         embed,
                         pdim,
                         pidx,
                         row_m,
                         0,
                         1,
                         m_exec_conf->getMPICommunicator(),
                         reinterpret_cast<int*>(h_cart_ranks.data));

        m_grid_comm_forward.reset(new CommunicatorGrid<kiss_fft_cpx>(m_sysdef,
                                                                     make_uint3(m_mesh_points.x,
                                                                                m_mesh_points.y,
                                                                                m_mesh_points.z),
                                                                     make_uint3(m_grid_dim.x,
                                                                                m_grid_dim.y,
                                                                                m_grid_dim.z),
                                                                     m_n_ghost_cells,
                                                                     true));

        m_grid_comm_reverse.reset(new CommunicatorGrid<kiss_fft_cpx>(m_sysdef,
                                                                     make_uint3(m_mesh_points.x,
                                                                                m_mesh_points.y,
                                                                                m_mesh_points.z),
                                                                     make_uint3(m_grid_dim.x,
                                                                                m_grid_dim.y,
                                                                                m_grid_dim.z),
                                                                     m_n_ghost_cells,
                                                                     false));

        m_dfft_initialized = true;
        }
    else
#endif
        {
        int dims[3];
        dims[0] = static_cast<int>(m_mesh_points.z);
        dims[1] = static_cast<int>(m_mesh_points.y);
        dims[2] = static_cast<int>(m_mesh_points.x);

        m_kiss_fft = kiss_fftnd_alloc(dims, 3, 0, nullptr, nullptr);
        m_kiss_ifft = kiss_fftnd_alloc(dims, 3, 1, nullptr, nullptr);

        if (!m_kiss_fft || !m_kiss_ifft)
            {
            throw std::runtime_error("ESPForceCompute: failed to allocate KISS FFT plans.");
            }

        m_kiss_fft_initialized = true;
        m_kiss_fft_initialized_priv = true;
        }
    }

void ESPForceCompute::computeInfluenceFunction()
    {
    ArrayHandle<Scalar> h_inf_f(m_inf_f, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar3> h_k(m_k, access_location::host, access_mode::overwrite);

    const BoxDim box = m_pdata->getGlobalBox();
    const Scalar3 L = box.getL();

    unsigned int kx_offset = 0;
    unsigned int ky_offset = 0;
    unsigned int kz_offset = 0;

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

    for (unsigned int kz_local = 0; kz_local < m_mesh_points.z; ++kz_local)
        {
        const unsigned int kz_global = kz_local + kz_offset;
        int nz = (kz_global <= m_global_dim.z / 2) ? int(kz_global) : int(kz_global) - int(m_global_dim.z);
        const Scalar kz_val = ESP_TWO_PI * Scalar(nz) / L.z;

        for (unsigned int ky_local = 0; ky_local < m_mesh_points.y; ++ky_local)
            {
            const unsigned int ky_global = ky_local + ky_offset;
            int ny = (ky_global <= m_global_dim.y / 2) ? int(ky_global) : int(ky_global) - int(m_global_dim.y);
            const Scalar ky_val = ESP_TWO_PI * Scalar(ny) / L.y;

            for (unsigned int kx_local = 0; kx_local < m_mesh_points.x; ++kx_local)
                {
                const unsigned int kx_global = kx_local + kx_offset;
                int nx = (kx_global <= m_global_dim.x / 2) ? int(kx_global) : int(kx_global) - int(m_global_dim.x);
                const Scalar kx_val = ESP_TWO_PI * Scalar(nx) / L.x;

                const unsigned int idx = index_3d(kx_local, ky_local, kz_local, m_mesh_points);

                const Scalar3 kval = make_scalar3(kx_val, ky_val, kz_val);
                h_k.data[idx] = kval;

                const Scalar k2 = kx_val * kx_val + ky_val * ky_val + kz_val * kz_val;

                if (k2 < Scalar(1.0e-16))
                    {
                    h_inf_f.data[idx] = Scalar(0.0);
                    continue;
                    }

                const Scalar sx = Scalar(0.5) * kx_val * hx;
                const Scalar sy = Scalar(0.5) * ky_val * hy;
                const Scalar sz = Scalar(0.5) * kz_val * hz;

                const Scalar wx = std::pow(std::fabs(sinc(sx)), m_order);
                const Scalar wy = std::pow(std::fabs(sinc(sy)), m_order);
                const Scalar wz = std::pow(std::fabs(sinc(sz)), m_order);

                const Scalar compact_x = Scalar(1.0) / (Scalar(1.0) + sqr(sx) / Scalar(m_order + 1));
                const Scalar compact_y = Scalar(1.0) / (Scalar(1.0) + sqr(sy) / Scalar(m_order + 1));
                const Scalar compact_z = Scalar(1.0) / (Scalar(1.0) + sqr(sz) / Scalar(m_order + 1));

                const Scalar pswf_hat = wx * wy * wz * compact_x * compact_y * compact_z;

                const Scalar denom = std::max(gf_denom_pswf(sx * sx, sy * sy, sz * sz), Scalar(1.0e-12));
                const Scalar screened_k2 = k2 + m_alpha * m_alpha;

                h_inf_f.data[idx] = (pswf_hat * pswf_hat) / (screened_k2 * denom);
                }
            }
        }

    const Scalar qstar = Scalar(0.25)
                         * (sqr(ESP_TWO_PI / hx) + sqr(ESP_TWO_PI / hy) + sqr(ESP_TWO_PI / hz));
    m_qstarsq = qstar;
    }

void ESPForceCompute::assignParticles()
    {
    ArrayHandle<kiss_fft_cpx> h_mesh(m_mesh, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_rho_coeff(m_rho_coeff, access_location::host, access_mode::read);

    for (unsigned int i = 0; i < m_n_cells; ++i)
        {
        h_mesh.data[i].r = 0;
        h_mesh.data[i].i = 0;
        }

    const BoxDim box = m_pdata->getBox();
    const int mult_fact = 2 * m_order + 1;
    const int nlower = -(m_order - 1) / 2;
    const int nupper = m_order / 2;

    for (unsigned int group_idx = 0; group_idx < m_group->getNumMembers(); ++group_idx)
        {
        const unsigned int idx = m_group->getMemberIndex(group_idx);
        const Scalar4 postype = h_postype.data[idx];
        const Scalar qi = h_charge.data[idx];

        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        if (std::isnan(pos.x) || std::isnan(pos.y) || std::isnan(pos.z))
            continue;

        Scalar3 f = box.makeFraction(pos);
        Scalar3 reduced_pos = make_scalar3(f.x * Scalar(m_mesh_points.x),
                                           f.y * Scalar(m_mesh_points.y),
                                           f.z * Scalar(m_mesh_points.z));
        reduced_pos.x += Scalar(m_n_ghost_cells.x);
        reduced_pos.y += Scalar(m_n_ghost_cells.y);
        reduced_pos.z += Scalar(m_n_ghost_cells.z);

        Scalar shift = (m_order % 2) ? Scalar(0.5) : Scalar(0.0);
        Scalar shiftone = (m_order % 2) ? Scalar(0.0) : Scalar(0.5);

        int ix = int(reduced_pos.x + shift);
        int iy = int(reduced_pos.y + shift);
        int iz = int(reduced_pos.z + shift);

        Scalar dx = shiftone + Scalar(ix) - reduced_pos.x;
        Scalar dy = shiftone + Scalar(iy) - reduced_pos.y;
        Scalar dz = shiftone + Scalar(iz) - reduced_pos.z;

        if (ix == int(m_grid_dim.x) && !m_n_ghost_cells.x)
            ix = 0;
        if (iy == int(m_grid_dim.y) && !m_n_ghost_cells.y)
            iy = 0;
        if (iz == int(m_grid_dim.z) && !m_n_ghost_cells.z)
            iz = 0;

        for (int i = nlower; i <= nupper; ++i)
            {
            Scalar Wx = Scalar(0.0);
            for (int iorder = m_order - 1; iorder >= 0; --iorder)
                {
                Wx = h_rho_coeff.data[(i - nlower) + iorder * mult_fact] + Wx * dx;
                }

            int neighi = ix + i;
            if (!m_n_ghost_cells.x)
                neighi = wrap_index(neighi, int(m_grid_dim.x));
            if (neighi < 0 || neighi >= int(m_grid_dim.x))
                continue;

            for (int j = nlower; j <= nupper; ++j)
                {
                Scalar Wy = Scalar(0.0);
                for (int iorder = m_order - 1; iorder >= 0; --iorder)
                    {
                    Wy = h_rho_coeff.data[(j - nlower) + iorder * mult_fact] + Wy * dy;
                    }

                int neighj = iy + j;
                if (!m_n_ghost_cells.y)
                    neighj = wrap_index(neighj, int(m_grid_dim.y));
                if (neighj < 0 || neighj >= int(m_grid_dim.y))
                    continue;

                for (int k = nlower; k <= nupper; ++k)
                    {
                    Scalar Wz = Scalar(0.0);
                    for (int iorder = m_order - 1; iorder >= 0; --iorder)
                        {
                        Wz = h_rho_coeff.data[(k - nlower) + iorder * mult_fact] + Wz * dz;
                        }

                    int neighk = iz + k;
                    if (!m_n_ghost_cells.z)
                        neighk = wrap_index(neighk, int(m_grid_dim.z));
                    if (neighk < 0 || neighk >= int(m_grid_dim.z))
                        continue;

                    const unsigned int mesh_idx = index_3d(static_cast<unsigned int>(neighi),
                                                           static_cast<unsigned int>(neighj),
                                                           static_cast<unsigned int>(neighk),
                                                           m_grid_dim);

                    h_mesh.data[mesh_idx].r += static_cast<kiss_fft_scalar>(qi * Wx * Wy * Wz);
                    }
                }
            }
        }
    }

void ESPForceCompute::updateMeshes()
    {
#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        m_grid_comm_forward->communicate(m_mesh);

        ArrayHandle<kiss_fft_cpx> h_mesh(m_mesh, access_location::host, access_mode::read);
        ArrayHandle<kiss_fft_cpx> h_fourier_mesh(m_fourier_mesh, access_location::host, access_mode::overwrite);

        dfft_execute(reinterpret_cast<cpx_t*>(h_mesh.data + m_ghost_offset),
                     reinterpret_cast<cpx_t*>(h_fourier_mesh.data),
                     0,
                     m_dfft_plan_forward);
        }
    else
#endif
        {
        ArrayHandle<kiss_fft_cpx> h_mesh(m_mesh, access_location::host, access_mode::read);
        ArrayHandle<kiss_fft_cpx> h_fourier_mesh(m_fourier_mesh, access_location::host, access_mode::overwrite);

        const size_t n = static_cast<size_t>(m_n_inner_cells);
        std::vector<kiss_fft_cpx> local_in(n);
        std::vector<kiss_fft_cpx> local_out(n);

        for (size_t i = 0; i < n; ++i)
            local_in[i] = h_mesh.data[i + m_ghost_offset];

        kiss_fftnd(m_kiss_fft, local_in.data(), local_out.data());

        for (size_t i = 0; i < n; ++i)
            h_fourier_mesh.data[i] = local_out[i];
        }

    {
    ArrayHandle<Scalar> h_inf_f(m_inf_f, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_k(m_k, access_location::host, access_mode::read);
    ArrayHandle<kiss_fft_cpx> h_fourier_mesh(m_fourier_mesh, access_location::host, access_mode::read);
    ArrayHandle<kiss_fft_cpx> h_fourier_mesh_G_x(m_fourier_mesh_G_x,
                                                 access_location::host,
                                                 access_mode::overwrite);
    ArrayHandle<kiss_fft_cpx> h_fourier_mesh_G_y(m_fourier_mesh_G_y,
                                                 access_location::host,
                                                 access_mode::overwrite);
    ArrayHandle<kiss_fft_cpx> h_fourier_mesh_G_z(m_fourier_mesh_G_z,
                                                 access_location::host,
                                                 access_mode::overwrite);

    const unsigned int NNN = m_global_dim.x * m_global_dim.y * m_global_dim.z;
    const Scalar inv_NNN = Scalar(1.0) / Scalar(NNN);

    for (unsigned int idx = 0; idx < m_n_inner_cells; ++idx)
        {
        const kiss_fft_cpx f = h_fourier_mesh.data[idx];
        const Scalar scale = h_inf_f.data[idx] * inv_NNN;
        const Scalar3 kval = h_k.data[idx];

        h_fourier_mesh_G_x.data[idx].r = static_cast<kiss_fft_scalar>( f.i * kval.x * scale);
        h_fourier_mesh_G_x.data[idx].i = static_cast<kiss_fft_scalar>(-f.r * kval.x * scale);

        h_fourier_mesh_G_y.data[idx].r = static_cast<kiss_fft_scalar>( f.i * kval.y * scale);
        h_fourier_mesh_G_y.data[idx].i = static_cast<kiss_fft_scalar>(-f.r * kval.y * scale);

        h_fourier_mesh_G_z.data[idx].r = static_cast<kiss_fft_scalar>( f.i * kval.z * scale);
        h_fourier_mesh_G_z.data[idx].i = static_cast<kiss_fft_scalar>(-f.r * kval.z * scale);
        }
    }

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        ArrayHandle<kiss_fft_cpx> h_fourier_mesh_G_x(m_fourier_mesh_G_x,
                                                     access_location::host,
                                                     access_mode::read);
        ArrayHandle<kiss_fft_cpx> h_fourier_mesh_G_y(m_fourier_mesh_G_y,
                                                     access_location::host,
                                                     access_mode::read);
        ArrayHandle<kiss_fft_cpx> h_fourier_mesh_G_z(m_fourier_mesh_G_z,
                                                     access_location::host,
                                                     access_mode::read);

        ArrayHandle<kiss_fft_cpx> h_inv_fourier_mesh_x(m_inv_fourier_mesh_x,
                                                       access_location::host,
                                                       access_mode::overwrite);
        ArrayHandle<kiss_fft_cpx> h_inv_fourier_mesh_y(m_inv_fourier_mesh_y,
                                                       access_location::host,
                                                       access_mode::overwrite);
        ArrayHandle<kiss_fft_cpx> h_inv_fourier_mesh_z(m_inv_fourier_mesh_z,
                                                       access_location::host,
                                                       access_mode::overwrite);

        dfft_execute(reinterpret_cast<cpx_t*>(h_fourier_mesh_G_x.data),
                     reinterpret_cast<cpx_t*>(h_inv_fourier_mesh_x.data + m_ghost_offset),
                     1,
                     m_dfft_plan_inverse);

        dfft_execute(reinterpret_cast<cpx_t*>(h_fourier_mesh_G_y.data),
                     reinterpret_cast<cpx_t*>(h_inv_fourier_mesh_y.data + m_ghost_offset),
                     1,
                     m_dfft_plan_inverse);

        dfft_execute(reinterpret_cast<cpx_t*>(h_fourier_mesh_G_z.data),
                     reinterpret_cast<cpx_t*>(h_inv_fourier_mesh_z.data + m_ghost_offset),
                     1,
                     m_dfft_plan_inverse);

        m_grid_comm_reverse->communicate(m_inv_fourier_mesh_x);
        m_grid_comm_reverse->communicate(m_inv_fourier_mesh_y);
        m_grid_comm_reverse->communicate(m_inv_fourier_mesh_z);
        }
    else
#endif
        {
        const size_t n = static_cast<size_t>(m_n_inner_cells);

        ArrayHandle<kiss_fft_cpx> h_fourier_mesh_G_x(m_fourier_mesh_G_x,
                                                     access_location::host,
                                                     access_mode::read);
        ArrayHandle<kiss_fft_cpx> h_fourier_mesh_G_y(m_fourier_mesh_G_y,
                                                     access_location::host,
                                                     access_mode::read);
        ArrayHandle<kiss_fft_cpx> h_fourier_mesh_G_z(m_fourier_mesh_G_z,
                                                     access_location::host,
                                                     access_mode::read);

        ArrayHandle<kiss_fft_cpx> h_inv_fourier_mesh_x(m_inv_fourier_mesh_x,
                                                       access_location::host,
                                                       access_mode::overwrite);
        ArrayHandle<kiss_fft_cpx> h_inv_fourier_mesh_y(m_inv_fourier_mesh_y,
                                                       access_location::host,
                                                       access_mode::overwrite);
        ArrayHandle<kiss_fft_cpx> h_inv_fourier_mesh_z(m_inv_fourier_mesh_z,
                                                       access_location::host,
                                                       access_mode::overwrite);

        std::vector<kiss_fft_cpx> in(n), out(n);

        for (size_t i = 0; i < n; ++i)
            in[i] = h_fourier_mesh_G_x.data[i];
        kiss_fftnd(m_kiss_ifft, in.data(), out.data());
        for (size_t i = 0; i < n; ++i)
            h_inv_fourier_mesh_x.data[i + m_ghost_offset] = out[i];

        for (size_t i = 0; i < n; ++i)
            in[i] = h_fourier_mesh_G_y.data[i];
        kiss_fftnd(m_kiss_ifft, in.data(), out.data());
        for (size_t i = 0; i < n; ++i)
            h_inv_fourier_mesh_y.data[i + m_ghost_offset] = out[i];

        for (size_t i = 0; i < n; ++i)
            in[i] = h_fourier_mesh_G_z.data[i];
        kiss_fftnd(m_kiss_ifft, in.data(), out.data());
        for (size_t i = 0; i < n; ++i)
            h_inv_fourier_mesh_z.data[i + m_ghost_offset] = out[i];
        }
    }

void ESPForceCompute::interpolateForces()
    {
    ArrayHandle<Scalar4> h_postype(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
    ArrayHandle<kiss_fft_cpx> h_inv_fourier_mesh_x(m_inv_fourier_mesh_x,
                                                   access_location::host,
                                                   access_mode::read);
    ArrayHandle<kiss_fft_cpx> h_inv_fourier_mesh_y(m_inv_fourier_mesh_y,
                                                   access_location::host,
                                                   access_mode::read);
    ArrayHandle<kiss_fft_cpx> h_inv_fourier_mesh_z(m_inv_fourier_mesh_z,
                                                   access_location::host,
                                                   access_mode::read);
    ArrayHandle<Scalar> h_rho_coeff(m_rho_coeff, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::readwrite);

    const BoxDim box = m_pdata->getBox();
    const int mult_fact = 2 * m_order + 1;
    const int nlower = -(m_order - 1) / 2;
    const int nupper = m_order / 2;
    const Scalar norm = Scalar(1.0) / Scalar(m_global_dim.x * m_global_dim.y * m_global_dim.z);

    for (unsigned int group_idx = 0; group_idx < m_group->getNumMembers(); ++group_idx)
        {
        const unsigned int idx = m_group->getMemberIndex(group_idx);
        const Scalar4 postype = h_postype.data[idx];
        const Scalar qi = h_charge.data[idx];

        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        if (std::isnan(pos.x) || std::isnan(pos.y) || std::isnan(pos.z))
            continue;

        Scalar3 f = box.makeFraction(pos);
        Scalar3 reduced_pos = make_scalar3(f.x * Scalar(m_mesh_points.x),
                                           f.y * Scalar(m_mesh_points.y),
                                           f.z * Scalar(m_mesh_points.z));
        reduced_pos.x += Scalar(m_n_ghost_cells.x);
        reduced_pos.y += Scalar(m_n_ghost_cells.y);
        reduced_pos.z += Scalar(m_n_ghost_cells.z);

        Scalar shift = (m_order % 2) ? Scalar(0.5) : Scalar(0.0);
        Scalar shiftone = (m_order % 2) ? Scalar(0.0) : Scalar(0.5);

        int ix = int(reduced_pos.x + shift);
        int iy = int(reduced_pos.y + shift);
        int iz = int(reduced_pos.z + shift);

        Scalar dx = shiftone + Scalar(ix) - reduced_pos.x;
        Scalar dy = shiftone + Scalar(iy) - reduced_pos.y;
        Scalar dz = shiftone + Scalar(iz) - reduced_pos.z;

        if (ix == int(m_grid_dim.x) && !m_n_ghost_cells.x)
            ix = 0;
        if (iy == int(m_grid_dim.y) && !m_n_ghost_cells.y)
            iy = 0;
        if (iz == int(m_grid_dim.z) && !m_n_ghost_cells.z)
            iz = 0;

        Scalar3 force = make_scalar3(0.0, 0.0, 0.0);

        for (int i = nlower; i <= nupper; ++i)
            {
            Scalar Wx = Scalar(0.0);
            for (int iorder = m_order - 1; iorder >= 0; --iorder)
                Wx = h_rho_coeff.data[(i - nlower) + iorder * mult_fact] + Wx * dx;

            int neighi = ix + i;
            if (!m_n_ghost_cells.x)
                neighi = wrap_index(neighi, int(m_grid_dim.x));
            if (neighi < 0 || neighi >= int(m_grid_dim.x))
                continue;

            for (int j = nlower; j <= nupper; ++j)
                {
                Scalar Wy = Scalar(0.0);
                for (int iorder = m_order - 1; iorder >= 0; --iorder)
                    Wy = h_rho_coeff.data[(j - nlower) + iorder * mult_fact] + Wy * dy;

                int neighj = iy + j;
                if (!m_n_ghost_cells.y)
                    neighj = wrap_index(neighj, int(m_grid_dim.y));
                if (neighj < 0 || neighj >= int(m_grid_dim.y))
                    continue;

                for (int k = nlower; k <= nupper; ++k)
                    {
                    Scalar Wz = Scalar(0.0);
                    for (int iorder = m_order - 1; iorder >= 0; --iorder)
                        Wz = h_rho_coeff.data[(k - nlower) + iorder * mult_fact] + Wz * dz;

                    int neighk = iz + k;
                    if (!m_n_ghost_cells.z)
                        neighk = wrap_index(neighk, int(m_grid_dim.z));
                    if (neighk < 0 || neighk >= int(m_grid_dim.z))
                        continue;

                    const unsigned int mesh_idx = index_3d(static_cast<unsigned int>(neighi),
                                                           static_cast<unsigned int>(neighj),
                                                           static_cast<unsigned int>(neighk),
                                                           m_grid_dim);

                    const Scalar w = Wx * Wy * Wz;
                    force.x += qi * w * Scalar(h_inv_fourier_mesh_x.data[mesh_idx].r) * norm;
                    force.y += qi * w * Scalar(h_inv_fourier_mesh_y.data[mesh_idx].r) * norm;
                    force.z += qi * w * Scalar(h_inv_fourier_mesh_z.data[mesh_idx].r) * norm;
                    }
                }
            }

        h_force.data[idx].x += force.x;
        h_force.data[idx].y += force.y;
        h_force.data[idx].z += force.z;
        }
    }

Scalar ESPForceCompute::computePE()
    {
    ArrayHandle<kiss_fft_cpx> h_fourier_mesh(m_fourier_mesh, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_inf_f(m_inf_f, access_location::host, access_mode::read);

    const Scalar inv_NNN = Scalar(1.0) / Scalar(m_global_dim.x * m_global_dim.y * m_global_dim.z);
    Scalar pe = Scalar(0.0);

    for (unsigned int idx = 0; idx < m_n_inner_cells; ++idx)
        {
        const Scalar re = Scalar(h_fourier_mesh.data[idx].r);
        const Scalar im = Scalar(h_fourier_mesh.data[idx].i);
        const Scalar mod2 = re * re + im * im;
        pe += Scalar(0.5) * h_inf_f.data[idx] * mod2 * inv_NNN;
        }

    pe -= m_pswf_self_energy_const * m_q2;
    pe += m_body_energy;
    return pe;
    }

void ESPForceCompute::computeVirialMesh()
    {
    ArrayHandle<kiss_fft_cpx> h_fourier_mesh(m_fourier_mesh, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_inf_f(m_inf_f, access_location::host, access_mode::read);
    ArrayHandle<Scalar3> h_k(m_k, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_virial_mesh(m_virial_mesh, access_location::host, access_mode::overwrite);

    const Scalar inv_NNN = Scalar(1.0) / Scalar(m_global_dim.x * m_global_dim.y * m_global_dim.z);

    for (unsigned int idx = 0; idx < m_n_inner_cells; ++idx)
        {
        const Scalar re = Scalar(h_fourier_mesh.data[idx].r);
        const Scalar im = Scalar(h_fourier_mesh.data[idx].i);
        const Scalar mod2 = (re * re + im * im) * h_inf_f.data[idx] * inv_NNN;

        const Scalar3 k = h_k.data[idx];
        const Scalar k2 = k.x * k.x + k.y * k.y + k.z * k.z;

        Scalar scale = (k2 > Scalar(1.0e-14)) ? (mod2 / k2) : Scalar(0.0);

        h_virial_mesh.data[0 * m_n_inner_cells + idx] = scale * k.x * k.x;
        h_virial_mesh.data[1 * m_n_inner_cells + idx] = scale * k.x * k.y;
        h_virial_mesh.data[2 * m_n_inner_cells + idx] = scale * k.x * k.z;
        h_virial_mesh.data[3 * m_n_inner_cells + idx] = scale * k.y * k.y;
        h_virial_mesh.data[4 * m_n_inner_cells + idx] = scale * k.y * k.z;
        h_virial_mesh.data[5 * m_n_inner_cells + idx] = scale * k.z * k.z;
        }
    }

void ESPForceCompute::computeVirial()
    {
    computeVirialMesh();

    ArrayHandle<Scalar> h_virial_mesh(m_virial_mesh, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);

    Scalar totals[6] = {Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0), Scalar(0.0)};

    for (unsigned int comp = 0; comp < 6; ++comp)
        {
        for (unsigned int idx = 0; idx < m_n_inner_cells; ++idx)
            totals[comp] += h_virial_mesh.data[comp * m_n_inner_cells + idx];
        }

    const size_t pitch = m_virial.getPitch();
    for (unsigned int comp = 0; comp < 6; ++comp)
        {
        for (unsigned int idx = 0; idx < m_pdata->getN(); ++idx)
            h_virial.data[comp * pitch + idx] = Scalar(0.0);
        }

    const unsigned int group_size = m_group->getNumMembers();
    if (group_size > 0)
        {
        for (unsigned int group_idx = 0; group_idx < group_size; ++group_idx)
            {
            const unsigned int idx = m_group->getMemberIndex(group_idx);
            for (unsigned int comp = 0; comp < 6; ++comp)
                {
                h_virial.data[comp * pitch + idx] = totals[comp] / Scalar(group_size);
                }
            }
        }
    }

void ESPForceCompute::fixExclusions()
    {
    if (!m_nlist->getExclusionsSet())
        return;

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::readwrite);

    ArrayHandle<unsigned int> h_n_ex(m_nlist->getNExArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_exlist(m_nlist->getExListArray(), access_location::host, access_mode::read);
    const Index2D ex_indexer = m_nlist->getExListIndexer();

    const size_t virial_pitch = m_virial.getPitch();

    auto eval_table_force = [this](Scalar r, Scalar& Lr, Scalar& minus_dLdr)
        {
        if (r >= m_rcut)
            {
            Lr = evalL_direct(r);
            minus_dLdr = evalDL_direct(r);
            return;
            }

        const Scalar s = std::max(Scalar(0.0), std::min(Scalar(0.999999999), r / m_rcut));
        const unsigned int seg = std::min(static_cast<unsigned int>(s * Scalar(m_n_table_segments)),
                                          m_n_table_segments - 1);

        const ESPTableEntry& e = m_pswf_table_cpu[seg];
        const Scalar t = (r - e.r_lo) * e.dr_inv;

        const std::vector<Scalar> pc
            = {e.potential_coeffs.x, e.potential_coeffs.y, e.potential_coeffs.z, e.potential_coeffs.w, e.potential_coeff4};
        const std::vector<Scalar> fc
            = {e.force_coeffs.x, e.force_coeffs.y, e.force_coeffs.z, e.force_coeffs.w, e.force_coeff4};

        Lr = horner_eval(pc, t);
        minus_dLdr = horner_eval(fc, t);
        };

    for (unsigned int i = 0; i < m_pdata->getN(); ++i)
        {
        const Scalar4 posi4 = h_pos.data[i];
        const Scalar3 posi = make_scalar3(posi4.x, posi4.y, posi4.z);
        const Scalar qi = h_charge.data[i];

        for (unsigned int ex = 0; ex < h_n_ex.data[i]; ++ex)
            {
            const unsigned int j = h_exlist.data[ex_indexer(ex, i)];
            if (j <= i || j >= m_pdata->getN())
                continue;

            const Scalar4 posj4 = h_pos.data[j];
            const Scalar3 posj = make_scalar3(posj4.x, posj4.y, posj4.z);
            const Scalar qj = h_charge.data[j];

            Scalar3 rij = posj - posi;
            rij = m_pdata->getBox().minImage(rij);

            const Scalar rsq = dot(rij, rij);
            if (rsq < Scalar(1.0e-20))
                continue;

            const Scalar r = std::sqrt(rsq);
            Scalar Lr = Scalar(0.0);
            Scalar minus_dLdr = Scalar(0.0);
            eval_table_force(r, Lr, minus_dLdr);

            const Scalar qiqj = qi * qj;
            const Scalar invr = Scalar(1.0) / r;
            const Scalar invr2 = invr * invr;

            const Scalar radial = qiqj * (invr2 - minus_dLdr);
            const Scalar force_prefac = radial * invr;

            const Scalar3 fij = make_scalar3(force_prefac * rij.x,
                                             force_prefac * rij.y,
                                             force_prefac * rij.z);

            h_force.data[i].x += fij.x;
            h_force.data[i].y += fij.y;
            h_force.data[i].z += fij.z;

            h_force.data[j].x -= fij.x;
            h_force.data[j].y -= fij.y;
            h_force.data[j].z -= fij.z;

            const Scalar vij = qiqj * (invr - Lr);
            h_force.data[i].w += Scalar(0.5) * vij;
            h_force.data[j].w += Scalar(0.5) * vij;

            const Scalar vir_xx = rij.x * fij.x;
            const Scalar vir_xy = rij.x * fij.y;
            const Scalar vir_xz = rij.x * fij.z;
            const Scalar vir_yy = rij.y * fij.y;
            const Scalar vir_yz = rij.y * fij.z;
            const Scalar vir_zz = rij.z * fij.z;

            h_virial.data[0 * virial_pitch + i] += Scalar(0.5) * vir_xx;
            h_virial.data[1 * virial_pitch + i] += Scalar(0.5) * vir_xy;
            h_virial.data[2 * virial_pitch + i] += Scalar(0.5) * vir_xz;
            h_virial.data[3 * virial_pitch + i] += Scalar(0.5) * vir_yy;
            h_virial.data[4 * virial_pitch + i] += Scalar(0.5) * vir_yz;
            h_virial.data[5 * virial_pitch + i] += Scalar(0.5) * vir_zz;

            h_virial.data[0 * virial_pitch + j] += Scalar(0.5) * vir_xx;
            h_virial.data[1 * virial_pitch + j] += Scalar(0.5) * vir_xy;
            h_virial.data[2 * virial_pitch + j] += Scalar(0.5) * vir_xz;
            h_virial.data[3 * virial_pitch + j] += Scalar(0.5) * vir_yy;
            h_virial.data[4 * virial_pitch + j] += Scalar(0.5) * vir_yz;
            h_virial.data[5 * virial_pitch + j] += Scalar(0.5) * vir_zz;
            }
        }
    }

void ESPForceCompute::setupCoeffs()
    {
    compute_pswf_rho_coeff();
    compute_pswf_gf_denom();
    buildPSWFTable();
    m_pswf_self_energy_const = computePSWFSelfEnergyConst();
    }

void ESPForceCompute::computeBodyCorrection()
    {
    m_body_energy = Scalar(0.0);
    }

void ESPForceCompute::compute_pswf_rho_coeff()
    {
    ArrayHandle<Scalar> h_rho_coeff(m_rho_coeff, access_location::host, access_mode::overwrite);

    const int mult_fact = 2 * m_order + 1;
    const int nlower = -(m_order - 1) / 2;
    const int nupper = m_order / 2;
    const Scalar radius = std::max(Scalar(1.0), Scalar(m_radius));

    for (int offset = nlower; offset <= nupper; ++offset)
        {
        std::vector<Scalar> x(static_cast<size_t>(m_order));
        std::vector<Scalar> y(static_cast<size_t>(m_order));

        for (int n = 0; n < m_order; ++n)
            {
            Scalar xi;
            if (m_order == 1)
                xi = Scalar(0.0);
            else
                xi = Scalar(-0.5) + Scalar(n) / Scalar(m_order - 1);

            const Scalar arg = std::fabs((Scalar(offset) - xi) / radius);
            const Scalar val = evalPSWFKernel(arg);

            x[static_cast<size_t>(n)] = xi;
            y[static_cast<size_t>(n)] = val;
            }

        std::vector<Scalar> coeffs = fit_power_polynomial(x, y);

        for (int deg = 0; deg < m_order; ++deg)
            {
            h_rho_coeff.data[(offset - nlower) + deg * mult_fact] = coeffs[static_cast<size_t>(deg)];
            }
        }

    for (int deg = 0; deg < m_order; ++deg)
        {
        Scalar sum = Scalar(0.0);
        for (int offset = nlower; offset <= nupper; ++offset)
            {
            sum += h_rho_coeff.data[(offset - nlower) + deg * mult_fact];
            }

        if (std::fabs(sum) > Scalar(1.0e-12))
            {
            for (int offset = nlower; offset <= nupper; ++offset)
                {
                h_rho_coeff.data[(offset - nlower) + deg * mult_fact] /= sum;
                }
            }
        }
    }

void ESPForceCompute::compute_pswf_gf_denom()
    {
    ArrayHandle<Scalar> h_gf_b(m_gf_b, access_location::host, access_mode::overwrite);

    const Scalar beta = Scalar(0.5) * Scalar(m_order) + std::max(Scalar(0.0), m_alpha * m_rcut);
    for (int l = 0; l < m_order; ++l)
        {
        Scalar coeff = Scalar((l % 2) ? -1.0 : 1.0);
        Scalar denom = Scalar(1.0);
        for (int n = 2; n <= 2 * l; ++n)
            denom *= Scalar(n);

        coeff *= std::pow(beta, Scalar(l)) / denom;
        h_gf_b.data[l] = coeff;
        }
    }

Scalar ESPForceCompute::gf_denom_pswf(Scalar x, Scalar y, Scalar z) const
    {
    ArrayHandle<Scalar> h_gf_b(m_gf_b, access_location::host, access_mode::read);

    Scalar sx = Scalar(0.0);
    Scalar sy = Scalar(0.0);
    Scalar sz = Scalar(0.0);

    for (int l = m_order - 1; l >= 0; --l)
        {
        sx = h_gf_b.data[l] + sx * x;
        sy = h_gf_b.data[l] + sy * y;
        sz = h_gf_b.data[l] + sz * z;
        }

    Scalar s = sx * sy * sz;
    return std::max(s * s, Scalar(1.0e-12));
    }

void ESPForceCompute::buildPSWFTable()
    {
    m_pswf_table_cpu.clear();
    m_pswf_table_cpu.resize(m_n_table_segments);

    const Scalar dr = m_rcut / Scalar(m_n_table_segments);

    for (unsigned int seg = 0; seg < m_n_table_segments; ++seg)
        {
        const Scalar r0 = Scalar(seg) * dr;
        const Scalar r1 = Scalar(seg + 1) * dr;

        std::vector<Scalar> t = {Scalar(0.0), Scalar(0.25), Scalar(0.5), Scalar(0.75), Scalar(1.0)};
        std::vector<Scalar> v(5), f(5);

        for (unsigned int p = 0; p < 5; ++p)
            {
            const Scalar r = r0 + (r1 - r0) * t[p];
            const Scalar rr = std::max(r, Scalar(1.0e-8) * m_rcut);
            v[p] = evalL_direct(rr);
            f[p] = evalDL_direct(rr);
            }

        std::vector<Scalar> vcoeff = fit_power_polynomial(t, v);
        std::vector<Scalar> fcoeff = fit_power_polynomial(t, f);

        ESPTableEntry entry;
        entry.potential_coeffs = make_scalar4(vcoeff[0], vcoeff[1], vcoeff[2], vcoeff[3]);
        entry.potential_coeff4 = vcoeff[4];
        entry.force_coeffs = make_scalar4(fcoeff[0], fcoeff[1], fcoeff[2], fcoeff[3]);
        entry.force_coeff4 = fcoeff[4];
        entry.r_lo = r0;
        entry.dr_inv = Scalar(1.0) / std::max(r1 - r0, Scalar(1.0e-20));
        entry._pad0 = Scalar(0.0);
        entry._pad1 = Scalar(0.0);

        m_pswf_table_cpu[seg] = entry;
        }

    ArrayHandle<Scalar> h_pswf_table_gpu(m_pswf_table_gpu, access_location::host, access_mode::overwrite);

    for (unsigned int seg = 0; seg < m_n_table_segments; ++seg)
        {
        const ESPTableEntry& e = m_pswf_table_cpu[seg];
        const unsigned int base = 12 * seg;

        h_pswf_table_gpu.data[base + 0] = e.potential_coeffs.x;
        h_pswf_table_gpu.data[base + 1] = e.potential_coeffs.y;
        h_pswf_table_gpu.data[base + 2] = e.potential_coeffs.z;
        h_pswf_table_gpu.data[base + 3] = e.potential_coeffs.w;
        h_pswf_table_gpu.data[base + 4] = e.potential_coeff4;

        h_pswf_table_gpu.data[base + 5] = e.force_coeffs.x;
        h_pswf_table_gpu.data[base + 6] = e.force_coeffs.y;
        h_pswf_table_gpu.data[base + 7] = e.force_coeffs.z;
        h_pswf_table_gpu.data[base + 8] = e.force_coeffs.w;
        h_pswf_table_gpu.data[base + 9] = e.force_coeff4;

        h_pswf_table_gpu.data[base + 10] = e.r_lo;
        h_pswf_table_gpu.data[base + 11] = e.dr_inv;
        }
    }

Scalar ESPForceCompute::evalPSWFKernel(Scalar x) const
    {
    if (x >= Scalar(1.0))
        return Scalar(0.0);

    if (x <= Scalar(0.0))
        x = Scalar(0.0);

    const Scalar s = Scalar(1.0) - x * x;
    const Scalar beta = Scalar(0.5) * Scalar(m_order) + std::max(Scalar(0.0), m_kappa * m_rcut);
    const Scalar poly = Scalar(1.0) + beta * s + Scalar(0.5) * beta * beta * s * s;
    return std::pow(s, Scalar(m_order)) * poly;
    }

Scalar ESPForceCompute::evalL_direct(Scalar r) const
    {
    const Scalar rr = std::max(r, Scalar(1.0e-8) * m_rcut);
    const Scalar x = std::min(rr / m_rcut, Scalar(1.0));

    auto kernel_u = [this](Scalar u) -> Scalar {
        return evalPSWFKernel(u);
    };

    const Scalar norm = integrate_trapezoid(kernel_u, Scalar(0.0), Scalar(1.0), 257);
    const Scalar cdf = integrate_trapezoid(kernel_u, Scalar(0.0), x, 257) / std::max(norm, Scalar(1.0e-12));

    return (Scalar(1.0) - cdf) / (Scalar(4.0) * ESP_PI * rr);
    }

Scalar ESPForceCompute::evalDL_direct(Scalar r) const
    {
    const Scalar h = std::max(Scalar(1.0e-5) * m_rcut, Scalar(1.0e-7));
    const Scalar rp = std::min(r + h, m_rcut);
    const Scalar rm = std::max(r - h, Scalar(1.0e-8) * m_rcut);

    const Scalar Lp = evalL_direct(rp);
    const Scalar Lm = evalL_direct(rm);

    return -(Lp - Lm) / std::max(rp - rm, Scalar(1.0e-20));
    }

Scalar ESPForceCompute::computePSWFSelfEnergyConst() const
    {
    auto integrand = [this](Scalar u) -> Scalar {
        const Scalar psi = evalPSWFKernel(u);
        return psi * psi;
    };

    const Scalar integral = integrate_trapezoid(integrand, Scalar(0.0), Scalar(1.0), 1025);
    return integral / std::max(m_rcut, Scalar(1.0e-20));
    }

uint3 ESPForceCompute::computeGhostCellNum() const
    {
#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        return make_uint3(m_radius, m_radius, m_radius);
        }
#endif
    return make_uint3(0, 0, 0);
    }

Scalar ESPForceCompute::rms(Scalar h, Scalar prd, Scalar natoms) const
    {
    const Scalar x = h / std::max(prd, Scalar(1.0e-12));
    return std::sqrt(std::max(natoms, Scalar(1.0))) * std::pow(x, Scalar(m_order + 1));
    }

void export_ESPForceCompute(pybind11::module& m)
    {
    pybind11::class_<ESPForceCompute, ForceCompute, std::shared_ptr<ESPForceCompute>>(m, "ESPForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<NeighborList>,
                            std::shared_ptr<ParticleGroup>>())
        .def("setParams", &ESPForceCompute::setParams)
        .def("getResolution", &ESPForceCompute::getResolution)
        .def("getOrder", &ESPForceCompute::getOrder)
        .def("getKappa", &ESPForceCompute::getKappa)
        .def("getRCut", &ESPForceCompute::getRCut)
        .def("getAlpha", &ESPForceCompute::getAlpha)
        .def("getQSum", &ESPForceCompute::getQSum)
        .def("getQ2Sum", &ESPForceCompute::getQ2Sum)
        .def("getTableSize", &ESPForceCompute::getTableSize)
        .def("invalidate", &ESPForceCompute::invalidate);
    }

} // end namespace md
} // end namespace hoomd