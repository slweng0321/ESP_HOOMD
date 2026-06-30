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
