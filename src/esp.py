# Copyright (c) 2025 Shih-Lun Weng.
# Part of RxMC, released under the BSD 3-Clause License.

"""Custom PPPM force wrappers used for RxMC."""

import math

import hoomd
from hoomd.md.force import Force

from .._core import _cpp, require_cpp_extension


def make_pppm_coulomb_forces(nlist, resolution, order, r_cut, alpha=0):
    """Build the real/reciprocal Coulomb force pair used by RxMC.

    Args:
        nlist: HOOMD neighbor list instance.
        resolution: FFT mesh resolution ``(Nx, Ny, Nz)``.
        order: PPPM interpolation order.
        r_cut: Real-space cutoff.
        alpha: Ewald damping parameter.

    Returns:
        tuple: ``(real_space_force, reciprocal_space_force)``.
    """
    real_space_force = hoomd.md.pair.Ewald(nlist)
    real_space_force.params.default = dict(kappa=0, alpha=0)
    real_space_force.r_cut.default = r_cut

    reciprocal_space_force = Coulomb(
        nlist=nlist,
        resolution=resolution,
        order=order,
        r_cut=r_cut,
        alpha=alpha,
        pair_force=real_space_force,
    )

    return real_space_force, reciprocal_space_force


class Coulomb(Force):
    """Reciprocal-space PPPM force with a lighter charge-cache refresh path."""

    def __init__(self, nlist, resolution, order, r_cut, alpha, pair_force):
        """Initialize reciprocal-space PPPM force wrapper.

        Args:
            nlist: HOOMD neighbor list instance.
            resolution: FFT mesh resolution ``(Nx, Ny, Nz)``.
            order: PPPM interpolation order.
            r_cut: Real-space cutoff.
            alpha: Ewald damping parameter.
            pair_force: Real-space Ewald pair force coupled to this object.
        """
        super().__init__()
        self._nlist = hoomd.data.typeconverter.OnlyTypes(hoomd.md.nlist.NeighborList)(
            nlist
        )
        self._param_dict.update(
            hoomd.data.parameterdicts.ParameterDict(
                resolution=(int, int, int),
                order=int,
                r_cut=float,
                alpha=float,
            )
        )
        self.resolution = resolution
        self.order = order
        self.r_cut = r_cut
        self.alpha = alpha
        self._pair_force = pair_force

    def _attach_hook(self) -> None:
        self.nlist._attach(self._simulation)

        if isinstance(self._simulation.device, hoomd.device.CPU):
            cls = _cpp.CustomPPPMForceCompute
        else:
            cls = _cpp.CustomPPPMForceComputeGPU

        nx, ny, nz = self.resolution
        order = self.order
        rcut = self.r_cut
        alpha = self.alpha

        group = self._simulation.state._get_group(hoomd.filter.All())
        self._cpp_obj = cls(
            self._simulation.state._cpp_sys_def,
            self.nlist._cpp_obj,
            group,
        )

        q2 = self._cpp_obj.getQ2Sum()
        num_particles = self._simulation.state.N_particles
        box = self._simulation.state.box
        lx = box.Lx
        ly = box.Ly
        lz = box.Lz

        hx = lx / nx
        hy = ly / ny
        hz = lz / nz

        gew1 = 0.0
        kappa = gew1
        f = _diffpr(hx, hy, hz, lx, ly, lz, num_particles, order, kappa, q2, rcut)
        hmin = min(hx, hy, hz)
        gew2 = 10.0 / hmin
        kappa = gew2
        fmid = _diffpr(hx, hy, hz, lx, ly, lz, num_particles, order, kappa, q2, rcut)

        if f * fmid >= 0.0:
            raise RuntimeError(
                "Cannot compute custom PPPM Coulomb forces: f*fmid >= 0.0"
            )

        if f < 0.0:
            dgew = gew2 - gew1
            rtb = gew1
        else:
            dgew = gew1 - gew2
            rtb = gew2

        ncount = 0
        while math.fabs(dgew) > 0.00001 and fmid != 0.0:
            dgew *= 0.5
            kappa = rtb + dgew
            fmid = _diffpr(
                hx, hy, hz, lx, ly, lz, num_particles, order, kappa, q2, rcut
            )
            if fmid <= 0.0:
                rtb = kappa
            ncount += 1
            if ncount > 10000.0:
                raise RuntimeError("Cannot compute custom PPPM kappa: not converging")

        particle_types = self._simulation.state.particle_types
        for type_a in particle_types:
            for type_b in particle_types:
                self._pair_force.params[(type_a, type_b)] = dict(
                    kappa=kappa, alpha=alpha
                )
                self._pair_force.r_cut[(type_a, type_b)] = rcut

        self._cpp_obj.setParams(nx, ny, nz, order, kappa, rcut, alpha)

    @property
    def nlist(self):
        """hoomd.md.nlist.NeighborList: Neighbor list used by this force."""
        return self._nlist


def _rms(h, prd, natoms, order, kappa, q2):
    acons = {
        1: [2.0 / 3.0],
        2: [1.0 / 50.0, 5.0 / 294.0],
        3: [1.0 / 588.0, 7.0 / 1440.0, 21.0 / 3872.0],
        4: [1.0 / 4320.0, 3.0 / 1936.0, 7601.0 / 2271360.0, 143.0 / 28800.0],
        5: [
            1.0 / 23232.0,
            7601.0 / 13628160.0,
            143.0 / 69120.0,
            517231.0 / 106536960.0,
            106640677.0 / 11737571328.0,
        ],
        6: [
            691.0 / 68140800.0,
            13.0 / 57600.0,
            47021.0 / 35512320.0,
            9694607.0 / 2095994880.0,
            733191589.0 / 59609088000.0,
            326190917.0 / 11700633600.0,
        ],
        7: [
            1.0 / 345600.0,
            3617.0 / 35512320.0,
            745739.0 / 838397952.0,
            56399353.0 / 12773376000.0,
            25091609.0 / 1560084480.0,
            1755948832039.0 / 36229939200000.0,
            4887769399.0 / 37838389248.0,
        ],
    }
    series = 0.0
    for exponent, coeff in enumerate(acons[order]):
        series += coeff * pow(h * kappa, 2.0 * exponent)
    return (
        q2
        * pow(h * kappa, order)
        * math.sqrt(kappa * prd * math.sqrt(2.0 * math.pi) * series / natoms)
        / (prd * prd)
    )


def _diffpr(hx, hy, hz, lx, ly, lz, natoms, order, kappa, q2, rcut):
    lprx = _rms(hx, lx, natoms, order, kappa, q2)
    lpry = _rms(hy, ly, natoms, order, kappa, q2)
    lprz = _rms(hz, lz, natoms, order, kappa, q2)
    lpr = math.sqrt(lprx * lprx + lpry * lpry + lprz * lprz) / math.sqrt(3.0)
    spr = (
        2.0
        * q2
        * math.exp(-kappa * kappa * rcut * rcut)
        / math.sqrt(natoms * rcut * lx * ly * lz)
    )
    value = lpr - spr
    return value
