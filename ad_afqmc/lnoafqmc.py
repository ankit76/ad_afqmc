#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Hong-Zhou Ye <hzyechem@gmail.com>
#


"""LNO-RCCSD and LNO-CCSD(T) (for both molecule and pbc w/ Gamma-point BZ sampling)

- Original publication by Kállay and co-workers:
    Rolik and Kállay, J. Chem. Phys. 135, 104111 (2011)

- Publication for this implementation by Ye and Berkelbach:
    Ye and Berkelbach, J. Chem. Theory Comput. 2024, 20, 20, 8948–8959
"""


import sys
import os
import numpy as np
from functools import reduce

from pyscf.lib import logger
from pyscf import lib

from pyscf.lno import LNO

_fdot = np.dot
fdot = lambda *args: reduce(_fdot, args)
einsum = lib.einsum


def impurity_solve(
    mcc,
    mo_coeff,
    uocc_loc,
    mo_occ,
    maskact,
    eris,
    log=None,
    verbose_imp=None,
    max_las_size_afqmc=None,
    kwargs_imp=None,
    frozen=None,
    n_blocks=50,
    n_walkers=10,
    seed=52,
    chol_cut=1e-4,
    maxError=1e-3,
    dt=0.01,
    nproc=1,
    tmpdir="./",
    output_file_name="afqmc_output.out",
    n_eql=2,
    n_ene_blocks=25,
    n_sr_blocks=2,
):
    r"""Solve impurity problem and calculate local correlation energy.

    Args:
        mo_coeff (np.ndarray):
            MOs where the impurity problem is solved.
        uocc_loc (np.ndarray):
            <i|I> where i is semi-canonical occ LNOs and I is LO.
        ccsd_t (bool):
            If True, CCSD(T) energy is calculated and returned as the third
            item (0 is returned otherwise).
        frozen (int or list; optional):
            Same syntax as `frozen` in MP2, CCSD, etc.

    Return:
        e_loc_corr_pt2, e_loc_corr_ccsd, e_loc_corr_ccsd_t:
            Local correlation energy at MP2, CCSD, and CCSD(T) level. Note that
            the CCSD(T) energy is 0 unless 'ccsd_t' is set to True.
    """
    mf = mcc._scf
    log = logger.new_logger(mcc if log is None else log)
    cput1 = (logger.process_clock(), logger.perf_counter())

    maskocc = mo_occ > 1e-10
    nmo = mo_occ.size

    orbfrzocc = mo_coeff[:, ~maskact & maskocc]
    orbactocc = mo_coeff[:, maskact & maskocc]
    orbactvir = mo_coeff[:, maskact & ~maskocc]
    orbfrzvir = mo_coeff[:, ~maskact & ~maskocc]
    nfrzocc, nactocc, nactvir, nfrzvir = [
        orb.shape[1] for orb in [orbfrzocc, orbactocc, orbactvir, orbfrzvir]
    ]
    nlo = uocc_loc.shape[1]
    nactmo = nactocc + nactvir
    log.debug(
        "    impsol:  %d LOs  %d/%d MOs  %d occ  %d vir",
        nlo,
        nactmo,
        nmo,
        nactocc,
        nactvir,
    )

    if nactocc == 0 or nactvir == 0:
        elcorr_pt2 = elcorr_afqmc = lib.tag_array(0.0, spin_comp=np.array((0.0, 0.0)))
        elcorr_afqmc = 0.0
    else:

        if nactmo > max_las_size_afqmc:
            log.warn(
                "Number of active space orbitals (%d) exceed "
                "`_max_las_size_afqmc` (%d). Impurity CCSD calculations "
                "will NOT be performed.",
                nactmo,
                max_las_size_afqmc,
            )
            elcorr_pt2 = elcorr_cc = lib.tag_array(0.0, spin_comp=np.array((0.0, 0.0)))
            elcorr_afqmc = 0.0
        else:
            # Impurity problem
            # Performing LNO-MP2
            imp_eris = mcc.ao2mo()
            if isinstance(imp_eris.ovov, np.ndarray):
                ovov = imp_eris.ovov
            else:
                ovov = imp_eris.ovov[()]
            oovv = ovov.reshape(nactocc, nactvir, nactocc, nactvir).transpose(
                0, 2, 1, 3
            )
            ovov = None
            cput1 = log.timer_debug1("imp sol - eri    ", *cput1)
            # MP2 fragment energy
            t1, t2 = mcc.init_amps(eris=imp_eris)[1:]
            cput1 = log.timer_debug1("imp sol - mp2 amp", *cput1)
            elcorr_pt2 = get_fragment_energy(oovv, t2, uocc_loc).real

            # Performing LNO-AFQMC
            prjlo = np.array([uocc_loc.flatten()])
            from ad_afqmc import lnoutils

            elcorr_afqmc, err_afqmc = lnoutils.run_afqmc_lno_mf(
                mf,
                mo_coeff=mo_coeff,
                norb_act=(nactocc + nactvir),
                nelec_act=nactocc * 2,
                norb_frozen=frozen,
                nwalk_per_proc=n_walkers,
                orbitalE=nactocc - 1,
                nblocks=n_blocks,
                seed=seed,
                chol_cut=chol_cut,
                maxError=maxError,
                dt=dt,
                nproc=nproc,
                prjlo=prjlo,
                tmpdir=tmpdir,
                output_file_name=output_file_name,
                n_eql=n_eql,
                n_ene_blocks=n_ene_blocks,
                n_sr_blocks=n_sr_blocks,
            )

            # elcorr_afqmc = 0.0

    frag_msg = "  ".join(
        [
            f"E_corr(MP2) = {elcorr_pt2:.15g}",
            f"E_corr(AFQMC) = {elcorr_afqmc:.15g} +/- {err_afqmc:.15g}",
        ]
    )

    return (elcorr_pt2, elcorr_afqmc, err_afqmc), frag_msg


def get_maskact(frozen, nmo):
    # Convert frozen to 0 bc PySCF solvers do not support frozen=None or empty list
    if frozen is None:
        frozen = 0
    elif isinstance(frozen, (list, tuple, np.ndarray)) and len(frozen) == 0:
        frozen = 0

    if isinstance(frozen, (int, np.integer)):
        maskact = np.hstack(
            [np.zeros(frozen, dtype=bool), np.ones(nmo - frozen, dtype=bool)]
        )
    elif isinstance(frozen, (list, tuple, np.ndarray)):
        maskact = np.array([i not in frozen for i in range(nmo)])
    else:
        raise RuntimeError

    return frozen, maskact


def get_fragment_energy(oovv, t2, uocc_loc):
    m = fdot(uocc_loc, uocc_loc.T.conj())
    # return einsum('ijab,kjab,ik->',t2,2*oovv-oovv.transpose(0,1,3,2),m)
    ed = einsum("ijab,kjab,ik->", t2, oovv, m) * 2
    ex = -einsum("ijab,kjba,ik->", t2, oovv, m)
    ed = ed.real
    ex = ex.real
    ess = ed * 0.5 + ex
    eos = ed * 0.5
    return lib.tag_array(ess + eos, spin_comp=np.array((ess, eos)))


class LNOAFQMC(LNO):
    """Use the following _max_las_size arguments to avoid calculations that have no
    hope of finishing. This may ease scanning thresholds.
    """

    _max_las_size_afqmc = 300

    def __init__(
        self, mf, lo_coeff, frag_lolist, lno_type=None, lno_thresh=None, frozen=None
    ):

        super().__init__(
            mf,
            lo_coeff,
            frag_lolist,
            lno_type=lno_type,
            lno_thresh=lno_thresh,
            frozen=frozen,
        )

        self.efrag_afqmc = None
        self.efrag_pt2 = None
        self.errfrag_afqmc = None

        # args for impurity solver
        self.kwargs_imp = None
        self.verbose_imp = 2  # ERROR and WARNING

        # Not inputs
        self._h1e = None
        self._vhf = None

        # AFQMC options
        self._max_las_size_afqmc = LNOAFQMC._max_las_size_afqmc
        self.n_walkers = 20
        self.n_blocks = 200
        self.seed = np.random.randint(1, 1000000)
        self.chol_cut = 1e-4
        self.maxError = 1e-4
        self.dt = 0.01
        self.nproc = 1
        self.tmpdir = "./"
        self.output_file_name = "afqmc_output.out"
        self.n_eql = 2
        self.n_ene_blocks = 25
        self.n_sr_blocks = 2
        os.system(f"rm -f " + self.output_file_name)

    @property
    def h1e(self):
        if self._h1e is None:
            self._h1e = self._scf.get_hcore()
        return self._h1e

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose=verbose)
        log = logger.new_logger(self, verbose)
        log.info("_max_las_size_afqmc = %s", self._max_las_size_afqmc)
        return self

    def impurity_solve(self, mf, mo_coeff, uocc_loc, eris, frozen=None, log=None):
        if log is None:
            log = logger.new_logger(self)
        mo_occ = self.mo_occ
        frozen, maskact = get_maskact(frozen, mo_occ.size)
        from pyscf.lno.lnoccsd import CCSD

        mcc = CCSD(mf, mo_coeff=mo_coeff, frozen=frozen).set(verbose=self.verbose_imp)
        mcc._s1e = self._s1e
        mcc._h1e = self._h1e
        mcc._vhf = self._vhf
        if self.kwargs_imp is not None:
            mcc = mcc.set(**self.kwargs_imp)

        return impurity_solve(
            mcc,
            mo_coeff,
            uocc_loc,
            mo_occ,
            maskact,
            eris,
            log=log,
            verbose_imp=self.verbose_imp,
            max_las_size_afqmc=self._max_las_size_afqmc,
            kwargs_imp=self.kwargs_imp,  # Need to check if this works
            frozen=frozen,
            n_blocks=self.n_blocks,
            n_walkers=self.n_walkers,
            seed=self.seed,
            chol_cut=self.chol_cut,
            maxError=self.maxError,
            dt=self.dt,
            nproc=self.nproc,
            tmpdir=self.tmpdir,
            output_file_name=self.output_file_name,
            n_eql=self.n_eql,
            n_ene_blocks=self.n_ene_blocks,
            n_sr_blocks=self.n_sr_blocks,
        )

    def _post_proc(self, frag_res, frag_wghtlist):
        """Post processing results returned by `impurity_solve` collected in `frag_res`."""
        # TODO: add spin-component for CCSD(T)
        nfrag = len(frag_res)
        efrag_pt2 = np.zeros(nfrag)
        efrag_afqmc = np.zeros(nfrag)
        errfrag_afqmc = np.zeros(nfrag)
        for i in range(nfrag):
            (
                ept2,
                eafqmc,
                err_afqmc,
            ) = frag_res[i]
            efrag_pt2[i] = float(ept2)
            efrag_afqmc[i] = float(eafqmc)
            errfrag_afqmc[i] = float(err_afqmc)
        self.efrag_pt2 = efrag_pt2 * frag_wghtlist
        self.efrag_afqmc = efrag_afqmc * frag_wghtlist
        self.errfrag_afqmc = errfrag_afqmc * frag_wghtlist

    def _finalize(self):
        r"""Hook for dumping results and clearing up the object."""
        logger.note(
            self,
            "E(%s) = %.15g  E_corr = %.15g",
            "LNOMP2",
            self.e_tot_pt2,
            self.e_corr_pt2,
        )
        sig_dec_corr = int(abs(np.floor(np.log10(self.afqmc_error_ecorr))))
        sig_err_corr = np.around(
            np.round(self.afqmc_error_ecorr * 10**sig_dec_corr) * 10 ** (-sig_dec_corr),
            sig_dec_corr,
        )
        sig_e_corr = np.around(self.e_corr, sig_dec_corr)
        sig_e_tot = np.around(self.e_tot, sig_dec_corr)
        logger.note(
            self,
            "E(%s) = %.15g  E_corr = %.15g +/- %.15g",
            "LNOAFQMC",
            sig_e_tot,
            sig_e_corr,
            sig_err_corr,
        )

        return self

    @property
    def e_tot_scf(self):
        return self._scf.e_tot

    @property
    def e_corr(self):
        return self.e_corr_afqmc

    @property
    def e_tot(self):
        return self.e_corr + self.e_tot_scf

    @property
    def e_corr_afqmc(self):
        e_corr = np.sum(self.efrag_afqmc)
        return e_corr

    @property
    def e_corr_pt2(self):
        e_corr = np.sum(self.efrag_pt2)
        return e_corr

    @property
    def e_tot_afqmc(self):
        return self.e_corr_afqmc + self.e_tot_scf

    @property
    def e_tot_pt2(self):
        return self.e_corr_pt2 + self.e_tot_scf

    @property
    def afqmc_error_ecorr(self):
        return np.sqrt(np.sum(self.errfrag_afqmc**2))

    def e_corr_pt2corrected(self, ept2):
        return self.e_corr - self.e_corr_pt2 + ept2

    def e_tot_pt2corrected(self, ept2):
        return self.e_tot_scf + self.e_corr_pt2corrected(ept2)

    def e_corr_afqmc_pt2corrected(self, ept2):
        return self.e_corr_afqmc - self.e_corr_pt2 + ept2

    def e_tot_afqmc_pt2corrected(self, ept2):
        return self.e_tot_scf + self.e_corr_afqmc_pt2corrected(ept2)


def fock_from_mo(mymf, s1e=None, force_exxdiv_none=True):
    if s1e is None:
        s1e = mymf.get_ovlp()
    mo0 = np.dot(s1e, mymf.mo_coeff)
    moe0 = mymf.mo_energy
    nocc0 = np.count_nonzero(mymf.mo_occ)
    if force_exxdiv_none:
        if hasattr(mymf, "exxdiv") and mymf.exxdiv == "ewald":  # remove madelung
            from pyscf.pbc.cc.ccsd import _adjust_occ
            from pyscf.pbc import tools

            madelung = tools.madelung(mymf.cell, mymf.kpt)
            moe0 = _adjust_occ(moe0, nocc0, madelung)
    fock = np.dot(mo0 * moe0, mo0.T.conj())
    return fock


if __name__ == "__main__":
    from pyscf import gto, scf, mp, cc, lo
    from pyscf.cc.ccsd_t import kernel as CCSD_T
    from pyscf.data.elements import chemcore
    from ad_afqmc import lnoutils

    log = logger.Logger(sys.stdout, 6)

    # S22-2: water dimer
    atom = """
    O   -1.485163346097   -0.114724564047    0.000000000000
    H   -1.868415346097    0.762298435953    0.000000000000
    H   -0.533833346097    0.040507435953    0.000000000000
    O    1.416468653903    0.111264435953    0.000000000000
    H    1.746241653903   -0.373945564047   -0.758561000000
    H    1.746241653903   -0.373945564047    0.758561000000
    """
    basis = "cc-pvdz"

    mol = gto.M(atom=atom, basis=basis)
    mol.verbose = 4
    frozen = chemcore(mol)

    mf = scf.RHF(mol).density_fit()
    mf.kernel()

    # canonical
    mmp = mp.MP2(mf, frozen=frozen)
    mmp.kernel()
    efull_mp2 = mmp.e_corr

    lo_coeff, frag_lolist = lnoutils.prep_local_orbitals(mf, frozen=frozen)
    # LNO-CCSD(T) calculation: here we scan over a list of thresholds
    mcc = LNOAFQMC(mf, lo_coeff, frag_lolist, frozen=frozen).set(verbose=5)
    mcc.n_blocks = 50
    mcc.n_walkers = 20
    mcc.nproc = 2
    mcc.maxError = 1e-3  # /np.sqrt(len(frag_lolist))
    mcc.nproc = 1

    gamma = 10  # thresh_occ / thresh_vir
    threshs = np.asarray([1e-5])
    elno_afqmc_uncorr = np.zeros_like(threshs)
    lno_stocherror_afqmc = np.zeros_like(threshs)
    elno_mp2 = np.zeros_like(threshs)
    for i, thresh in enumerate(threshs):
        mcc.lno_thresh = [thresh * gamma, thresh]
        mcc.kernel()
        lno_stocherror_afqmc[i] = mcc.afqmc_error_ecorr
        elno_afqmc_uncorr[i] = mcc.e_corr_afqmc
        elno_mp2[i] = mcc.e_corr_pt2

    elno_afqmc = elno_afqmc_uncorr - elno_mp2 + efull_mp2

    log.info("")
    for i, thresh in enumerate(threshs):
        e0 = elno_afqmc_uncorr[i]
        e1 = elno_afqmc[i]
        err = lno_stocherror_afqmc[i]
        log.info(
            "thresh = %.3e  E_corr(LNO-AFQMC) = %.15g +/- %.15g E_corr(LNO-AFQMC+∆PT2) = %.15g +/- %.15g",
            thresh,
            e0,
            err,
            e1,
            err,
        )
