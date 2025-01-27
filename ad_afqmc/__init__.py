from typing import Optional, Union

from pyscf import scf
from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD

from ad_afqmc import afqmc


def AFQMC(mf_or_cc: Union[scf.uhf.UHF, scf.rhf.RHF, scf.rohf.ROHF, CCSD, UCCSD]):
    return afqmc.AFQMC(mf_or_cc)


AFQMC.__doc__ = afqmc.AFQMC.__doc__
