import sys
import numpy as np

from pyscf import __config__, scf
from pyscf.cc.ccsd import CCSD
from pyscf.cc.uccsd import UCCSD

class Options:

    def __init__(
        self,
        mf_or_cc = None,
        dt: float = 0.01, #0.005,
        n_walkers: int = 50,
        n_prop_steps: int = 50,
        n_ene_blocks: int = 50, #1,
        n_sr_blocks: int = 1, #5,
        n_blocks: int = 50, #200,
        n_ene_blocks_eql: int = 5, #1,
        n_sr_blocks_eql: int = 10, #5,
        seed: int = np.random.randint(1, int(1e6)),
        n_eql: int = 1, #20,
        ad_mode = None,
        orbital_rotation: bool = True,
        do_sr: bool = True,
        walker_type: str = "restricted",
        symmetry: bool = False,
        save_walkers: bool = False,
        trial = None,
        ene0: float = 0.0,
        free_projection: bool = False,
        n_batch: int = 1,
        vhs_mixed_precision: bool = False,
        trial_mixed_precision: bool = False,
        memory_mode: str = "low",
        verbose: int = 3,
        ):

        if trial is None:
            if isinstance(mf_or_cc, scf.uhf.UHF) or isinstance(mf_or_cc, scf.rohf.ROHF):
                trial = "uhf"
            elif isinstance(mf_or_cc, scf.rhf.RHF):
                trial = "rhf"
            elif isinstance(mf_or_cc, UCCSD):
                trial = "ucisd"
            elif isinstance(mf_or_cc, CCSD):
                trial = "cisd"
        self.trial = trial
        assert self.trial in [None, "rhf", "uhf", "noci", "cisd", "ucisd"]

        # Set default values for options
        self.dt = dt
        self.n_walkers = n_walkers
        self.n_prop_steps = n_prop_steps
        self.n_ene_blocks = n_ene_blocks
        self.n_sr_blocks = n_sr_blocks
        self.n_blocks = n_blocks
        self.n_ene_blocks_eql = n_ene_blocks_eql
        self.n_sr_blocks_eql = n_sr_blocks_eql
        self.seed = seed
        self.n_eql = n_eql

        # AD mode options
        self.ad_mode = ad_mode
        assert self.ad_mode in [None, "forward", "reverse", "2rdm"]

        # Wavefunction and algorithm options
        self.orbital_rotation = orbital_rotation
        self.do_sr = do_sr
        self.walker_type = walker_type

        # Handle backwards compatibility for walker types
        if self.walker_type == "rhf":
            self.walker_type = "restricted"
        elif self.walker_type == "uhf":
            self.walker_type = "unrestricted"
        assert self.walker_type in ["restricted", "unrestricted"]
        
        self.symmetry = symmetry
        self.save_walkers = save_walkers

        self.ene0 = ene0
        self.free_projection = free_projection

        # performance and memory options
        self.n_batch = n_batch
        self.vhs_mixed_precision = vhs_mixed_precision
        self.trial_mixed_precision = trial_mixed_precision
        self.memory_mode = memory_mode
        self.verbose = verbose

    def get_keys():
        keys = [
            "dt",
            "n_walkers",
            "n_prop_steps",
            "n_ene_blocks",
            "n_sr_blocks",
            "n_blocks",
            "n_ene_blocks_eql",
            "n_sr_blocks_eql",
            "seed",
            "n_eql",
            "ad_mode",
            "orbital_rotation",
            "do_sr",
            "walker_type",
            "symmetry",
            "save_walkers",
            "trial",
            "ene0",
            "free_projection",
            "n_batch",
            "vhs_mixed_precision",
            "trial_mixed_precision",
            "memory_mode",
            "verbose",
        ]

        return keys


    def from_dict(options: dict):
        assert type(options) == dict
        filtered_dict = {key: val for key, val in options.items() if key in Options.get_keys()}

        return Options(**options)

    def to_dict(self):
       return {key: val for key, val in self.__dict__.items()}
