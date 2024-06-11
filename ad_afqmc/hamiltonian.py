import os

os.environ["XLA_FLAGS"] = (
    "--xla_force_host_platform_device_count=1 --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "True"
from dataclasses import dataclass
from functools import partial

# os.environ['JAX_DISABLE_JIT'] = 'True'
import jax.numpy as jnp
import jax.scipy as jsp
from jax import jit, vmap

print = partial(print, flush=True)

# TODO: move rotation member functions to wavefunctions, prop_ham to propagators, and use wrappers around them as member functions of one hamiltonian class


@dataclass
class hamiltonian:
    norb: int  # number of spatial orbitals
    nelec: int  # number of electrons of each spin, so nelec = total_nelec // 2
    nchol: int

    @partial(jit, static_argnums=(0,))
    def rot_orbs(self, ham, mo_coeff):
        ham["h1"] = mo_coeff.T.dot(ham["h1"]).dot(mo_coeff)
        ham["chol"] = jnp.einsum(
            "gij,jp->gip", ham["chol"].reshape(-1, self.norb, self.norb), mo_coeff
        )
        ham["chol"] = jnp.einsum("qi,gip->gqp", mo_coeff.T, ham["chol"]).reshape(
            -1, self.norb * self.norb
        )
        return ham

    @partial(jit, static_argnums=(0,))
    def rot_ham(self, ham, wave_data=None):
        ham["h1"] = (ham["h1"] + ham["h1"].T) / 2.0
        ham["rot_h1"] = ham["h1"][: self.nelec, :].copy()
        ham["rot_chol"] = (
            ham["chol"].reshape(-1, self.norb, self.norb)[:, : self.nelec, :].copy()
        )
        return ham

    @partial(jit, static_argnums=(0, 3))
    def prop_ham(self, ham, dt, _trial, wave_data=None):
        ham["mf_shifts"] = 2.0j * vmap(
            lambda x: jnp.sum(jnp.diag(x.reshape(self.norb, self.norb))[: self.nelec])
        )(ham["chol"])
        ham["mf_shifts_fp"] = ham["mf_shifts"] / 2.0 / self.nelec
        ham["h0_prop"] = -ham["h0"] - jnp.sum(ham["mf_shifts"] ** 2) / 2.0
        ham["h0_prop_fp"] = [
            (ham["h0_prop"] + ham["ene0"]) / self.nelec,
            (ham["h0_prop"] + ham["ene0"]) / self.nelec,
        ]
        v0 = 0.5 * jnp.einsum(
            "gik,gjk->ij",
            ham["chol"].reshape(-1, self.norb, self.norb),
            ham["chol"].reshape(-1, self.norb, self.norb),
            optimize="optimal",
        )
        h1_mod = ham["h1"] - v0
        h1_mod = h1_mod - jnp.real(
            1.0j
            * jnp.einsum(
                "g,gik->ik",
                ham["mf_shifts"],
                ham["chol"].reshape(-1, self.norb, self.norb),
            )
        )
        ham["exp_h1"] = jsp.linalg.expm(-dt * h1_mod / 2.0)
        return ham

    def __hash__(self):
        return hash((self.norb, self.nelec, self.nchol))


@dataclass
class hamiltonian_uhf:
    norb: int  # number of spatial orbitals
    nelec: tuple
    nchol: int

    @partial(jit, static_argnums=(0,))
    def rot_orbs(self, ham, wave_data):
        return ham

    @partial(jit, static_argnums=(0,))
    def rot_ham(self, ham, wave_data):
        ham["h1"] = ham["h1"].at[0].set((ham["h1"][0] + ham["h1"][0].T) / 2.0)
        ham["h1"] = ham["h1"].at[1].set((ham["h1"][1] + ham["h1"][1].T) / 2.0)
        trial = [wave_data[0][:, : self.nelec[0]], wave_data[1][:, : self.nelec[1]]]
        ham["rot_h1"] = [trial[0].T @ ham["h1"][0], trial[1].T @ ham["h1"][1]]
        ham["rot_chol"] = [
            jnp.einsum(
                "pi,gij->gpj", trial[0].T, ham["chol"].reshape(-1, self.norb, self.norb)
            ),
            jnp.einsum(
                "pi,gij->gpj", trial[1].T, ham["chol"].reshape(-1, self.norb, self.norb)
            ),
        ]
        return ham

    @partial(jit, static_argnums=(0, 3))
    def prop_ham(self, ham, dt, _trial, wave_data):
        trial = [wave_data[0][:, : self.nelec[0]], wave_data[1][:, : self.nelec[1]]]
        dm = trial[0] @ trial[0].T + trial[1] @ trial[1].T
        ham["mf_shifts"] = 1.0j * vmap(
            lambda x: jnp.sum(x.reshape(self.norb, self.norb) * dm)
        )(ham["chol"])
        ham["mf_shifts_fp"] = jnp.stack(
            (
                ham["mf_shifts"] / self.nelec[0] / 2.0,
                ham["mf_shifts"] / self.nelec[1] / 2.0,
            )
        )
        ham["h0_prop"] = -ham["h0"] - jnp.sum(ham["mf_shifts"] ** 2) / 2.0
        ham["h0_prop_fp"] = jnp.stack(
            (
                (ham["h0_prop"] + ham["ene0"]) / self.nelec[0] / 2.0,
                (ham["h0_prop"] + ham["ene0"]) / self.nelec[1] / 2.0,
            )
        )
        v0 = 0.5 * jnp.einsum(
            "gik,gjk->ij",
            ham["chol"].reshape(-1, self.norb, self.norb),
            ham["chol"].reshape(-1, self.norb, self.norb),
            optimize="optimal",
        )
        v1 = jnp.real(
            1.0j
            * jnp.einsum(
                "g,gik->ik",
                ham["mf_shifts"],
                ham["chol"].reshape(-1, self.norb, self.norb),
            )
        )
        h1_mod = ham["h1"] - jnp.array([v0 + v1, v0 + v1])
        ham["exp_h1"] = jnp.array(
            [
                jsp.linalg.expm(-dt * h1_mod[0] / 2.0),
                jsp.linalg.expm(-dt * h1_mod[1] / 2.0),
            ]
        )
        return ham

    def __hash__(self):
        return hash((self.norb, self.nelec, self.nchol))


@dataclass
class hamiltonian_ghf:
    norb: int  # number of spatial orbitals
    nelec: tuple
    nchol: int

    @partial(jit, static_argnums=(0,))
    def rot_orbs(self, ham, wave_data):
        return ham

    @partial(jit, static_argnums=(0,))
    def rot_ham(self, ham, wave_data):
        ham["h1"] = ham["h1"].at[0].set((ham["h1"][0] + ham["h1"][0].T) / 2.0)
        ham["h1"] = ham["h1"].at[1].set((ham["h1"][1] + ham["h1"][1].T) / 2.0)
        trial = wave_data[:, : self.nelec[0] + self.nelec[1]]
        ham["rot_h1"] = trial.T @ jnp.block(
            [
                [ham["h1"][0], jnp.zeros_like(ham["h1"][1])],
                [jnp.zeros_like(ham["h1"][0]), ham["h1"][1]],
            ]
        )
        ham["rot_chol"] = vmap(
            lambda x: jnp.hstack(
                [trial.T[:, : self.norb] @ x, trial.T[:, self.norb :] @ x]
            ),
            in_axes=(0),
        )(ham["chol"].reshape(-1, self.norb, self.norb))
        return ham

    @partial(jit, static_argnums=(0, 3))
    def prop_ham(self, ham, dt, _trial, wave_data):
        trial = wave_data[:, : self.nelec[0] + self.nelec[1]]
        dm_ghf = trial @ trial.T
        dm = dm_ghf[: self.norb, : self.norb] + dm_ghf[self.norb :, self.norb :]
        ham["mf_shifts"] = 1.0j * vmap(
            lambda x: jnp.sum(x.reshape(self.norb, self.norb) * dm)
        )(ham["chol"])
        ham["mf_shifts_fp"] = jnp.stack(
            (
                ham["mf_shifts"] / self.nelec[0] / 2.0,
                ham["mf_shifts"] / self.nelec[1] / 2.0,
            )
        )
        ham["h0_prop"] = -ham["h0"] - jnp.sum(ham["mf_shifts"] ** 2) / 2.0
        ham["h0_prop_fp"] = jnp.stack(
            (
                (ham["h0_prop"] + ham["ene0"]) / self.nelec[0] / 2.0,
                (ham["h0_prop"] + ham["ene0"]) / self.nelec[1] / 2.0,
            )
        )
        v0 = 0.5 * jnp.einsum(
            "gik,gjk->ij",
            ham["chol"].reshape(-1, self.norb, self.norb),
            ham["chol"].reshape(-1, self.norb, self.norb),
            optimize="optimal",
        )
        v1 = jnp.real(
            1.0j
            * jnp.einsum(
                "g,gik->ik",
                ham["mf_shifts"],
                ham["chol"].reshape(-1, self.norb, self.norb),
            )
        )
        h1_mod = ham["h1"] - jnp.array([v0 + v1, v0 + v1])
        ham["exp_h1"] = jnp.array(
            [
                jsp.linalg.expm(-dt * h1_mod[0] / 2.0),
                jsp.linalg.expm(-dt * h1_mod[1] / 2.0),
            ]
        )
        return ham

    def __hash__(self):
        return hash((self.norb, self.nelec, self.nchol))


@dataclass
class hamiltonian_noci:
    norb: int  # number of spatial orbitals
    nelec: tuple
    nchol: int

    @partial(jit, static_argnums=(0,))
    def rot_orbs(self, ham, wave_data):
        return ham

    @partial(jit, static_argnums=(0,))
    def rot_orbs_single_det(self, ham, trial_up, trial_dn):
        rot_h1 = [
            trial_up[:, : self.nelec[0]].T @ ham["h1"][0],
            trial_dn[:, : self.nelec[1]].T @ ham["h1"][1],
        ]
        rot_chol = [
            jnp.einsum(
                "pi,gij->gpj",
                trial_up[:, : self.nelec[0]].T,
                ham["chol"].reshape(-1, self.norb, self.norb),
            ),
            jnp.einsum(
                "pi,gij->gpj",
                trial_dn[:, : self.nelec[1]].T,
                ham["chol"].reshape(-1, self.norb, self.norb),
            ),
        ]
        return rot_h1, rot_chol

    @partial(jit, static_argnums=(0,))
    def rot_ham(self, ham, wave_data):
        ham["h1"] = ham["h1"].at[0].set((ham["h1"][0] + ham["h1"][0].T) / 2.0)
        ham["h1"] = ham["h1"].at[1].set((ham["h1"][1] + ham["h1"][1].T) / 2.0)
        trial = wave_data[1]
        ham["rot_h1"], ham["rot_chol"] = vmap(
            self.rot_orbs_single_det, in_axes=(None, 0, 0)
        )(ham, trial[0], trial[1])
        return ham

    @partial(jit, static_argnums=(0, 3))
    def prop_ham(self, ham, dt, trial, wave_data):
        dm = trial.get_rdm1(wave_data)
        ham["mf_shifts"] = 1.0j * vmap(
            lambda x: jnp.sum(x.reshape(self.norb, self.norb) * dm)
        )(ham["chol"])
        ham["mf_shifts_fp"] = jnp.stack(
            (
                ham["mf_shifts"] / self.nelec[0] / 2.0,
                ham["mf_shifts"] / self.nelec[1] / 2.0,
            )
        )
        ham["h0_prop"] = -ham["h0"] - jnp.sum(ham["mf_shifts"] ** 2) / 2.0
        ham["h0_prop_fp"] = jnp.stack(
            (
                (ham["h0_prop"] + ham["ene0"]) / self.nelec[0] / 2.0,
                (ham["h0_prop"] + ham["ene0"]) / self.nelec[1] / 2.0,
            )
        )
        v0 = 0.5 * jnp.einsum(
            "gik,gjk->ij",
            ham["chol"].reshape(-1, self.norb, self.norb),
            ham["chol"].reshape(-1, self.norb, self.norb),
            optimize="optimal",
        )
        v1 = jnp.real(
            1.0j
            * jnp.einsum(
                "g,gik->ik",
                ham["mf_shifts"],
                ham["chol"].reshape(-1, self.norb, self.norb),
            )
        )
        h1_mod = ham["h1"] - jnp.array([v0 + v1, v0 + v1])
        ham["exp_h1"] = jnp.array(
            [
                jsp.linalg.expm(-dt * h1_mod[0] / 2.0),
                jsp.linalg.expm(-dt * h1_mod[1] / 2.0),
            ]
        )
        return ham

    def __hash__(self):
        return hash((self.norb, self.nelec, self.nchol))
