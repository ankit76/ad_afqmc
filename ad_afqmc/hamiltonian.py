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
    def rot_orbs(self, ham_data, mo_coeff):
        ham_data["h1"] = mo_coeff.T.dot(ham_data["h1"]).dot(mo_coeff)
        ham_data["chol"] = jnp.einsum(
            "gij,jp->gip", ham_data["chol"].reshape(-1, self.norb, self.norb), mo_coeff
        )
        ham_data["chol"] = jnp.einsum(
            "qi,gip->gqp", mo_coeff.T, ham_data["chol"]
        ).reshape(-1, self.norb * self.norb)
        return ham_data

    @partial(jit, static_argnums=(0,))
    def rot_ham(self, ham_data, wave_data=None):
        ham_data["h1"] = (ham_data["h1"] + ham_data["h1"].T) / 2.0
        ham_data["rot_h1"] = ham_data["h1"][: self.nelec, :].copy()
        ham_data["rot_chol"] = (
            ham_data["chol"]
            .reshape(-1, self.norb, self.norb)[:, : self.nelec, :]
            .copy()
        )
        return ham_data

    @partial(jit, static_argnums=(0, 3))
    def prop_ham(self, ham_data, dt, _trial, wave_data=None):
        ham_data["mf_shifts"] = 2.0j * vmap(
            lambda x: jnp.sum(jnp.diag(x.reshape(self.norb, self.norb))[: self.nelec])
        )(ham_data["chol"])
        ham_data["mf_shifts_fp"] = ham_data["mf_shifts"] / 2.0 / self.nelec
        ham_data["h0_prop"] = (
            -ham_data["h0"] - jnp.sum(ham_data["mf_shifts"] ** 2) / 2.0
        )
        ham_data["h0_prop_fp"] = [
            (ham_data["h0_prop"] + ham_data["ene0"]) / self.nelec,
            (ham_data["h0_prop"] + ham_data["ene0"]) / self.nelec,
        ]
        v0 = 0.5 * jnp.einsum(
            "gik,gjk->ij",
            ham_data["chol"].reshape(-1, self.norb, self.norb),
            ham_data["chol"].reshape(-1, self.norb, self.norb),
            optimize="optimal",
        )
        h1_mod = ham_data["h1"] - v0
        h1_mod = h1_mod - jnp.real(
            1.0j
            * jnp.einsum(
                "g,gik->ik",
                ham_data["mf_shifts"],
                ham_data["chol"].reshape(-1, self.norb, self.norb),
            )
        )
        ham_data["exp_h1"] = jsp.linalg.expm(-dt * h1_mod / 2.0)
        return ham_data

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
    def rot_ham(self, ham_data, wave_data):
        ham_data["h1"] = (
            ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
        )
        ham_data["h1"] = (
            ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
        )
        trial = [wave_data[0][:, : self.nelec[0]], wave_data[1][:, : self.nelec[1]]]
        ham_data["rot_h1"] = [
            trial[0].T @ ham_data["h1"][0],
            trial[1].T @ ham_data["h1"][1],
        ]
        ham_data["rot_chol"] = [
            jnp.einsum(
                "pi,gij->gpj",
                trial[0].T,
                ham_data["chol"].reshape(-1, self.norb, self.norb),
            ),
            jnp.einsum(
                "pi,gij->gpj",
                trial[1].T,
                ham_data["chol"].reshape(-1, self.norb, self.norb),
            ),
        ]
        return ham_data

    @partial(jit, static_argnums=(0, 3))
    def prop_ham(self, ham_data, dt, _trial, wave_data):
        trial = [wave_data[0][:, : self.nelec[0]], wave_data[1][:, : self.nelec[1]]]
        dm = trial[0] @ trial[0].T + trial[1] @ trial[1].T
        ham_data["mf_shifts"] = 1.0j * vmap(
            lambda x: jnp.sum(x.reshape(self.norb, self.norb) * dm)
        )(ham_data["chol"])
        ham_data["mf_shifts_fp"] = jnp.stack(
            (
                ham_data["mf_shifts"] / self.nelec[0] / 2.0,
                ham_data["mf_shifts"] / self.nelec[1] / 2.0,
            )
        )
        ham_data["h0_prop"] = (
            -ham_data["h0"] - jnp.sum(ham_data["mf_shifts"] ** 2) / 2.0
        )
        ham_data["h0_prop_fp"] = jnp.stack(
            (
                (ham_data["h0_prop"] + ham_data["ene0"]) / self.nelec[0] / 2.0,
                (ham_data["h0_prop"] + ham_data["ene0"]) / self.nelec[1] / 2.0,
            )
        )
        v0 = 0.5 * jnp.einsum(
            "gik,gjk->ij",
            ham_data["chol"].reshape(-1, self.norb, self.norb),
            ham_data["chol"].reshape(-1, self.norb, self.norb),
            optimize="optimal",
        )
        v1 = jnp.real(
            1.0j
            * jnp.einsum(
                "g,gik->ik",
                ham_data["mf_shifts"],
                ham_data["chol"].reshape(-1, self.norb, self.norb),
            )
        )
        h1_mod = ham_data["h1"] - jnp.array([v0 + v1, v0 + v1])
        ham_data["exp_h1"] = jnp.array(
            [
                jsp.linalg.expm(-dt * h1_mod[0] / 2.0),
                jsp.linalg.expm(-dt * h1_mod[1] / 2.0),
            ]
        )
        return ham_data

    def __hash__(self):
        return hash((self.norb, self.nelec, self.nchol))


@dataclass
class hamiltonian_ghf:
    norb: int  # number of spatial orbitals
    nelec: tuple
    nchol: int

    @partial(jit, static_argnums=(0,))
    def rot_orbs(self, ham_data, wave_data):
        return ham_data

    @partial(jit, static_argnums=(0,))
    def rot_ham(self, ham_data, wave_data):
        ham_data["h1"] = (
            ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
        )
        ham_data["h1"] = (
            ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
        )
        trial = wave_data[:, : self.nelec[0] + self.nelec[1]]
        ham_data["rot_h1"] = trial.T @ jnp.block(
            [
                [ham_data["h1"][0], jnp.zeros_like(ham_data["h1"][1])],
                [jnp.zeros_like(ham_data["h1"][0]), ham_data["h1"][1]],
            ]
        )
        ham_data["rot_chol"] = vmap(
            lambda x: jnp.hstack(
                [trial.T[:, : self.norb] @ x, trial.T[:, self.norb :] @ x]
            ),
            in_axes=(0),
        )(ham_data["chol"].reshape(-1, self.norb, self.norb))
        return ham_data

    @partial(jit, static_argnums=(0, 3))
    def prop_ham(self, ham_data, dt, _trial, wave_data):
        trial = wave_data[:, : self.nelec[0] + self.nelec[1]]
        dm_ghf = trial @ trial.T
        dm = dm_ghf[: self.norb, : self.norb] + dm_ghf[self.norb :, self.norb :]
        ham_data["mf_shifts"] = 1.0j * vmap(
            lambda x: jnp.sum(x.reshape(self.norb, self.norb) * dm)
        )(ham_data["chol"])
        ham_data["mf_shifts_fp"] = jnp.stack(
            (
                ham_data["mf_shifts"] / self.nelec[0] / 2.0,
                ham_data["mf_shifts"] / self.nelec[1] / 2.0,
            )
        )
        ham_data["h0_prop"] = (
            -ham_data["h0"] - jnp.sum(ham_data["mf_shifts"] ** 2) / 2.0
        )
        ham_data["h0_prop_fp"] = jnp.stack(
            (
                (ham_data["h0_prop"] + ham_data["ene0"]) / self.nelec[0] / 2.0,
                (ham_data["h0_prop"] + ham_data["ene0"]) / self.nelec[1] / 2.0,
            )
        )
        v0 = 0.5 * jnp.einsum(
            "gik,gjk->ij",
            ham_data["chol"].reshape(-1, self.norb, self.norb),
            ham_data["chol"].reshape(-1, self.norb, self.norb),
            optimize="optimal",
        )
        v1 = jnp.real(
            1.0j
            * jnp.einsum(
                "g,gik->ik",
                ham_data["mf_shifts"],
                ham_data["chol"].reshape(-1, self.norb, self.norb),
            )
        )
        h1_mod = ham_data["h1"] - jnp.array([v0 + v1, v0 + v1])
        ham_data["exp_h1"] = jnp.array(
            [
                jsp.linalg.expm(-dt * h1_mod[0] / 2.0),
                jsp.linalg.expm(-dt * h1_mod[1] / 2.0),
            ]
        )
        return ham_data

    def __hash__(self):
        return hash((self.norb, self.nelec, self.nchol))


@dataclass
class hamiltonian_noci:
    norb: int  # number of spatial orbitals
    nelec: tuple
    nchol: int

    @partial(jit, static_argnums=(0,))
    def rot_orbs(self, ham_data, wave_data):
        return ham_data

    @partial(jit, static_argnums=(0,))
    def rot_orbs_single_det(self, ham_data, trial_up, trial_dn):
        rot_h1 = [
            trial_up[:, : self.nelec[0]].T @ ham_data["h1"][0],
            trial_dn[:, : self.nelec[1]].T @ ham_data["h1"][1],
        ]
        rot_chol = [
            jnp.einsum(
                "pi,gij->gpj",
                trial_up[:, : self.nelec[0]].T,
                ham_data["chol"].reshape(-1, self.norb, self.norb),
            ),
            jnp.einsum(
                "pi,gij->gpj",
                trial_dn[:, : self.nelec[1]].T,
                ham_data["chol"].reshape(-1, self.norb, self.norb),
            ),
        ]
        return rot_h1, rot_chol

    @partial(jit, static_argnums=(0,))
    def rot_ham(self, ham_data, wave_data):
        ham_data["h1"] = (
            ham_data["h1"].at[0].set((ham_data["h1"][0] + ham_data["h1"][0].T) / 2.0)
        )
        ham_data["h1"] = (
            ham_data["h1"].at[1].set((ham_data["h1"][1] + ham_data["h1"][1].T) / 2.0)
        )
        trial = wave_data[1]
        ham_data["rot_h1"], ham_data["rot_chol"] = vmap(
            self.rot_orbs_single_det, in_axes=(None, 0, 0)
        )(ham_data, trial[0], trial[1])
        return ham_data

    @partial(jit, static_argnums=(0, 3))
    def prop_ham(self, ham_data, dt, trial, wave_data):
        dm = trial.get_rdm1(wave_data)
        ham_data["mf_shifts"] = 1.0j * vmap(
            lambda x: jnp.sum(x.reshape(self.norb, self.norb) * dm)
        )(ham_data["chol"])
        ham_data["mf_shifts_fp"] = jnp.stack(
            (
                ham_data["mf_shifts"] / self.nelec[0] / 2.0,
                ham_data["mf_shifts"] / self.nelec[1] / 2.0,
            )
        )
        ham_data["h0_prop"] = (
            -ham_data["h0"] - jnp.sum(ham_data["mf_shifts"] ** 2) / 2.0
        )
        ham_data["h0_prop_fp"] = jnp.stack(
            (
                (ham_data["h0_prop"] + ham_data["ene0"]) / self.nelec[0] / 2.0,
                (ham_data["h0_prop"] + ham_data["ene0"]) / self.nelec[1] / 2.0,
            )
        )
        v0 = 0.5 * jnp.einsum(
            "gik,gjk->ij",
            ham_data["chol"].reshape(-1, self.norb, self.norb),
            ham_data["chol"].reshape(-1, self.norb, self.norb),
            optimize="optimal",
        )
        v1 = jnp.real(
            1.0j
            * jnp.einsum(
                "g,gik->ik",
                ham_data["mf_shifts"],
                ham_data["chol"].reshape(-1, self.norb, self.norb),
            )
        )
        h1_mod = ham_data["h1"] - jnp.array([v0 + v1, v0 + v1])
        ham_data["exp_h1"] = jnp.array(
            [
                jsp.linalg.expm(-dt * h1_mod[0] / 2.0),
                jsp.linalg.expm(-dt * h1_mod[1] / 2.0),
            ]
        )
        return ham_data

    def __hash__(self):
        return hash((self.norb, self.nelec, self.nchol))
