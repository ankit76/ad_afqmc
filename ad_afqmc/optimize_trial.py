import jax.numpy as jnp

try:
    import jaxopt  # type: ignore
except ImportError:
    jaxopt = None
from jax import value_and_grad, vmap


def packArray_s2(A, B):
    return jnp.hstack([A, B])


def packArray_s2_tr(A, B):
    return jnp.hstack([A.real, A.imag, B.real, B.imag])


def unpackArray_s2(walkerArray, nelec):
    return [walkerArray[:, : nelec[0]], walkerArray[:, nelec[0] :]]


def unpackArray_s2_tr(walkerArray, nelec):
    return [
        walkerArray[:, : nelec[0]] + 1.0j * walkerArray[:, nelec[0] : 2 * nelec[0]],
        walkerArray[:, 2 * nelec[0] : 2 * nelec[0] + nelec[1]]
        + 1.0j * walkerArray[:, 2 * nelec[0] + nelec[1] :],
    ]


# @partial(jit, static_argnums=(1,2,3))
def calc_energy_s2(walkerArray, nelec, ham_data, wave_data):

    norb = walkerArray.shape[0]
    S, Sz, wigner, beta_vals = wave_data["wigner"]

    RotMatrix = vmap(
        lambda beta: jnp.array(
            [
                [jnp.cos(beta / 2), jnp.sin(beta / 2)],
                [-jnp.sin(beta / 2), jnp.cos(beta / 2)],
            ]
        )
    )(beta_vals)

    def applyRotMat(detA, detB, mat):
        A, B = detA * mat[0, 0], detB * mat[0, 1]
        C, D = detA * mat[1, 0], detB * mat[1, 1]

        detAout = jnp.block([[A, B], [C, D]])
        return detAout

    Alpha, Beta = unpackArray_s2(walkerArray, nelec)
    # Alpha = walkerArray[:,:nelec[0]]
    # Beta  = walkerArray[:,nelec[0]:]

    def calc_overlap(A, B, walker):
        return jnp.linalg.det(
            jnp.vstack(
                [
                    A.T.conj() @ walker[:norb],
                    B.T.conj() @ walker[norb:],
                ]
            )
        )

    def calc_energy(Atrial, Btrial, walker, ham_data):
        norb = Atrial.shape[0]
        h0, h1, chol = (
            ham_data["h0"],
            ham_data["h1"][0],
            ham_data["chol"].reshape(-1, norb, norb),
        )
        ene0 = h0

        bra = jnp.block([[Atrial, 0 * Btrial], [0 * Atrial, Btrial]])
        gf = (walker.dot(jnp.linalg.inv(bra.T.conj() @ walker)) @ bra.T.conj()).T

        gfA, gfB = gf[:norb, :norb], gf[norb:, norb:]
        gfAB, gfBA = gf[:norb, norb:], gf[norb:, :norb]

        ene1 = jnp.sum(gfA * h1) + jnp.sum(gfB * h1)

        f_up = jnp.einsum("gij,jk->gik", chol, gfA.T, optimize="optimal")
        f_dn = jnp.einsum("gij,jk->gik", chol, gfB.T, optimize="optimal")
        c_up = vmap(jnp.trace)(f_up)
        c_dn = vmap(jnp.trace)(f_dn)
        J = jnp.sum(c_up * c_up) + jnp.sum(c_dn * c_dn) + 2.0 * jnp.sum(c_up * c_dn)

        K = jnp.sum(vmap(lambda x: x * x.T)(f_up)) + jnp.sum(
            vmap(lambda x: x * x.T)(f_dn)
        )

        f_ab = jnp.einsum("gip,pj->gij", chol, gfBA.T, optimize="optimal")
        f_ba = jnp.einsum("gip,pj->gij", chol, gfAB.T, optimize="optimal")
        K += 2.0 * jnp.sum(vmap(lambda x, y: x * y.T)(f_ab, f_ba))

        return ene1 + (J - K) / 2.0 + h0

    S2walkers = vmap(applyRotMat, (None, None, 0))(Alpha, Beta, RotMatrix)
    ovlp1 = vmap(calc_overlap, (None, None, 0))(Alpha, Beta, S2walkers)
    Eloc1 = vmap(calc_energy, (None, None, 0, None))(Alpha, Beta, S2walkers, ham_data)
    # ovlp2 = vmap(calc_overlap, (None, None, 0))(Alpha, Beta, S2walkers.conj())
    # Eloc2 = vmap(calc_energy, (None, None, 0, None))(Alpha, Beta, S2walkers.conj(), ham_data)
    totalOvlp = jnp.sum(ovlp1 * wigner)  # + ovlp2 * wigner)
    return (
        jnp.sum(Eloc1 * ovlp1 * wigner) / totalOvlp
    )  # + Eloc2 * ovlp2 * wigner )/totalOvlp).real


# @partial(jit, static_argnums=(1,2,3))
def calc_energy_s2_tr(walkerArray, nelec, ham_data, wave_data):

    norb = walkerArray.shape[0]
    S, Sz, wigner, beta_vals = wave_data["wigner"]

    RotMatrix = vmap(
        lambda beta: jnp.array(
            [
                [jnp.cos(beta / 2), jnp.sin(beta / 2)],
                [-jnp.sin(beta / 2), jnp.cos(beta / 2)],
            ]
        )
    )(beta_vals)

    def applyRotMat(detA, detB, mat):
        A, B = detA * mat[0, 0], detB * mat[0, 1]
        C, D = detA * mat[1, 0], detB * mat[1, 1]

        detAout = jnp.block([[A, B], [C, D]])
        return detAout

    Alpha, Beta = unpackArray_s2_tr(walkerArray, nelec)

    def calc_overlap(A, B, walker):
        return jnp.linalg.det(
            jnp.vstack(
                [
                    A.T.conj() @ walker[:norb],
                    B.T.conj() @ walker[norb:],
                ]
            )
        )

    def calc_energy(Atrial, Btrial, walker, ham_data):
        norb = Atrial.shape[0]
        h0, h1, chol = (
            ham_data["h0"],
            ham_data["h1"][0],
            ham_data["chol"].reshape(-1, norb, norb),
        )
        ene0 = h0

        bra = jnp.block([[Atrial, 0 * Btrial], [0 * Atrial, Btrial]])
        gf = (walker.dot(jnp.linalg.inv(bra.T.conj() @ walker)) @ bra.T.conj()).T

        gfA, gfB = gf[:norb, :norb], gf[norb:, norb:]
        gfAB, gfBA = gf[:norb, norb:], gf[norb:, :norb]

        ene1 = jnp.sum(gfA * h1) + jnp.sum(gfB * h1)

        f_up = jnp.einsum("gij,jk->gik", chol, gfA.T, optimize="optimal")
        f_dn = jnp.einsum("gij,jk->gik", chol, gfB.T, optimize="optimal")
        c_up = vmap(jnp.trace)(f_up)
        c_dn = vmap(jnp.trace)(f_dn)
        J = jnp.sum(c_up * c_up) + jnp.sum(c_dn * c_dn) + 2.0 * jnp.sum(c_up * c_dn)

        K = jnp.sum(vmap(lambda x: x * x.T)(f_up)) + jnp.sum(
            vmap(lambda x: x * x.T)(f_dn)
        )

        f_ab = jnp.einsum("gip,pj->gij", chol, gfBA.T, optimize="optimal")
        f_ba = jnp.einsum("gip,pj->gij", chol, gfAB.T, optimize="optimal")
        K += 2.0 * jnp.sum(vmap(lambda x, y: x * y.T)(f_ab, f_ba))

        return ene1 + (J - K) / 2.0 + h0

    S2walkers = vmap(applyRotMat, (None, None, 0))(Alpha, Beta, RotMatrix)
    ovlp1 = vmap(calc_overlap, (None, None, 0))(Alpha, Beta, S2walkers)
    Eloc1 = vmap(calc_energy, (None, None, 0, None))(Alpha, Beta, S2walkers, ham_data)
    ovlp2 = vmap(calc_overlap, (None, None, 0))(Alpha, Beta, S2walkers.conj())
    Eloc2 = vmap(calc_energy, (None, None, 0, None))(
        Alpha, Beta, S2walkers.conj(), ham_data
    )
    totalOvlp = jnp.sum(ovlp1 * wigner + ovlp2 * wigner)
    return (jnp.sum(Eloc1 * ovlp1 * wigner + Eloc2 * ovlp2 * wigner) / totalOvlp).real


def optimize_trial(ham_data, trial, wave_data):
    # np.random.seed(5)
    # ham_data, trial, wave_data, options = (
    #     launch_script.setup_afqmc(options, options["tmpdir"])
    # )

    na, nb = trial.nelec[0], trial.nelec[1]
    # wa = wave_data['mo_coeff'][0] @ jnp.linalg.qr(np.random.random((na,na)) + 1.j*np.random.random((na,na)))[0]
    # wb = wave_data['mo_coeff'][1] @ jnp.linalg.qr(np.random.random((nb,nb)) + 1.j*np.random.random((nb,nb)))[0]
    walkerArray = jnp.hstack([wave_data["mo_coeff"][0], wave_data["mo_coeff"][1]])
    # walkerArray = jnp.hstack( [wa.real, wa.imag, wb.real, wb.imag])
    E = calc_energy_s2(walkerArray, trial.nelec, ham_data, wave_data)
    print(E)

    # wa = wave_data['mo_coeff'][0] @ jnp.linalg.qr(np.random.random((na,na)) + 1.j*np.random.random((na,na)))[0]
    # wb = wave_data['mo_coeff'][1] @ jnp.linalg.qr(np.random.random((nb,nb)) + 1.j*np.random.random((nb,nb)))[0]
    # print(calc_energy_s2(packArray_s2(wa, wb), trial.nelec, ham_data, wave_data))
    # print(calc_energy_s2_tr(packArray_s2_tr(wa, wb), trial.nelec, ham_data, wave_data))

    calc_energy1 = lambda x: calc_energy_s2(x, trial.nelec, ham_data, wave_data)
    energy_and_grad_fn = value_and_grad(calc_energy1, argnums=0)

    # optimizer = jaxopt.GradientDescent(
    #     fun=energy_and_grad_fn,
    #     value_and_grad=True,  # Tell it your function returns both
    #     stepsize=0.01,
    #     verbose = True
    # )
    if jaxopt is None:
        raise ImportError(
            "jaxopt is not installed. Please install it to use this function."
        )
    optimizer = jaxopt.LBFGS(
        fun=energy_and_grad_fn, value_and_grad=True, maxiter=500, tol=1e-6, verbose=True
    )

    result = optimizer.run(walkerArray)
    print(f"Optimal value: {result.state.value}")
    print(f"Converged in {result.state.iter_num} iterations")

    # walkerArray = packArray_s2_tr(wa, wb)
    # for i in range(1000):
    #     E, dE_dwalker = energy_and_grad_fn(walkerArray)

    #     Atemp, Btemp = unpackArray_s2_tr(walkerArray, trial.nelec)
    #     Es2 = calc_energy_s2(packArray_s2(Atemp, Btemp), trial.nelec, ham_data, wave_data)
    #     print(i, E, jnp.linalg.norm(dE_dwalker), Es2)
    #     if (i == 23):
    #         import pdb; pdb.set_trace()
    #     walkerArray = walkerArray - 0.01 * dE_dwalker
    #     #A, _ = jnp.linalg.qr(walkerArray[:,:trial.nelec[0]])
    #     #B, _ = jnp.linalg.qr(walkerArray[:,trial.nelec[0]:])
    #     #wave_data["mo_coeff"] = [A,B]

    # exit(0)
    orbitals = unpackArray_s2(result.params, trial.nelec)
    return [jnp.linalg.qr(orbitals[0])[0], jnp.linalg.qr(orbitals[1])[0]]
