"""
Code from ipie.
"""

import os
import sys
import socket
import subprocess
from typing import Dict

import numpy

def get_node_mem():
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3.0
    except:
        return 0.0

def get_numpy_blas_info() -> Dict[str, str]:
    """Get useful numpy blas / lapack info."""
    info = {}
    try:
        config = numpy.show_config(mode="dicts")
        blas_config = config["Build Dependencies"]["blas"]
        info["BLAS"] = {
            "lib": blas_config["name"],
            "version": blas_config["version"],
            "include directory": blas_config["lib directory"],
        }
        for k, v in info["BLAS"].items():
            print(f"# - BLAS {k}: {v}")
    except TypeError:
        try:
            np_lib = numpy.__config__.blas_opt_info["libraries"]  # pylint:  disable=no-member
            lib_dir = numpy.__config__.blas_opt_info["library_dirs"]  # pylint: disable=no-member
        except AttributeError:
            np_lib = numpy.__config__.blas_ilp64_opt_info["libraries"]  # pylint:  disable=no-member
            lib_dir = numpy.__config__.blas_ilp64_opt_info[  # pylint:  disable=no-member
                "library_dirs"
            ]
        print(f"# - BLAS lib: {' '.join(np_lib):s}")
        print(f"# - BLAS dir: {' '.join(lib_dir):s}")
        info["BLAS"] = {
            "lib": " ".join(np_lib),
            "path": " ".join(lib_dir),
        }
    return info

def get_git_info():
    """Return git info.

    Adapted from:
        http://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script

    Returns
    -------
    sha1 : string
        git hash with -dirty appended if uncommitted changes.
    branch : string
        Current branch
    local_mod : list of strings
        List of locally modified files tracked and untracked.
    """

    under_git = True
    try:
        src = os.path.dirname(__file__) + "/../"
        sha1 = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=src, stderr=subprocess.DEVNULL
        ).strip()
        suffix = subprocess.check_output(
            ["git", "status", "-uno", "--porcelain", "./ad_afqmc"], cwd=src
        ).strip()
        local_mods = (
            subprocess.check_output(["git", "status", "--porcelain", "./ad_afqmc"], cwd=src)
            .strip()
            .decode("utf-8")
            .split()
        )
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=src
        ).strip()
    except subprocess.CalledProcessError:
        under_git = False
    except Exception as error:
        suffix = False
        print(f"couldn't determine git hash : {error}")
        sha1 = "none".encode()
        local_mods = []
    if under_git:
        if suffix:
            return sha1.decode("utf-8") + "-dirty", branch.decode("utf-8"), local_mods
        else:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=src
            ).strip()
            return sha1.decode("utf-8"), branch.decode("utf_8"), local_mods
    else:
        return None, None, []

def print_env_info(sha1, branch, local_mods):
    import ad_afqmc

    version = getattr(ad_afqmc, "__version__", "Unknown")
    print(f"# ad_afqmc version: {version}")
    if sha1 is not None:
        print(f"# Git hash: {sha1:s}.")
        print(f"# Git branch: {branch:s}.")
    if len(local_mods) > 0:
        print("# Found uncommitted changes and/or untracked files.")
        for prefix, file in zip(local_mods[::2], local_mods[1::2]):
            if prefix == "M":
                print(f"# Modified : {file:s}")
            elif prefix == "??":
                print(f"# Untracked : {file:s}")
    mem = get_node_mem()
    print(f"# Approximate memory available per node: {mem:.4f} GB.")
    hostname = socket.gethostname()
    print(f"# Root processor name: {hostname}")
    py_ver = sys.version.splitlines()
    print(f"# Python interpreter: {' '.join(py_ver):s}")
    info = {"python": py_ver, "branch": branch, "sha1": sha1}
    from importlib import import_module

    for lib in ["numpy", "scipy", "h5py", "mpi4py", "cupy"]:
        try:
            l = import_module(lib)
            # Strip __init__.py
            path = l.__file__[:-12]
            vers = l.__version__
            print(f"# Using {lib:s} v{vers:s} from: {path:s}.")
            info[f"{lib:s}"] = {"version": vers, "path": path}
            if lib == "numpy":
                info[f"{lib:s}"] = get_numpy_blas_info()
            elif lib == "mpi4py":
                mpicc = l.get_config().get("mpicc", "none")
                print(f"# - mpicc: {mpicc:s}")
                info[f"{lib:s}"]["mpicc"] = mpicc
            elif lib == "cupy":
                try:
                    cu_info = l.cuda.runtime.getDeviceProperties(0)
                    cuda_compute = l.cuda.Device().compute_capability
                    cuda_version = str(l.cuda.runtime.runtimeGetVersion())
                    cuda_compute = cuda_compute[0] + "." + cuda_compute[1]
                    # info['{:s}'.format(lib)]['cuda'] = {'info': ' '.join(np_lib),
                    #                                    'path': ' '.join(lib_dir)}
                    version_string = (
                        cuda_version[:2] + "." + cuda_version[2:4] + "." + cuda_version[4]
                    )
                    print(f"# - CUDA compute capability: {cuda_compute:s}")
                    print(f"# - CUDA version: {version_string}")
                    print(f"# - GPU Type: {str(cu_info['name'])[1:]:s}")
                    print(f"# - GPU Mem: {cu_info['totalGlobalMem'] / 1024 ** 3.0:.3f} GB")
                    print(f"# - Number of GPUs: {l.cuda.runtime.getDeviceCount():d}")
                except:
                    print("# cupy import error")
        except (ModuleNotFoundError, ImportError):
            print(f"# Package {lib:s} not found.")
    return info

