"""
Author: Kaitlin
"""

import jax
import time
import os
import argparse
import logging
import cimist as cst
import dill as pkl  # type: ignore
import mdtraj as md  # type: ignore
from jax import jit, vmap, Array
from typing import List, NamedTuple
from functools import partial, wraps
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "-t",
        "--trajectory",
        type=str,
        help="""The path to the trajectory file,
or to a directory that contains all trajectory files and nothing else.
Note that if a directory is supplied, all files in that directory must be valid molecular
dynamics trajectory files.""",
    )
    parser.add_argument("-s", "--topology", type=str, help="""The path to the topology file.""")

    parser.add_argument("-o", "--output_prefix", help="""The prefix for the output directory.""")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="The random number generator seed, default 0.",
    )
    parser.add_argument(
        "--min_mass",
        type=float,
        default=0.01,
        help="Minimum probability for conformations, default 0.01.",
    )
    parser.add_argument(
        "--prior",
        type=str,
        default="haldane",
        choices={"percs", "haldane", "jeffreys", "laplace"},
        help="""Prior to use for residue entropy and pairwise mutual information estimation 
        with the Dirichlet distribution. Each prior corresponds to adding the same number of 
        pseudocounts to each conformation.
          Options are:
              -'haldane' : 0 pseudocounts (DEFAULT)
              -'percs' : 1/K pseudocounts, where K is the number of conformations
              -'jeffreys' : 1/2 pseudocounts
              -'laplace' : 1 pseudocount
              
        Note that of these options, only 'haldane' and 'percs' add the same total number of
        pseudocounts to each distribution.""",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # some setup so we don't have to wait for imports to see if the inputs are bad
    args = parse_args()


t0 = time.time()


dt = time.time() - t0
m, s = divmod(dt, 60)


class MDData(NamedTuple):
    traj: md.Trajectory
    angles: Array
    cloud_mask: Array
    residues: List


def time_and_message(message, func, printer=logging.info):
    @wraps(func)
    def wrapped(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        dt = time.time() - t0
        m, s = divmod(dt, 60)
        printer(message + f" in {int(m)} minutes and {round(s, 3)} seconds.")
        return result

    return wrapped


def combine_mixture_fit_batches(batches):
    concatenated = [jnp.concatenate(x) for x in list(zip(*batches))]
    return cst.ci.vmm.MixtureFit(*concatenated)


def load_dataset(
    xyz,
    topology,
    stride=1,
) -> MDData:
    traj = cst.io.load_traj(xyz, topology, stride=stride)
    traj, residues = to_single_chain(traj)
    internal = cst.utils.traj_to_internal(traj)
    angles, cloud_mask = cst.utils.flatten_masks(internal)
    return MDData(traj, angles, cloud_mask, residues)


def to_single_chain(traj):
    traj = traj.atom_slice(traj.topology.select("protein"))
    df = traj.topology.to_dataframe()[0]
    seq_ordered = df[df.name == "CA"]
    ix_update_dict = {
        (s, n, c): i for (i, (s, n, c)) in enumerate(zip(seq_ordered.resSeq, seq_ordered.resName, seq_ordered.chainID))
    }

    df["resSeq"] = df[["resSeq", "resName", "chainID"]].apply(lambda x: ix_update_dict[tuple(x)], axis=1)
    df["chainID"] = 0
    new_top = md.Topology.from_dataframe(df)

    alphabet = "abcdefghijklmnopqrstuvwxyz".upper()
    residues = [f"{r}{s}_{alphabet[c]}" for (s, r, c) in ix_update_dict.keys()]
    return md.Trajectory(traj.xyz, new_top), residues


def main():
    args = parse_args()

    import datetime

    today = datetime.datetime.today()
    # CONFIGURE LOGGING
    output_prefix = f"{args.output_prefix}_seed_{args.seed}" + os.sep
    if not os.path.exists(output_prefix):
        os.mkdir(output_prefix)

    logfile = output_prefix + "logging.txt"
    logging.basicConfig(filename=logfile, level=logging.INFO, filemode="w")
    logging.info("Began pipeline at " + str(today) + ".")

    data = load_dataset(args.trajectory, args.topology)
    reference_structure = md.load(args.topology)

    output_prefix = f"{args.output_prefix}_seed_{args.seed}" + os.sep
    if not os.path.exists(output_prefix):
        os.mkdir(output_prefix)

    def make_keys(seed):
        return jax.random.split(jax.random.PRNGKey(seed), data.cloud_mask.shape[0])

    # make batches for kmeans++
    init_keys = make_keys(args.seed)
    n_chunks = max(1, data.angles.shape[0] // 20)
    angles_batches = jnp.array_split(data.angles, n_chunks)
    mask_batches = jnp.array_split(data.cloud_mask, n_chunks)
    keys_batches = jnp.array_split(init_keys, n_chunks)
    init_states_func = vmap(cst.ci.vmm.init_random_mixture_state)

    # define path to folder and create it if it does not exist
    logging.info("Initializing mixture model fits")
    mixture_state = combine_mixture_fit_batches(
        [init_states_func(*args) for args in zip(angles_batches, mask_batches, keys_batches)]
    )

    logging.info("Beginning von Mises mixture models fits with expectation-maximization...")
    step_if_converged = jit(partial(cst.ci.vmm.step_if_not_converged, gtol=1e-3, gmaxiter=500))
    _advance = vmap(step_if_converged)
    num_residues = len(mixture_state.converged)  # TODO: ????
    num_converged = jnp.sum(mixture_state.converged)
    for i in range(100):
        if num_converged == num_residues:
            # i > 0 ensures we didn't start with a converged set of models
            # update responsibilities one last time for predicting barcodes
            if i > 0:
                update_r = time_and_message(
                    "Carried out final update of mixture component responsibilities",
                    vmap(cst.ci.vmm.update_r),
                )
                mixture_state = update_r(data.angles, mixture_state)
            break
        print(80 * "-")
        advance = time_and_message(f"Completed EM iteration {i} for all residues", _advance)
        mixture_state = advance(data.angles, mixture_state)
        num_converged = jnp.sum(mixture_state.converged)

        logging.info(
            f"Reached convergence for {jnp.sum(mixture_state.converged)} out of \
              {len(mixture_state.converged)} residues."
        )

    if mixture_state.r.sum() == 0:
        new_r = vmap(cst.ci.vmm.e_step)(
            data.angles,
            mixture_state.mu,
            mixture_state.kappa,
            mixture_state.logw,
            mixture_state.mask,
        )
        mixture_state = mixture_state._replace(logw=jnp.log(new_r.mean(axis=1)), r=new_r)

    n_components = jnp.int64(mixture_state.n_components)
    n_iter = jnp.int64(mixture_state.n_iter)
    converged = jnp.bool(mixture_state.converged)
    atol = jnp.float64(mixture_state.atol)

    mixture_state = cst.ci.vmm.MixtureFitState(
        n_components=n_components,
        mu=mixture_state.mu,
        kappa=mixture_state.kappa,
        logw=mixture_state.logw,
        mask=mixture_state.mask,
        r=mixture_state.r,
        log_likelihood=mixture_state.log_likelihood,
        n_iter=n_iter,
        converged=converged,
        atol=atol,
        statuses=mixture_state.statuses,
        njevs=mixture_state.njevs,
        nfevs=mixture_state.nfevs,
        nits=mixture_state.nits,
        successes=mixture_state.successes,
    )

    D = cst.ci.dbscan.compute_distances(mixture_state)
    weights = mixture_state.r.mean(axis=1)

    states = vmap(cst.ci.dbscan.dbscan_eps_std)(D, weights, mixture_state.r)
    tree = cst.MIST.from_residue_states(states, [str(r) for r in data.residues], prior=args.prior, uncertainty=True)
    fname = output_prefix + "RESULTS_ciMIST.h5"

    with open(fname.replace(".h5", ".pkl"), "wb") as f:
        pkl.dump(tree, f)

    cst.io.save_tree_h5(tree, fname)

    cst.pymol.tree_cartoon(tree, reference_structure, output_prefix + "pymol" + os.sep)
    cst.io.save_coarse_graining_h5(states, fname)
    cst.io.save_VMM_h5(mixture_state, fname)
    return None


if __name__ == "__main__":
    main()
