#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
# Import the new sharding and dataloader utilities
import cimist.sharding_utils as su
from cimist.dataloader import MDStreamer

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
    parser.add_argument("--seed", type=int, default=0, help="The random number generator seed, default 0.")
    parser.add_argument("--min_mass", type=float, default=0.01, help="Minimum probability for conformations, default 0.01.")
    parser.add_argument("--prior", type=str, default="haldane", choices={"percs", "haldane", "jeffreys", "laplace"}, help="""Prior to use for residue entropy and pairwise mutual information estimation.""")
    # Add arguments for mini-batch training
    parser.add_argument("--batch_size", type=int, default=1000, help="Number of frames per batch for streaming.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of passes over the entire trajectory.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()


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


def main():
    args = parse_args()

    import datetime

    today = datetime.datetime.today()
    output_prefix = f"{args.output_prefix}_seed_{args.seed}" + os.sep
    if not os.path.exists(output_prefix):
        os.mkdir(output_prefix)

    logfile = output_prefix + "logging.txt"
    logging.basicConfig(filename=logfile, level=logging.INFO, filemode="w")
    logging.info("Began pipeline at " + str(today) + ".")

    # --- 1. SHARDING AND DATALOADER SETUP ---
    mesh = su.setup_mesh()
    logging.info(f"JAX mesh created with devices: {mesh.devices.tolist()}")

    # Initialize the data streamer. This doesn't load the data, just sets up the iterator.
    data_streamer = MDStreamer(args.trajectory, args.topology, batch_size=args.batch_size)
    logging.info(f"MDStreamer initialized with batch size {args.batch_size}.")
    
    # We still need a reference structure for saving the PyMOL visualization.
    reference_structure = md.load(args.topology)

    with mesh:
        # --- 2. SHARD CONSTANT DATA ---
        # The cloud_mask is constant for all frames. We get it from the streamer and shard it.
        sharded_mask = su.shard_array(mesh, data_streamer.cloud_mask, axis=0)
        logging.info("Sharded mask array across devices.")

        def make_keys(seed):
            keys = jax.random.split(jax.random.PRNGKey(seed), data_streamer.n_residues)
            return su.shard_array(mesh, keys, axis=0)

        # --- 3. INITIALIZE VMM STATE ON DEVICES ---
        init_keys = make_keys(args.seed)
        
        # We need a dummy `angles` array to get the shape for initialization.
        # This is a bit of a workaround for `vmap` to know the input shapes.
        # The shape is (n_residues, batch_size, n_dihedrals)
        dummy_angles_batch = jnp.zeros(
            (data_streamer.n_residues, args.batch_size, data_streamer.cloud_mask.shape[1]), 
            dtype=jnp.float32
        )
        sharded_dummy_angles = su.shard_array(mesh, dummy_angles_batch, axis=0)

        init_states_func = vmap(cst.ci.vmm.init_random_mixture_state)
        logging.info("Initializing mixture model fits on sharded data")
        mixture_state = init_states_func(sharded_dummy_angles, sharded_mask, init_keys)

        # --- 4. RUN ONLINE EM FITTING IN PARALLEL ---
        logging.info("Beginning online von Mises mixture models fits...")
        
        step_fn = jit(vmap(partial(cst.ci.vmm.em_step, gtol=1e-3, gmaxiter=500)))
        
        for epoch in range(args.epochs):
            logging.info(f"--- Starting Epoch {epoch + 1}/{args.epochs} ---")
            batch_num = 0
            for batch in data_streamer:
                t0_batch = time.time()
                # Shard the incoming batch of angles along the residue axis.
                sharded_angles_batch = su.shard_array(mesh, batch.angles, axis=0)
                
                # Perform one EM step using the current batch of data.
                mixture_state = step_fn(sharded_angles_batch, mixture_state)
                
                dt_batch = time.time() - t0_batch
                logging.info(f"Epoch {epoch+1}, Batch {batch_num+1} processed in {dt_batch:.2f}s.")
                batch_num += 1

        logging.info("Completed online EM fitting.")

        # --- 5. RUN DBSCAN IN PARALLEL ---
        # Final responsibilities (`r`) are needed for DBSCAN. We compute them based on the
        # full dataset by iterating one last time. A more practical approach for very
        # large data would be to use the final model state directly, or compute `r` on a
        # representative subset. Here we'll re-iterate for accuracy.
        
        logging.info("Calculating final responsibilities for DBSCAN...")
        # Re-initialize the `r` component of the state to accumulate new values
        final_r_shape = (data_streamer.n_residues, mixture_state.r.shape[1], mixture_state.r.shape[2])
        mixture_state = mixture_state._replace(r=jnp.zeros(final_r_shape, dtype=jnp.float32))
        
        # Accumulate responsibilities (the E-step) over the whole dataset
        e_step_fn = jit(vmap(cst.ci.vmm.e_step))
        total_frames = 0
        for batch in data_streamer:
            sharded_angles_batch = su.shard_array(mesh, batch.angles, axis=0)
            r_batch = e_step_fn(sharded_angles_batch, mixture_state.mu, mixture_state.kappa, mixture_state.logw, sharded_mask)
            # This accumulation is an approximation; a more correct M-step would be needed.
            # For DBSCAN, we primarily need the final weights, which this gives.
            mixture_state = mixture_state._replace(r=mixture_state.r + r_batch.sum(axis=1)) # sum over frames in batch
            total_frames += batch.angles.shape[1]
        
        # Average the responsibilities
        mixture_state = mixture_state._replace(r=mixture_state.r / total_frames)

        D = cst.ci.dbscan.compute_distances(mixture_state)
        weights = mixture_state.r.mean(axis=1) # Average over frames

        dbscan_fn = jit(vmap(cst.ci.dbscan.dbscan_eps_std))
        states = dbscan_fn(D, weights, mixture_state.r)

        # --- 6. GATHER RESULTS FOR CPU-BASED MIST CALCULATION ---
        logging.info("Gathering data back to host for MIST calculation.")
        states_on_host = jax.device_get(states)

    # --- MIST Calculation and Saving (on Host) ---
    tree = cst.MIST.from_residue_states(states_on_host, [str(r) for r in data_streamer.residues], prior=args.prior, uncertainty=True)
    
    fname = output_prefix + "RESULTS_ciMIST.h5"
    with open(fname.replace(".h5", ".pkl"), "wb") as f:
        pkl.dump(tree, f)

    cst.io.save_tree_h5(tree, fname)
    cst.pymol.tree_cartoon(tree, reference_structure, output_prefix + "pymol" + os.sep)
    cst.io.save_coarse_graining_h5(states_on_host, fname)

    mixture_state_on_host = jax.device_get(mixture_state)
    cst.io.save_VMM_h5(mixture_state_on_host, fname)

    return None


if __name__ == "__main__":
    main()
