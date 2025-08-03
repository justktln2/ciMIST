
"""
Author: Kaitlin
"""


import os
import argparse
import logging
from typing import List, NamedTuple
from functools import partial, wraps
import colabdesign.mpnn.ensemble_model as dmpnn

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
    description="""ProteinMPNN-MIST.
    Run maximum information spanning tree on a molecular dynamics ensemble using ProteinMPNN inverse folding to determine residue states.
    WARNING: EXPERIMENTAL.""")

    parser.add_argument("-t", "--trajectory", type=str,
    help=f"""The path to the trajectory file,
or to a directory that contains all trajectory files and nothing else.
Note that if a directory is supplied, all files in that directory must be valid molecular
dynamics trajectory files.""")
    parser.add_argument("-s", "--topology", type=str,
    help=f"""The path to the topology file.""")
    parser.add_argument("-o", "--output_prefix", 
    help=f"""The prefix for the output directory.""")
    parser.add_argument("--temperature", type=float, default=0.2,
    help="""Temperature parameter for ProteinMPNN""",
    )
    parser.add_argument("--weights", type=str,
    default="soluble", choices={"soluble", "original"},
    help="ProteinMPNN weights to use.")
    parser.add_argument("--dropout", type=float, default=0.0,
    help="""'dropout' argument for ProteinMPNN""")
    parser.add_argument("--temperature_mpnn", type=float, default=0.1,
    help="Sampling temperature for ProteinMPNN")
    parser.add_argument("--seed", type=int, default=0,
    help="""Random seed.""")
    parser.add_argument("--prior",
                        type=str,
                        default="haldane",
                        choices={"percs", "haldane", "jeffreys", "laplace"},
                        help="""Prior to use for residue entropy and pairwise mutual information estimation with the Dirichlet distribution.
Each prior corresponds to adding the same number of pseudocounts to each conformation.
Options are:
    -'haldane' : 0 pseudocounts (DEFAULT)
    -'percs' : 1/K pseudocounts, where K is the number of conformations
    -'jeffreys' : 1/2 pseudocounts
    -'laplace' : 1 pseudocount
    
Note that of these options, only 'haldane' and 'percs' add the same total number of pseudocounts to each distribution."""
                        )
    parser.add_argument("--mpnn_batch_size", type=int, default=500,
    help="""Batch size (in number of trajectory frames) to disatch to ProteinMPNN via jax.vmap. Should be set depending on available memory.
    DEFAULT: 500.""")
    return parser.parse_args()

if __name__ == "__main__":
    # some setup so we don't have to wait for imports to see if the inputs are bad
    args = parse_args()
    

import time
t0 = time.time()

import jax


import cimist as cst
import dill as pkl
from jax import jit, vmap, Array
import jax.numpy as jnp
import mdtraj as md



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
    # CONFIGURE LOGGING
    output_prefix = f"{args.output_prefix}_seed_{args.seed}" + os.sep
    if not os.path.exists(output_prefix):
        os.mkdir(output_prefix)

    logfile = output_prefix + "logging.txt"
    logging.basicConfig(filename=logfile, level=logging.INFO, filemode='w')
    logging.info("Began pipeline at " + str(today) + ".")

    reference_structure = md.load(args.topology)
    #if reference_structure.topology.n_chains > 1:
    #    raise NotImplementedError("mpnnMIST not yet implemented for multi-chain trajectories.")

    traj = cst.io.load_traj(args.trajectory, args.topology)

    dmpnn_model = dmpnn.mk_mpnn_ensemble_model(weights=args.weights, dropout=args.dropout, batch_size=args.mpnn_batch_size)
    dmpnn_model.prep_inputs(traj)

    
    samples = dmpnn_model.sample_minimal(temperature=args.temperature_mpnn)
    states = cst.ci.mpnn.samples_to_states(samples)    

    build_tree = time_and_message("Calculated MIs and inferred maximimum information spanning tree", cst.MIST.from_residue_states)
    residue_names = list([str(t) for t in traj.topology.residues])
    tree = build_tree(states, residue_names, prior=args.prior, uncertainty=True)
    fname = output_prefix + "RESULTS_mpnnMIST.h5"

    import shutil
    ANALYSIS_PATH = cst.__path__[0] + os.sep + "templates/analysis_template.ipynb"
    shutil.copy(ANALYSIS_PATH, output_prefix + "analysis_template.ipynb")
    logging.info("copied analysis template to output directory.")

    
    cst.pymol.tree_cartoon(tree, reference_structure, output_prefix + "pymol" + os.sep)
    cst.io.save_tree_h5(tree, fname)
    
    with open(fname.replace(".h5", ".pkl"), "wb") as f:
        pkl.dump(tree, f)

    with open(output_prefix + "mpnn_samples.pkl", "wb") as f:
        pkl.dump(samples, f)

    return None

if __name__ == "__main__":
    main()