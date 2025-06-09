#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX-compatible dataloader for streaming molecular dynamics trajectories.
"""
import mdtraj as md
import jax.numpy as jnp
from typing import NamedTuple, List, Iterator

import cimist as cst

class MDDataBatch(NamedTuple):
    """Represents one batch of processed trajectory data."""
    angles: jnp.ndarray

class MDStreamer:
    """
    Streams and processes molecular dynamics trajectories in batches without
    loading the entire file into memory.

    This class acts as a Python iterator, yielding batches of dihedral angles
    ready for use in JAX-based models.

    Parameters:
    ----------
    traj_path : str
        Path to the trajectory file or a directory of files.
    top_path : str
        Path to the topology file.
    batch_size : int, optional
        The number of frames to include in each batch, by default 1000.
    stride : int, optional
        The stride to use when reading frames, by default 1.
    """
    def __init__(self, traj_path: str, top_path: str, batch_size: int = 1000, stride: int = 1):
        self.traj_path = traj_path
        self.top_path = top_path
        self.batch_size = batch_size
        self.stride = stride

        # Load the topology and a reference frame for superposition.
        # This part assumes the first frame is a good reference.
        self.ref_traj = md.load_frame(self.traj_path, 0, top=self.top_path)
        self.ref_traj, self.residues = self._to_single_chain(self.ref_traj)

        # Get the number of residues and the cloud mask from the reference.
        # This will be constant for all batches.
        internal = cst.utils.traj_to_internal(self.ref_traj)
        _, self.cloud_mask = cst.utils.flatten_masks(internal)
        self.n_residues = self.cloud_mask.shape[0]

        # Create an iterator that loads the trajectory chunk by chunk.
        self._traj_iterator = md.iterload(
            self.traj_path,
            top=self.top_path,
            chunk=self.batch_size,
            stride=self.stride
        )

    def _to_single_chain(self, traj: md.Trajectory):
        """Processes the topology to ensure a single, continuous chain."""
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

    def __iter__(self) -> Iterator[MDDataBatch]:
        """Yields batches of processed dihedral angles."""
        for chunk in self._traj_iterator:
            # For each chunk of frames, perform the required processing.
            chunk.superpose(self.ref_traj)
            
            # This part needs to match the topology processing of the reference.
            processed_chunk, _ = self._to_single_chain(chunk)

            internal_coords = cst.utils.traj_to_internal(processed_chunk)
            angles, _ = cst.utils.flatten_masks(internal_coords)

            yield MDDataBatch(angles=jnp.asarray(angles))
