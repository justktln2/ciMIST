"""
Functions for manipulating trajectories and converting between Cartesian and internal coordinates.
"""

from functools import partial
from typing import Callable, Optional, Tuple
from . import nerfax

from jax.tree_util import tree_map
from jax import Array, vmap, pmap
from jax.lax import scan
import jax.numpy as jnp
import mdtraj as md


def traj_to_internal(traj: md.Trajectory) -> dict:
    apo_scnet, apo_seq = traj_to_scnet(traj)
    apo_internal = scnet_cartesian_to_internal(apo_scnet, apo_seq)
    return apo_internal


def internal_to_traj(
    internal_coordinates: dict,
    reference_traj: md.Trajectory,
    superpose: Optional[bool] = True,
    mean_bond_length: Optional[bool] = True,
) -> md.Trajectory:
    _, _restrict, _to_list = nerfax.parser.get_scnet_loader_fns(reference_traj)
    cartesian = internal_to_cartesian_scnet(internal_coordinates, _to_list, mean_bond_length=mean_bond_length)
    traj = scnet_cartesian_to_traj(cartesian, reference_traj)
    if superpose:
        traj.superpose(_restrict(reference_traj))
    return traj


def restrict_to_scnet(traj: md.Trajectory):
    _, _restrict, _ = nerfax.parser.get_scnet_loader_fns(traj)
    return _restrict(traj)


def internal_to_cartesian_scnet(
    internal_coordinates: dict,
    scnet_to_list: Callable,
    mean_bond_length: Optional[bool] = True,
) -> Array:
    # switch to nerfax format
    internal_coordinates = vmap(convert_scnet_to_natural)(internal_coordinates)  # type: ignore
    # these are always the same
    cloud_mask = internal_coordinates["cloud_mask"][0]
    point_ref_mask = internal_coordinates["point_ref_mask"][0]
    angles_mask = internal_coordinates["angles_mask"]
    bond_mask = internal_coordinates["bond_mask"]
    if mean_bond_length:
        bond_length = bond_mask.mean(axis=0)
        cartesian = vmap(lambda A: nerfax.plugin.protein_fold(cloud_mask, point_ref_mask, A, bond_length))(angles_mask)
    else:
        cartesian = vmap(lambda A, l: nerfax.plugin.protein_fold(cloud_mask, point_ref_mask, A, l))(
            angles_mask, bond_mask
        )
    return cartesian


def scnet_cartesian_to_traj(
    cartesian: Array,
    reference_traj: md.Trajectory,
) -> md.Trajectory:
    _, _restrict, _to_list = nerfax.parser.get_scnet_loader_fns(reference_traj)
    # coordinates, properly formatted for mdtraj
    xyz = vmap(_to_list)(cartesian)
    # only need the first frame for topology
    topology = _restrict(reference_traj[0]).topology
    return md.Trajectory(xyz, topology)


def flatten_masks(internal: dict) -> Tuple[Array, Array]:
    def concat(x, y):
        return jnp.concatenate((x, y), axis=-1)

    angles = concat(internal["angles_mask"][:, 0, :, :], internal["angles_mask"][:, 1, :, :])
    angles = jnp.swapaxes(angles, 0, 1)

    cloud_mask = concat(internal["cloud_mask"][0], internal["cloud_mask"][0])
    # angles dimensions are residue, time, feature
    # cloud dimensionare are residue, feature
    return angles, cloud_mask


def traj_to_scnet(traj: md.Trajectory) -> Tuple[Array, str]:
    """
    convert mdtraj.Trajectory object to cartesian coordinates in sidechainnet
    format

    Parameters
    ----------
    traj : mdtraj.Trajectory
        DESCRIPTION.

    Returns
    -------
    coords : jax array of dimensions T x L x 14 x 3
        cartesian coordinates.
    seq : str
        the protein sequence

    """
    _parse_coords, _, _ = nerfax.parser.get_scnet_loader_fns(traj)
    if len(traj) == 1:
        coords = _parse_coords(traj.xyz[0])  # type: ignore
    else:
        coords = vmap(_parse_coords)(traj.xyz)
    seq = traj.topology.to_fasta()[0]  # type: ignore
    return coords, seq


# @partial(jit, static_argnames=("seq",))
def scnet_cartesian_to_internal(scnet_coords: Array, seq: str) -> dict:
    """


    Parameters
    ----------
    scnet_coords :  jax array of dimensions T x L x 14 x 3
        cartesian coordinates.
    seq : str
        the protein sequence.

    Returns
    -------
    internal_coords : dict of arrays
        keys are ('angles_mask', 'bond_mask', 'cloud_mask', 'point_ref_mask').
        shapes of arrays are T x 2 x L x 14

    """
    make_scaffolds = vmap(nerfax.parser.make_scaffolds, in_axes=(0, None))
    internal_coords = make_scaffolds(scnet_coords, seq)
    return vmap(convert_natural_to_scnet)(internal_coords)  # type: ignore


# -----------------------------------------------------------------------------
# adapted from
# https://github.com/PeptoneLtd/nerfax/blob/8d04e1a919c7a86ad9372917b12de3991a117cbe/nerfax/plugin.py#L7-L30
def roll_first_col_in_last_axis(x: Array, roll: int = 1) -> Array:
    """
    Roll the first column of the last axis by `roll` positions.
    """
    # Roll the first column of the last axis by `roll` positions
    # and concatenate with the rest of the array.
    return jnp.concatenate([jnp.roll(x[..., :1], roll, axis=-2), x[..., 1:]], axis=-1)


# Convert between scnet odd conventions and 'natural' (i.e. nerfax) ones.
def converter(input, roll):
    """
    if a dict, it's assumed to have `angles_mask` key which is altered.
    if an array, assumed to be the `angles_mask` array in mp_nerf to alter
    """

    def _converter(angles_mask):
        # Roll the first angle and dihedral round by 1 to match 'natural' syntax
        angles_mask = roll_first_col_in_last_axis(angles_mask, roll=roll)

        # Fix difference in how angle is specified
        angles, torsions = angles_mask
        # no conversion needed here because I don't care, but this is in nerfax code
        # angles = jnp.pi-angles # due to historical scnet reasons, the scnet angle is defined as pi-angle
        angles_mask = jnp.stack([angles, torsions])
        return angles_mask

    if isinstance(input, dict):
        # is scaffolds dict, where we fix angles mask
        return {**input, "angles_mask": _converter(input["angles_mask"])}
    else:
        # Is angles_mask tensor of shape (2,L,14)
        return _converter(input)


convert_scnet_to_natural = partial(converter, roll=1)
convert_natural_to_scnet = partial(converter, roll=-1)


def chunked_pmap(f, chunksize, *, batch_size=None):
    ## Adapted from https://github.com/Joshuaalbert/jaxns/blob/master/jaxns/internals/maps.py#L101
    def _f(*args, batch_size=batch_size, **kwargs):
        def queue(*args, **kwargs):
            """
            Distributes the computation in queues which are computed with scan.
            Args:
                *args:
            """

            def body(state, X):
                (args, kwargs) = X
                return state, f(*args, **kwargs)

            _, result = scan(body, (), (args, kwargs))
            return result

        if chunksize > 1:
            if batch_size is None:
                batch_size = args[0].shape[0] if len(args) > 0 else None
            assert batch_size is not None, "Couldn't get batch_size, please provide explicitly"
            remainder = batch_size % chunksize
            extra = (chunksize - remainder) % chunksize
            args = tree_map(lambda arg: _pad_extra(arg, chunksize), args)
            kwargs = tree_map(lambda arg: _pad_extra(arg, chunksize), kwargs)
            result = pmap(queue)(*args, **kwargs)
            result = tree_map(lambda arg: jnp.reshape(arg, (-1,) + arg.shape[2:]), result)
            if extra > 0:
                result = tree_map(lambda x: x[:-extra], result)
        else:
            result = queue(*args, **kwargs)
        return result

    _f.__doc__ = f.__doc__
    _f.__annotations__ = f.__annotations__
    return _f


def _pad_extra(arg, chunksize):
    ## Adapted from https://github.com/Joshuaalbert/jaxns/blob/master/jaxns/internals/maps.py#L101
    N = arg.shape[0]
    remainder = N % chunksize
    # print(N, remainder, chunksize)
    if (remainder != 0) and (N > chunksize):
        # only pad if not a zero remainder
        extra = (chunksize - remainder) % chunksize
        arg = jnp.concatenate([arg] + [arg[0:1]] * extra, axis=0)
        N = N + extra
    else:
        extra = 0
    arg = jnp.reshape(arg, (chunksize, N // chunksize) + arg.shape[1:])
    return arg
