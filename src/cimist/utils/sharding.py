#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX sharding utilities for CIMIST.
"""
import jax
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental import mesh_utils
import jax.numpy as jnp

def setup_mesh():
    """
    Creates a 1D mesh of all available JAX devices.

    This function identifies all available devices (GPUs or TPU cores) and arranges
    them into a one-dimensional mesh. This is ideal for data parallelism, where
    we split our data along one axis across the devices.

    Returns:
    -------
    jax.sharding.Mesh
        A 1D mesh object mapping to all available devices.
    """
    # Get all visible JAX devices.
    devices = jax.devices()
    
    # Create a 1D mesh, e.g., for 8 devices, it's a mesh of shape (8,).
    device_mesh = mesh_utils.create_device_mesh((len(devices),))
    
    # Create the Mesh object with a logical axis name 'data'.
    mesh = Mesh(device_mesh, axis_names=('data',))
    return mesh

def shard_array(mesh: Mesh, arr: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """
    Shards a JAX array along a specified axis across the device mesh.

    Parameters:
    ----------
    mesh : jax.sharding.Mesh
        The device mesh to shard the array across.
    arr : jnp.ndarray
        The JAX array to be sharded.
    axis : int, optional
        The axis of the array to shard, by default 0.

    Returns:
    -------
    jnp.ndarray
        A sharded JAX array (GlobalDeviceArray).
    """
    # Create a PartitionSpec. For an array of shape (A, B, C) and axis=0, this
    # will be PartitionSpec('data', None, None), meaning the first axis is sharded
    # across the 'data' mesh axis, and the other axes are not sharded (replicated).
    sharding_spec = [None] * arr.ndim
    sharding_spec[axis] = 'data'
    sharding = NamedSharding(mesh, PartitionSpec(*sharding_spec))
    
    # Place the array on the devices with the specified sharding.
    return jax.device_put(arr, sharding)

def replicate_array(mesh: Mesh, arr: jnp.ndarray) -> jnp.ndarray:
    """
    Replicates a JAX array across all devices in the mesh.

    This is useful for smaller arrays that every device needs to access.

    Parameters:
    ----------
    mesh : jax.sharding.Mesh
        The device mesh to replicate the array across.
    arr : jnp.ndarray
        The JAX array to be replicated.

    Returns:
    -------
    jnp.ndarray
        A replicated sharded JAX array.
    """
    # An empty PartitionSpec signifies replication.
    sharding = NamedSharding(mesh, PartitionSpec())
    return jax.device_put(arr, sharding)
