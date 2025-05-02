#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Maximum information spanning trees
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import norm

import mdtraj as md  # type: ignore


from itertools import combinations
from functools import reduce
from typing import List, Optional, NamedTuple

import jax
from jax import Array, jit, vmap
from jax.typing import ArrayLike
import jax.numpy as jnp


import cimist.entropy as ee
import cimist.pymol as pml
import cimist.ci.dbscan as dbs


PRIORS = ("haldane", "independent", "jeffreys")


class ResidueInteraction(NamedTuple):
    S_uv: ArrayLike
    S_u: ArrayLike
    S_v: ArrayLike
    I_uv: ArrayLike
    P_uv: Array
    S_uv_pos_mean: ArrayLike
    S_uv_se: ArrayLike


class MIST:
    @staticmethod
    def from_residue_states(
        residue_states: dbs.ResidueStates,
        names: List[str],
        prior: str = "haldane",
        uncertainty: bool = True,
    ):
        """
        Determine the maximum information (maximum likeihood) spanning tree from residue states.
        """

        states = residue_states.states_traj
        matrix_size = int(jnp.max(states) + 1)

        counts_array_size = matrix_size**2

        def pairwise_interaction(s1: Array, s2: Array):
            init_matrix = jnp.empty((matrix_size + 1, matrix_size + 1))
            state_pairs = jnp.stack([s1, s2])
            pair_counts = jnp.unique(
                state_pairs,
                axis=1,
                return_counts=True,
                size=counts_array_size,
                fill_value=-1,
            )
            counts_matrix = init_matrix.at[pair_counts[0][0], pair_counts[0][1]].set(
                pair_counts[1]
            )
            counts_matrix = counts_matrix[
                :-1, :-1
            ]  # drop the row and column we added to account for -1 fill value
            P = counts_matrix / counts_matrix.sum()

            S_uv = ee.S_mle(P.flatten())
            S_u = ee.S_mle(P.sum(axis=1))  # axis 1 sums out p(v)
            S_v = ee.S_mle(P.sum(axis=0))
            S_uv_pos_mean, S_uv_pos_se = jnp.nan, jnp.nan
            I_uv = jnp.subtract(jnp.add(S_u, S_v), S_uv)
            return ResidueInteraction(
                S_uv, S_u, S_v, I_uv, P, S_uv_pos_mean, S_uv_pos_se
            )

        pairwise_interaction_ = jit(pairwise_interaction)
        pairs = jnp.array(list(combinations(list(range(len(names))), 2)))
        name_pairs = [(names[i], names[j]) for (i, j) in pairs]
        compute_pairs = vmap(
            lambda x: pairwise_interaction_(states[x[0]], states[x[1]])
        )
        interactions = compute_pairs(pairs)

        MI_graph: nx.Graph = nx.Graph()
        for u, ctz, K in zip(
            names,
            residue_states.state_counts,
            residue_states.n_states,  # type: ignore
        ):
            p = ctz / ctz.sum()

            S_pos_mean, S_std = jnp.nan, jnp.nan

            S_mle = ee.S_mle(p)
            bias = S_mle - S_pos_mean
            S_se = S_std
            MI_graph.add_node(
                u,
                S=S_mle,
                S_pos_mean=S_pos_mean,
                S_se=S_se,
                p=ctz / ctz.sum(),
                N_states=K,
            )

        for (u, v), I, P, S_uv, S_se_uv in zip(
            name_pairs,
            interactions.I_uv,
            interactions.P_uv,
            interactions.S_uv_pos_mean,
            interactions.S_uv_se,
        ):
            S_u, S_v = MI_graph.nodes[u]["S_pos_mean"], MI_graph.nodes[v]["S_pos_mean"]

            S_u_se, S_v_se = MI_graph.nodes[u]["S_se"], MI_graph.nodes[v]["S_se"]
            I_uv_pos_mean = S_u + S_v - S_uv

            bias = I - I_uv_pos_mean
            variance = S_se_uv**2 + S_u_se**2 + S_v_se**2
            I_se = jnp.sqrt(variance)
            MI_graph.add_edge(
                u, v, P=P, I=I, axes=(u, v), I_pos_mean=I_uv_pos_mean, I_se=I_se
            )

        tree = MIST(
            MI_graph,
            calculate_marginals=False,
            N_obs=ctz.sum(),
            uncertainty=uncertainty,
            prior=prior,
        )
        return tree

    def __init__(
        self,
        MI_graph: nx.Graph,
        N_obs: int,
        tree: Optional[nx.Graph] = None,
        calculate_marginals: Optional[bool] = False,
        uncertainty: bool = False,
        prior: str = "haldane",
        update_posterior: bool = False,
    ):
        self.MI_graph = MI_graph
        if tree is None:
            self.fit()
        else:
            self.T = tree

        self.N_obs = N_obs
        self.prior = prior
        self.nodes = self.T.nodes
        self.edges = self.T.edges
        self._implied_marginals_set = False

        for u in self.nodes():
            # counts formatting in residue states is currently off
            # causes problems in implied pairwise marginals without this
            neighbors = list(nx.neighbors(self.T, u))
            if len(neighbors) > 0:
                v = neighbors[0]
                P = self.T.edges[u, v]["P"]
                axes = self.T.edges[u, v]["axes"]
                if u != axes[0]:
                    P = P.T
                self.T.nodes[u]["p"] = P.sum(axis=1)
                self.MI_graph.nodes[u]["p"] = P.sum(axis=1)

            # finally, set the posterior mean entropy
            if update_posterior or jnp.isnan(
                self.T.nodes[u].get("S_pos_mean", jnp.nan)
            ):
                _set_S_posterior(self, u, prior=self.prior, uncertainty=uncertainty)

        # Set posterior mean mutual informations
        # vmap opporunity, but profiling suggested very small marginal benefit
        for u, v in self.T.edges():
            if update_posterior or jnp.isnan(
                self.T.edges[u, v].get("I_pos_mean", jnp.nan)
            ):
                _set_I_posterior(self, u, v, uncertainty=uncertainty, prior=self.prior)
                self.MI_graph.edges[u, v]["I_pos_mean"] = self.T.edges[u, v][
                    "I_pos_mean"
                ]
                self.MI_graph.edges[u, v]["I_se"] = self.T.edges[u, v]["I_se"]

        if calculate_marginals:
            self._compute_MI_CL()
        _, _, self.log_prob, self.sample = get_log_prob_and_sampling_fns(self)

        if uncertainty:
            self.entropy_se, self.entropy_mle_bias = compute_error(self)
        else:
            self.entropy_se, self.entropy_mle_bias = jnp.nan, jnp.nan
            self.est_bias = jnp.nan
            self.structural_variance = jnp.nan
            self.counts_variance = jnp.nan

    def fit(self) -> None:
        """
        Calculate the maximum informational spanning tree from the mutual
        information network.

        Returns
        -------
        None
        """
        self.T = nx.maximum_spanning_tree(self.MI_graph, weight="I")
        for n in self.MI_graph.nodes():
            self.T.nodes[n]["S"] = self.MI_graph.nodes[n]["S"]
        return None

    def entropy_ci(self, ci: float = 0.95) -> Array:
        """
        Calculate a 100*ci percent credible interval for the entropy (in nats)
        of the tree distribution.
        """

        Z_alpha = abs(norm.ppf((1 - ci) / 2))
        credibility_interval = (
            self.entropy() + jnp.array([-1, 1]) * Z_alpha * self.entropy_se
        )
        return credibility_interval

    def __getitem__(self, n):
        return self.T.__getitem__(n)

    @property
    def resnames(self) -> List:
        """
        Notice that here we actually do want to use property to avoid
        getter/setter stuff.
        """
        return list(self.T.nodes())

    def residue_entropies(self) -> pd.Series:
        """
        Compute the residue marginal entropies (in nats).
        """
        return pd.Series(
            np.array([self.T.nodes[n]["S_pos_mean"] for n in self.resnames]),
            index=self.resnames,
        )

    def residue_sum_MIs(self) -> pd.Series:
        """
        Compute each residue's summed mutual information with all neighbors (in nats).
        """
        return pd.Series(dict(nx.degree(self.T, weight="I_pos_mean")))

    def sum_marginal_entropy(self) -> float:
        """
        Compute the sum of all residue marginal entropies (in nats).
        """
        return np.sum(self.residue_entropies())

    def sum_MIs(self):
        """
        Compute the sum of mutual informations over all tree edges (in nats).
        """
        return self.T.size(weight="I_pos_mean")

    def entropy(self):
        """
        Compute the entropy of the tree distribution.
        """
        return self.sum_marginal_entropy() - self.sum_MIs()

    def information_matrix(self, resis=None, kind="empirical", diag=False):
        """


        Parameters
        ----------
        resis : TYPE, optional
            DESCRIPTION. The default is None.
        kind : string, optional
            The default is "hybrid". Options are "hybrid", "empirical", and "chow_liu".
        diag : bool, optional
            Whether to plot the entropies (self-informations) on the diagonal. The default is False.

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if kind == "hybrid":
            udag = np.triu(self.empirical_information_matrix())
            ldag = np.tril(self.chow_liu_information_matrix())
            return udag + ldag
        elif kind == "chow_liu":
            return self.chow_liu_information_matrix()
        elif kind == "empirical":
            return self.empirical_information_matrix()
        else:
            raise Exception(
                "argument kind must be 'hybrid', 'chow_liu' , or 'empirical'."
            )

    def empirical_information_matrix(self):
        return np.array(
            nx.adjacency_matrix(self.MI_graph, weight="I_pos_mean").todense()
        )

    def chow_liu_information_matrix(self):
        return np.array(nx.adjacency_matrix(self.MI_graph, weight="I_CL").todense())

    def tree_MIs(self):
        return pd.Series(
            [self.T.edges[e]["I_pos_mean"] for e in self.T.edges()],
            index=self.T.edges(),
        )  # type: ignore

    def trim(self, nodelist):
        paths = [
            list(nx.utils.pairwise(nx.shortest_path(self.T, u, v)))
            for (u, v) in combinations(nodelist, 2)
        ]
        return self.T.edge_subgraph(reduce(lambda x, y: list(set(x + y)), paths))

    def to_pymol(
        self,
        protein_structure: md.Trajectory,
        path: str,
        edge_cmap="turbid",
        edge_vmin=0,
        edge_vmax=1,
        min_MI=0.0,
        node_cmap="algae",
        node_vmin=0,
        node_vmax=3.5,
        entropy_decay_factor=1,
        base_stick_radius=1,
        base_sphere_radius=1,
        min_stick_radius=0.1,
        cbar_figsize=(3, 1),
    ):
        """
        Create files for visualization of the tree in PyMOL.

        Required Parameters
        -------------------
        protein_structure : the protein structure to be saved for visualization in PyMOL
        path : the path to the directory in which to create the files.

        Returns
        -------
        None.
        Creates a directory with files 'tree.pdb', 'structure.pdb', 'draw_tree.pml',
        and 'colorbar.png'. The tree is visualized by opening PyMOL and running
        draw_tree.pml.

        Notes
        -----
        Available colormaps are those in the Python libraries cmasher and cmocean.
        You will need to have installed these in PyMOL.

        See also
        --------
        https://matplotlib.org/cmocean/
        https://cmasher.readthedocs.io/user/introduction.html#colormap-overview
        """
        pml.tree_cartoon(
            self,
            protein_structure,
            path,
            edge_cmap=edge_cmap,
            edge_vmin=edge_vmin,
            edge_vmax=edge_vmax,
            node_cmap=node_cmap,
            node_vmin=node_vmin,
            node_vmax=node_vmax,
            cbar_figsize=cbar_figsize,
            min_MI=min_MI,
            base_stick_radius=base_stick_radius,
            base_sphere_radius=base_sphere_radius,
            entropy_decay_factor=entropy_decay_factor,
            min_stick_radius=min_stick_radius,
        )
        print(f"Wrote files for PyMOL visualization to directory {path}.", flush=True)

    def _compute_implied_pairwise_marginals(self):
        self._implied_marginals_set = True
        M = {(u, v): joint_distribution(self, u, v) for (u, v) in self.MI_graph.edges()}
        for u, v in M.keys():
            self.MI_graph[u][v]["P_CL"] = M[(u, v)]

    def _compute_MI_CL(self):
        if not self._implied_marginals_set:
            self._compute_implied_pairwise_marginals()
        for u, v in self.MI_graph.edges():
            H_u = self.nodes[u]["S"]
            H_v = self.nodes[v]["S"]
            H_uv = ee.S_mle(self.MI_graph[u][v]["P_CL"].flatten())
            self.MI_graph[u][v]["I_CL"] = H_u + H_v - H_uv
        return None

    def refit_to_subset(self, nodelist, uncertainty=False):
        MI = self.MI_graph.subgraph(nodelist)
        raise NotImplementedError
        # return MIST(MI, uncertainty=uncertainty) missing Nobs

    def update_prior(self, prior: str):
        return MIST(
            self.MI_graph,
            self.N_obs,
            prior=prior,
            uncertainty=True,
            update_posterior=True,
        )


def entropy(p):
    return ee.S_mle(p)


def significance_test_edges(tree, fdr=0.01):
    from scipy.stats import expon
    from statsmodels.stats.multitest import multipletests  # type: ignore

    def get_upper_triu(mat):
        ix = np.triu_indices_from(mat, k=1)
        return mat[ix].flatten()

    mis = get_upper_triu(tree.empirical_information_matrix())
    fitted = expon(*(expon.fit(mis)))
    N_hypotheses = len(tree.nodes()) - 1
    adj_pval = lambda x: 1 - fitted.cdf(x)
    mis_tree = np.array([MI for (_, _, MI) in tree.edges(data="I")])
    reject, pvals_corrected, sidak_alphac, bonf_alphac = multipletests(
        adj_pval(mis_tree), method="fdr_bh", alpha=fdr
    )
    mis_signif = mis_tree[reject]
    edges_signif = [(u, v) for (r, (u, v)) in zip(reject, tree.edges()) if r]
    return edges_signif, mis_signif


def joint_distribution(T, u, v):
    path_nodes = nx.shortest_path(T.T, source=u, target=v)
    path_edges = list(nx.utils.pairwise(path_nodes))

    def _transfer_matrix(T, u, v):
        axes = T[u][v]["axes"]
        if axes == (u, v):
            P_uv = T[u][v]["P"]
        elif axes == (v, u):
            P_uv = T[u][v]["P"].T
        else:
            raise Exception(
                f"Axes specification error, received edges {(u,v)} and got axes {axes}"
            )
        replace_inf_with_zero = lambda x: np.where(np.isinf(x), 0, x)
        P_ui = np.diag(replace_inf_with_zero(1.0 / T.nodes[u]["p"]))
        return np.dot(P_ui, P_uv)

    P_u = np.diag(T.nodes[u]["p"])
    P_uv = np.linalg.multi_dot(
        [P_u] + [_transfer_matrix(T.T, u, v) for (u, v) in path_edges]
    )
    return P_uv


def compute_error(tree):
    node_variance = jnp.sum(jnp.array([se**2 for (_, se) in tree.nodes(data="S_se")]))
    edge_variance = jnp.sum(
        jnp.array([se**2 for (_, _, se) in tree.edges(data="I_se")])
    )

    node_bias = jnp.sum(
        jnp.array(
            [tree.nodes[u]["S"] - tree.nodes[u]["S_pos_mean"] for u in tree.nodes()]
        )
    )
    edge_bias = jnp.sum(
        jnp.array(
            [
                tree.edges[u, v]["I"] - tree.edges[u, v]["I_pos_mean"]
                for u, v in tree.edges()
            ]
        )
    )

    bias = node_bias - edge_bias

    sse_counts = node_variance + edge_variance + bias**2

    S_jack, S_jack_contrib, contribution_se = residue_jackknife(tree)

    tree.S_jack_entropies = S_jack
    tree.S_jack_contrib = S_jack_contrib
    tree.est_bias = bias
    tree.structural_variance = jnp.var(S_jack)
    tree.counts_variance = node_variance + edge_variance

    for r in contribution_se.index:
        tree.nodes[r]["S_contribution_se"] = contribution_se.loc[r]

    rmse = jnp.sqrt(tree.structural_variance + tree.counts_variance + bias**2)

    tree.rmse = rmse
    return rmse, bias


def get_log_prob_and_sampling_fns(tree, root=None):
    """
    returns a jitted function that calculates the log-probability of a barcode from the tree model
    return transfer_matrices, edgelist, log_prob_fn, sample_fn
    """
    if root is None:
        root = list(tree.nodes())[0]

    # What we need to do here is make a function that has a transfer tensor
    # and a way of mapping the barcode indices to indices of the transfer tensor
    nodes = list(tree.nodes())
    root_ix = nodes.index(root)  # this is index of the root in the barcode
    root_log_p = jnp.log(tree.nodes[root]["p"])

    # gets all directed edges from root to leaves in an order where parents precede
    # children and forms a stack of transfer matrices in that order
    Paths = get_paths_to_leaves(tree, root)
    aggregate_paths = lambda p1, p2: p1 + [e for e in p2 if e not in p1]
    edgelist = list(reduce(aggregate_paths, Paths))
    transfer_matrices = [
        transfer_matrix(tree.T, u, v) for (u, v) in edgelist
    ]  # transfer matrices

    # i gets the index of the edge in the product over conditional probabilities
    # nodes.index(u), nodes.index(v) gets the index of each node appearing that edge for use in the barcode
    edge_ixs = [
        (i, nodes.index(u), nodes.index(v)) for i, (u, v) in enumerate(edgelist)
    ]

    @jit
    def log_prob_fn(barcode):
        root_term = root_log_p[barcode[root_ix]]
        sum_couplings = jnp.sum(
            jnp.stack(
                [
                    jnp.log(
                        transfer_matrices[i][barcode[u], barcode[v]]
                    )  # barcode[u], barcode[v] gets the
                    for (i, u, v) in edge_ixs
                ]
            )
        )
        return root_term + sum_couplings

    N = len(nodes)

    @jit
    def sample_fn(key):
        keys = jax.random.split(key, N)
        barcode = jnp.empty(N, dtype=int)
        barcode = barcode.at[root_ix].set(jax.random.categorical(keys[-1], root_log_p))
        for i, u, v in edge_ixs:
            logits = jnp.log(transfer_matrices[i][barcode[u]])
            barcode = barcode.at[v].set(jax.random.categorical(keys[i], logits))
        return barcode

    return transfer_matrices, edgelist, log_prob_fn, sample_fn


def transfer_matrix(T, u, v, fix_v=None, pad2dim=100):
    """
    For a tree model, calculate the conditional probability (transfer)
    matrix with entries P_{uv} = P(X_v=x_v | X_u = x_u)

    There is an optional argument, fix_v.
    If fix_v is not None, it must be an integer, and it sets
    $P_{uv} = \delta_{vv'}$ for v' = fix_v; i.e. it sets the transfer
    matrix to transition to state $v$ from every state with probability one.
    """

    # Return a matrix with only one column of ones if
    # option fix_v is set
    if fix_v is not None:
        P_uv = jnp.zeros((pad2dim, pad2dim))
        P_uv = P_uv.at[:, fix_v].set(1.0)
        return P_uv

    # transpose the matrix if needed
    axes = T[u][v]["axes"]
    if axes == (u, v):
        P_uv = jnp.array(T[u][v]["P"])
    elif axes == (v, u):
        P_uv = jnp.array(T[u][v]["P"].T)
    else:
        raise Exception(
            f"Axes specification error, received edges {(u,v)} and got axes {axes}"
        )

    # Calculate conditional probability (transfer) matrix
    # P(X_v=x_v | X_u = x_u)
    replace_inf_with_zero = lambda x: jnp.where(jnp.isinf(x), 0, x)
    P_ui = replace_inf_with_zero(1.0 / jnp.array(T.nodes[u]["p"]))
    # below is equivalent and saves memory
    # tmat = jnp.dot(P_ui, P_uv)
    tmat = P_uv * P_ui[:, jnp.newaxis]  # type: ignore
    pad_u, pad_v = [(0, max(pad2dim - d, 0)) for d in tmat.shape]
    return jnp.pad(tmat, (pad_u, pad_v))


def get_paths_to_leaves(tree, root=None):
    # returns a generator of zips
    if type(tree) != nx.Graph:
        T = tree.T
    else:
        T = tree
    if root is None:
        root = list(tree.nodes())[0]
    # L is the set of leaves
    L = [u for u in tree.nodes() if T.degree(u) == 1 and u != root]
    return list(list(nx.utils.pairwise(nx.shortest_path(T, root, l))) for l in L)


def make_new_tree(tree, edges, ixs, transfer_tensor, marginals):
    """
    Return a new tree that updates the pairwise marginals in tree to agree with the
    single variable marginals in marginals
    """
    G = nx.Graph()
    for i, n in enumerate(tree.nodes()):
        G.add_node(n, p=marginals[i])
    for (n, i, j), (u, v) in zip(ixs, edges):
        P = jnp.diag(marginals[i]).dot(transfer_tensor[n])
        G.add_edge(u, v, P=P, axes=(u, v))
    return G


def edge_knockout_P(tree, u, v):
    """
    Calculate the joint probabilities for nodes u and v from the maximum information
    spanning tree that does not include u or v
    """

    if (u, v) not in tree.T.edges():
        return tree.MI_graph[u][v]["P_CL"]
    else:
        G_new = tree.MI_graph.copy()
        G_new[u][v]["I"] = 0
        T_new = MIST(G_new, uncertainty=False)  # type: ignore
        P_CL = joint_distribution(T_new, u, v)
        if P_CL.shape != tree.T[u][v]["P"].shape:
            return P_CL.T
        else:
            return P_CL


def compute_all_MI_CL(tree, verbose=False):
    if verbose:
        print("Calculating implied MIs")
    tree._compute_MI_CL()
    for u, v in tree.edges():
        if verbose:
            print(u, v)
        P_CL = edge_knockout_P(tree, u, v)
        S_uv_CL = ee.S_mle(P_CL.flatten())
        MI_CL = tree.T.nodes[u]["S"] + tree.T.nodes[v]["S"] - S_uv_CL
        tree.MI_graph[u][v]["I_CL"] = MI_CL
        tree.MI_graph[u][v]["P_CL"] = P_CL
        tree.T[u][v]["I_CL"] = MI_CL
        tree.T[u][v]["P_CL"] = P_CL
    return tree


def residue_jackknife(tree):
    entropies = []
    contributions = []
    for n in tree.nodes():
        other_nodes = [n_ for n_ in tree.MI_graph.nodes() if n_ != n]
        G = nx.subgraph(tree.MI_graph, other_nodes)
        tree_resamp = MIST(
            G,
            N_obs=tree.N_obs,
            calculate_marginals=False,
            uncertainty=False,
            prior=tree.prior,
        )
        entropies.append(tree_resamp.entropy())
        contributions.append(
            tree_resamp.residue_entropies() - 0.5 * tree_resamp.residue_sum_MIs()
        )
    contributions_df = pd.concat(contributions, axis=1).map(float)
    contribution_se = contributions_df.std(axis=1, skipna=True)
    return jnp.array(entropies), contributions_df, contribution_se


def _set_S_posterior(tree: MIST, u, prior: str = "haldane", uncertainty: bool = False):
    N_u = tree.N_obs * tree.nodes[u]["p"]
    if prior == "haldane" or prior == "independent":
        pc = jnp.zeros_like(N_u)
    elif prior == "percs":
        occupied_states = jnp.where(N_u > 1e-9, 1, 0)
        pc = occupied_states / occupied_states.sum()
    elif prior == "jeffreys":
        pc = jnp.where(N_u > 1e-9, 0.5, 0.0)
    elif prior == "laplace":
        pc = jnp.where(N_u > 1e-9, 1.0, 0.0)

    alpha = N_u + pc
    if uncertainty:
        mu, se = ee.S_posterior_mean_std(alpha.flatten())
        tree.T.nodes[u]["S_pos_mean"] = mu
        tree.T.nodes[u]["S_se"] = se
        tree.T.nodes[u]["bias"] = tree.nodes[u]["S"] - mu
        tree.T.nodes[u]["variance"] = se**2

        tree.MI_graph.nodes[u]["S_pos_mean"] = mu
        tree.MI_graph.nodes[u]["S_se"] = se
        tree.MI_graph.nodes[u]["bias"] = tree.nodes[u]["S"] - mu
        tree.MI_graph.nodes[u]["variance"] = se**2
    else:
        mu = ee.S_posterior_mean(alpha.flatten())
        tree.T.nodes[u]["S_pos_mean"] = mu
        tree.T.nodes[u]["S_se"] = jnp.nan
        tree.T.nodes[u]["bias"] = tree.nodes[u]["S"] - mu
        tree.T.nodes[u]["variance"] = jnp.nan

        tree.MI_graph.nodes[u]["S_pos_mean"] = mu
        tree.MI_graph.nodes[u]["S_se"] = jnp.nan
        tree.MI_graph.nodes[u]["bias"] = tree.nodes[u]["S"] - mu
        tree.MI_graph.nodes[u]["variance"] = jnp.nan


def _set_I_posterior(
    tree: MIST, u, v, prior: str = "haldane", uncertainty: bool = False
):
    N_uv = tree.N_obs * tree.edges[u, v]["P"]
    p_u = tree.nodes[u]["p"]
    p_v = tree.nodes[v]["p"]

    if prior == "haldane":
        N_uv_pos = N_uv
    elif prior == "percs":
        # use the Percs prior
        # the dirichlet posterior  N_uv_pos just the empirical
        # counts N_uv + 1/K, where K is the number of possible states
        occupied_states = jnp.outer(p_u > 1e-9, p_v > 1e-9)
        pseudocount = occupied_states / occupied_states.sum()
        N_uv_pos = N_uv + pseudocount
    elif prior == "jeffreys":
        # use the Jeffreys prior
        # the dirichlet posterior  N_uv_pos just the empirical
        # counts N_uv + 1/2
        occupied_states = jnp.outer(p_u > 1e-9, p_v > 1e-9)
        pseudocount = 0.5 * occupied_states
        N_uv_pos = N_uv + pseudocount
    elif prior == "laplace":
        # the dirichlet posterior  N_uv_pos just the empirical
        # counts N_uv + one pseudocount from the product of the
        # marginal distributions of u and v
        occupied_states = jnp.outer(p_u > 1e-9, p_v > 1e-9)
        pseudocount = 1.0 * occupied_states
        N_uv_pos = N_uv + pseudocount
    elif prior == "independent":
        # the dirichlet posterior  N_uv_pos just the empirical
        # counts N_uv + one pseudocount from the product of the
        # marginal distributions of u and v
        N_uv_pos = N_uv + jnp.outer(p_u, p_v)

    if uncertainty:
        mu, se = ee.S_posterior_mean_std(N_uv_pos.flatten())
        I_mu = tree.nodes[u]["S_pos_mean"] + tree.nodes[v]["S_pos_mean"] - mu
        variance = tree.nodes[u]["S_se"] ** 2 + tree.nodes[v]["S_se"] ** 2 + se**2
        bias = tree.edges[u, v]["I"] - I_mu
        tree.T.edges[u, v]["I_pos_mean"] = I_mu
        tree.T.edges[u, v]["bias"] = bias
        tree.T.edges[u, v]["variance"] = variance
        tree.T.edges[u, v]["I_se"] = jnp.sqrt(variance)
    else:
        S_mu = ee.S_posterior_mean(N_uv_pos.flatten())
        I_mu = tree.nodes[u]["S_pos_mean"] + tree.nodes[v]["S_pos_mean"] - S_mu
        tree.T.edges[u, v]["I_pos_mean"] = I_mu
        bias = tree.edges[u, v]["I"] - I_mu
        tree.T.edges[u, v]["bias"] = bias
        tree.T.edges[u, v]["variance"] = jnp.nan
        tree.T.edges[u, v]["I_se"] = jnp.nan
    return None
