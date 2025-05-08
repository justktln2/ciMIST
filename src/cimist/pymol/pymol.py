import os
import numpy as np
import matplotlib as mpl
import cmocean as cmo  # type: ignore
import cmasher as cm  # type: ignore
import mdtraj as md  # type: ignore
import matplotlib.pyplot as plt
import dill as pkl  # type: ignore

CMAPS = {k: v for k, v in cmo.cm.__dict__.items() if "LinearSegmentedColormap" in str(type(v))}
CMAPS.update({k: v for k, v in cm.__dict__.items() if "ListedColormap" in str(type(v))})


def tree_cartoon(
    tree,
    protein_structure,
    savedir,
    edge_cmap="turbid",
    edge_vmin=0,
    edge_vmax=1,
    node_vmin=0,
    node_vmax=3.5,
    min_MI=0.2,
    base_stick_radius=1,
    base_sphere_radius=1,
    entropy_decay_factor=1,
    node_cmap="algae",
    cbar_figsize=(3, 1),
    min_stick_radius=0.1,
):
    if not os.path.exists(savedir):
        os.mkdir(savedir)

    if isinstance(edge_cmap, str):
        edge_cmap = CMAPS[edge_cmap]

    bfactors = np.exp(-entropy_decay_factor * tree.residue_entropies().values)
    bfactors_structure = np.zeros(protein_structure.xyz.shape[1])
    bfactors_structure[protein_structure.topology.select("name CA")] = bfactors
    protein_structure.save_pdb(savedir + os.sep + "structure.pdb", bfactors=bfactors_structure)

    # CREATE PDB FILE WITH ONLY ALPHA CARBONS
    CA_only = protein_structure.atom_slice(protein_structure.topology.select("name CA"))
    CA_only_df = CA_only.topology.to_dataframe()[0]
    CA_only_df["serial"] = CA_only_df.index + 1

    # atoms have to be renamed as simple carbons, else molecular visualization software
    # will connect them with bonds
    # CA_only_df["name"] = "C"
    CA_only = md.Trajectory(CA_only.xyz, md.Topology.from_dataframe(CA_only_df))

    # SAVE TREE EDGES AS CONNECT RECORDS
    # connect_records = []

    bond_creation_lines = [5 * "\n" + "#MAKE BONDS FOR TREE EDGES\n"]
    set_bond_radius_lines = [5 * "\n" + "#TREE EDGE RADIUS SETTINGS\n"]
    norm = mpl.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax)  # type: ignore

    set_color_lines = [5 * "\n", "#COLOR DEFINITIONS\n"]
    set_bond_color_lines = [5 * "\n", "#BOND COLOR SETTINGS\n"]

    set_q_lines = [5 * "\n", "#q-values set to entropy for spectrum\n"]

    reslist = list(tree.nodes())
    for u, v in tree.edges():
        # atom indices
        a1 = reslist.index(u) + 1
        a2 = reslist.index(v) + 1
        # connect_records.append(f"CONECT {a1} {a2}\n")

        # residue numbers
        # r1 = CA_only_df.loc[reslist.index(u), "resSeq"]
        # r2 = CA_only_df.loc[reslist.index(v), "resSeq"]
        bond_creation_lines.append(f"cmd.bond('tree and id {a1}', 'tree and id {a2}')\n")
        # define color for tree edge
        color_name = f"I_{a1}_{a2}"
        I = tree.edges[u, v]["I_pos_mean"]
        RGB = edge_cmap(norm(I))[:-1]
        RGB = tuple(np.format_float_positional(x, 16) for x in RGB)

        set_bond_color_lines.append(
            f"cmd.set_bond('stick_color', '{color_name}', 'tree and id {a1}', 'tree and id {a2}')\n"
        )

        distance = np.squeeze(md.compute_distances(CA_only, [[a1 - 1, a2 - 1]]))
        if I > min_MI:
            radius = base_sphere_radius * np.sqrt(I / (np.pi * distance))
        else:
            radius = 0
        set_bond_radius_lines.append(
            f"cmd.set_bond('stick_radius', {max(radius, min_stick_radius)}, 'tree and id {a1}', 'tree and id {a2}')\n"
        )
        # set_bond_color_lines.append(
        #    f"cmd.set_bond('stick_color', '{color_name}', 'tree and id {a2}', 'tree and id {a1}')\n"
        # )

        set_color_lines.append(f"cmd.set_color('{color_name}', {RGB})\n")

    for n, r in zip(tree.nodes(), CA_only.topology.residues):
        entropy_ = tree.nodes[n]["S_pos_mean"]
        set_q_lines.append(f"cmd.alter('resi {r.resSeq} and name CA', 'q={entropy_}')\n")

        set_q_lines.append(
            f"cmd.alter('resi {r.resSeq} and name CA and tree',\
            'vdw={base_sphere_radius * float(np.power(entropy_, 1 / 3))}')\n"
        )

    tree_file = f"{savedir}" + os.sep + "tree.pdb"
    CA_only.save_pdb(tree_file, bfactors=bfactors)

    # with open(tree_file, "r") as f:
    #    pdblines = f.readlines()[:-1]

    pmlfname = savedir + os.sep + "draw_tree.pml"
    with open(pmlfname, "w") as f:
        load_lines = ["load structure.pdb\n", "load tree.pdb\n", "hide cartoon, tree"]

        pml_lines = load_lines + bond_creation_lines + set_color_lines + set_bond_color_lines + set_bond_radius_lines
        pml_lines += set_q_lines
        pml_lines += [
            "set cartoon_cylindrical_helices, 1\n",
            f"spectrum q, {node_cmap}, name CA, {node_vmin}, {node_vmax}\n",
        ]
        f.writelines(
            pml_lines
            + [
                "show sticks, tree\n",
                "show spheres, tree\n",
                "set cartoon_transparency, 0.6\n",
                "set sphere_transparency, 0.3, all\n",
                "set cartoon_putty_radius, 0.25\n",
                "set cartoon_putty_transform, 4\n",
                "set cartoon_putty_scale_power, 1.\n",
                "set cartoon_putty_scale_min, -1\n",
                "set cartoon_putty_scale_max, -1\n",
                "orient\n",
                "zoom visible, complete=1\n",
            ],
        )

    # CREATE THE COLORBAR
    fig, ax = plt.subplots(2, 1, figsize=cbar_figsize, layout="constrained")
    norm_mi = mpl.colors.Normalize(vmin=edge_vmin, vmax=edge_vmax)
    norm_s = mpl.colors.Normalize(vmin=node_vmin, vmax=node_vmax)

    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_mi, cmap=edge_cmap),
        cax=ax[0],
        orientation="horizontal",
    )
    ax[0].xaxis.set_ticks_position("top")
    ax[0].xaxis.set_label_position("top")

    ax[0].set_xticks([edge_vmin, edge_vmax])
    ax[0].set_xlabel("Mutual Information (nats)", labelpad=-10, fontsize=7)

    node_cmap = CMAPS[node_cmap]
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm_s, cmap=node_cmap),
        cax=ax[1],
        orientation="horizontal",
    )
    ax[1].set_xticks([node_vmin, node_vmax])

    ax[1].set_xlabel("Entropy (nats)", labelpad=-10, fontsize=7)
    fig.savefig(f"{savedir}/colorbar.png", dpi=300)

    # PICKLE THE COLORBAR FOR LATER EDITING IF NEEDED
    with open(f"{savedir}" + os.sep + "colorbar.pkl", "wb") as f:
        pkl.dump((fig, ax), f)

    print(f"Output files written to directory {savedir}.")
    print(f"Navigate to {savedir} and run 'pymol draw_tree.pml' to view tree and open 'colorbar.png' to view colorbar.")

    return fig, ax
