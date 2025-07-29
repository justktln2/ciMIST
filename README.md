# ciMIST
### (conformational inference/maximum information spanning tree)

### Decoding protein dynamics with residue-wise conformational inference and tree-structured Potts models

<img src="https://github.com/justktln2/ciMIST/blob/main/images/GR_on_HSP90.png" width=600>

## Installation
After downloading, navigate to the directory containing ciMIST and run the terminal command.
```
python pip install -m .
```

## Requirements
Python requirements are listed in `pyproject.toml`.
We have typically run ciMIST on CPUs with between 256GB and 512GB of RAM.

## Recommendations
In our experience, ciMIST has given good results on between 5 and 10 microseconds of molecular dynamics data sampled at frequences of once or twice per nanosecond for proteins up to about 300 amino acids long.

## Usage
ciMIST ships with a command line tool `ci-mist`. An analysis template illustrating basic aspects of the API is provided in the outputs of each run in the form of a Jupyter notebook.

```
usage: ci-mist [-h] [-t TRAJECTORY] [-s TOPOLOGY] [-o OUTPUT_PREFIX] [--seed SEED]
                  [--min_mass MIN_MASS] [--prior {percs,haldane,jeffreys,laplace}]

options:
  -h, --help            show this help message and exit
  -t, --trajectory TRAJECTORY
                        The path to the trajectory file,
                        or to a directory that contains all trajectory files and nothing else.
                        Note that if a directory is supplied, all files in that directory must be valid molecular
                        dynamics trajectory files.
  -s, --topology TOPOLOGY
                        The path to the topology file.
  -o, --output_prefix OUTPUT_PREFIX
                        The prefix for the output directory.
  --seed SEED           The random number generator seed, default 0.
  --min_mass MIN_MASS   Minimum probability for conformations, default 0.01.
  --prior {percs,haldane,jeffreys,laplace}
                        Prior to use for residue entropy and pairwise mutual information estimation with the Dirichlet distribution.
                        Each prior corresponds to adding the same number of pseudocounts to each conformation.
                        Options are:
                            -'haldane' : 0 pseudocounts (DEFAULT)
                            -'percs' : 1/K pseudocounts, where K is the number of conformations
                            -'jeffreys' : 1/2 pseudocounts
                            -'laplace' : 1 pseudocount
                            
                        Note that of these options, only 'haldane' and 'percs' add the same total number of pseudocounts to each distribution.

```

An example demonstrating how to analyze outputs is given in [examples/CRIPT.ipynb](examples/CRIPT.ipynb).

## Visualization of results in PyMOL
The program will create a directory of your choosing containing pre-generated scripts that will visualize trees on protein structure.
These visualizations have the option to be color-coded using the [cmocean](https://matplotlib.org/cmocean/) or [cmasher](https://github.com/1313e/CMasher) colormaps, which provide nice contrasts for protein structures.
In order to access these in PyMOL, you will need to add a line to your `.pymolrc` file that runs the script `pymol_palettes/pymol_palettes.py` included with ciMIST.
