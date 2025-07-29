# ciMIST
### (conformational inference/maximum information spanning tree)

Decoding protein dynamics with residue-wise conformational inference and tree-structured Potts models

## Installation
After downloading, navigate to the directory containing ciMIST and run the terminal command.
```
python pip install -m .
```

## Requirements
Python requirements are listed in `pyproject.toml`.
We have typically run ciMIST on CPUs with between 256GB and 512GB of RAM.

## Recommendations
In our experience, ciMIST has given good results on between 5 and 10 microseconds of molecular dynamics data sampled at frequences of (1 ns)^{-1} and (500 ps)^{-1} for proteins up to about 300 amino acids long.
Please report any memory issues.

## Usage
ciMIST ships with a command line tool, cimist-fit.
Once the package is installed, options can be seen with 

```
cimist-fit --help
```

An example demonstrating how to analyze the outputs is given in [examples/CRIPT.ipynb](examples/CRIPT.ipynb).

## Visualization of results in PyMOL
The program will create a directory of your choosing.
In order to access the colormaps used by ciMIST in PyMOL, you will need to load some colormaps.
This can be done by adding a line to your `.pymolrc` file that runs the script `pymol_palettes/pymol_palettes.py` included with ciMIST.
