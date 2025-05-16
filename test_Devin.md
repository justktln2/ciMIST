This is the documentation for Devin Meng's test on the ciMIST package.
## Environment Installation
It may be helpful to include a command creating conda environment before install anything so that nobody will wrongly install packages into the base environment. It's also a good place to specify python version.

I just continue by running: `conda create -n ciMIST python=3.9`

The environment installation command doesn't work:
```
(ciMIST) [s439821@Nucleus039 ciMIST]$ python pip install -m .
python: can't open file '/endosome/work/greencenter/s439821/ciMIST/pip': [Errno 2] No such file or directory
```
I run this command to continue: `pip install .`. Packages in `pyproject.toml` are successfully installed.
## Quick Start Test
The first thing I try to run is the jupyter notebook in `/examples`. Everything looks good.

Then I check the command line tool by `cimist-fit --help`.
Here are some feedback for the help info:
- It may be helpful adding what file format is supported as input to the tool, as currently the help info only say trajectory and topology, which is a bit vague.
- the parameter min_mass probably should be made more clear. Now it's hard to understand why and how to set this parameter.
- It will be great to include one or few small example or usecase in the help info.
- Don't know why but the command runs really slow to get help info.

Then I input an actual md trajectory and topology to test it out. (my GPCR 7F0T simulation)
Feedback from this trial:
- It doesn't have the slow issue as getting help info
- Probably need more clear definition or examples to specify output_prefix, because now is confusing.