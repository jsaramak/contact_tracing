# contact_tracing
Code for Barrat, Kivelä, Lehmann, Saramäki

The Python code contact_tracing.py can be imported into iPython (for e.g. reading data), but its main intended use is for a batch run using a cluster. It was originally run at Aalto University's Triton cluster (SLURM). Therefore simply running the .py file prints all output, so if you run it from a command line, please direct the output to a file (python contact_tracing.py > results.csv). To modify the parameters, please see the __main__ function at the end of the code.

An example .sh file, intended to be used with sbatch in the slurm cluster, is included.
