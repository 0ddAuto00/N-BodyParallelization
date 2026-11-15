# N-BodyParallelization

This program will go over the implementation of the N-body problem. We will be showcasing the all-tiled pairs implementation in comparison to a both a sequential and parallel rowwise implementation. Parallel efficiency will be evaluated at multiple numbers of bodies to assess speedup as the system grows.

## Installation

Run the following command in a command line from a local directory:

`git clone https://github.com/0ddAuto00/N-BodyParallelization`

You might be prompted to login with your github credentials.

## Running the Programs

By default, the sequential program comes with a .exe that you can run by double-clicking it.

You can also run it through a command prompt through either `nbody` or `./nbody`

The `nbody_rowwise.c` parallel rowwise implementation can be compiled with `gcc -O2 -fopenmp nbody_rowwise.c -o nbody_rowwise -lm` and run with `./nbody_rowwise [config_file]`

## Editing the Program

If changes are made to the codebase, a new executable is needed.

For `nbody.c` :

- run `make` in the command line within the directory of the codebase.

To edit variables, open `nbody.cfg` to change the values of each variable. Do not edit the variable names.

## Progress and Reports

Updates and status of current version can be found in `CHANGELOG.md`. Prose documents with reports of findings and methodology can be found in `docs/`.
