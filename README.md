# MEC8211_Hiver2024_Devoir2

This is the repository for the 2nd MEC8211 homework of the Winter 2024 term.

## Authors 

Amishga Alphonius, Ayman Benkiran, and Maxence Farin.

# Project Description

This codes solves the stationary state salt concentration ($`C`$) in an under-water cylindrical concrete pillar. The problem is modeled as follows:

$`\frac{\partial C}{\partial t} = D_\mathrm{eff} \nabla^2 C - S`$

where $`D_\mathrm{eff}`$ is the effective diffusion coefficient of salt and $`S`$ is the source term from a reaction between the salt and a concrete component.

Assuming that the pillar is tall and that the environment is homogenous, through polar coordinates, the problem can be written as a 1D problem along the r-axis.

# Code Requirements
This project uses a Python code to solve the problem described above using the finite difference method.

A simple call of the main script (`python3 devoir2_main.py`) produces a convergence analysis of both a first-order spatial finite difference method and a second-order one.

To run the code, the following modules are required:

- matplotlib
- numpy
- scipy
- sympy
- math
- typing
- os

# Code Architecture

- All source files are located in the `src` folder.
  
  - `devoir2_main.py`: Main code. Can be called using `python3 devoir2_main.py`.
  - `devoir2_functions.py`: Function library for the code.
  - `devoir2_postresults.py`: Postprocessing library for the code.
  - `devoir2_tests_unitaires.py`: Unit test script. Can be called using `python3 devoir2_tests_unitaires.py`
  
- Produced raw solutions are stored in `CSV` format in the `data` folder.
- Postprocessed results (figures and data) are stored in the `results` folder.
- A change log is available in the `doc` folder.
