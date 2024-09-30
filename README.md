# Credal Two-sample Tests Implementation
This repository contains code to reproduce experiments from the paper on Credal Two-sample Tests.

- `src/testing_algorithms` contains the implementation of plausibility test and specification test, inclusion and equality tests are simply repeated application of specification test.
- `src/experiment_scripts` contains the experiment scripts to reproduce the experiments in the paper:
  - `src/experiment_scripts/synthetic/` contains scripts to reproduce the synthetic experiments based on mixture of Gaussians versus mixture of students distributions.
  - `src/experiment_scripts/mnist/` contains scripts to reproduce the real data experiments based on MNIST.
