# Installation

To install, just run `./scripts/create_env.sh`. This will create a new conda environment with all necessary dependencies. Note that you will not be able to run robosuite and robocasa experiments with the same environment due to version conflicts -- change the `ROBOSUITE` variable from `true` to `false` to download the robocasa dependencies.

# Reproducing Results

See `./scripts/reproduce_results.sh` for all necessary code to reproduce BC and DARP results from the paper. For MuJoCo tasks, we provide the data for you. However, if you want to run Robosuite benchmarks, you'll have to generate the data yourself. See MimicGen's documentation for more information on that: https://mimicgen.github.io/docs/introduction/overview.html.

We provide scripts to turn MimicGen's .hdf5 file to a dataset usable by this codebase: see `hdf5_to_d4rl.py`. We also provide scripts to turn these demonstrations into a dataset of RGB images, and another script to turn those into R3M features.


# Data Format

Data is stored in the `data` folder as a pickle file. Each data file is a list of trajectories,
where a trajectory is a dict with `"observations"` and `"actions"` keys. The respective corresponding
values are 2D ndarrays, where the first dimension is horizon and the second dimension is state
dimension and action dimension, respectively.

For example, if `T` is the length of a trajectory, `o` is the observation dimension, and `a` is the action dimension:

```
[
    {
        "observations": np.ndarray of size `T x o`,
        "actions": np.ndarray of size `T x a`
    },
    {
        "observations": np.ndarray of size `T x o`,
        "actions": np.ndarray of size `T x a`
    },
    ...
]
```

We provide scripts to turn MimicGen's .hdf5 file to a dataset usable by this codebase: see `hdf5_to_d4rl.py`. We also provide scripts to turn these demonstrations into a dataset of RGB images, and another script to turn those into R3M features.
