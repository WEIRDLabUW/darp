# Colab
We provide a simplified, slimmed-down version of this codebase in this Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1N0kBjaT773HkzESaXw884wmsmEpZJjEy?authuser=2#scrollTo=1zv_wdGb6W6d)
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

# Retrieval Hyperparameter Explanation
While we find that DARP works well with simple KNN retrieval, we also find performance boosts when we include a history of states with which to do retrieval with. The following 4 hyperparameters define this process:
- **Candidates**: The 'K' in KNN - how many candidate neighbors we want to do cumulative distance on.
- **Lookback**: How far back we want to look (in states) into each trajectory when doing the cumulative distance function.
- **Decay**: How exponentially we want to decrease the influence of older neighbors. For each index:
  - i.e. `i=1` is the most recent observation, and `i=10` is the 10th newest observation.
  - Each `i` will have its respective distance multiplied by `i^decay`.
  - Typically, we want decay to be negative (older observations have less influence).
- **Final Neighbors Ratio**: After calculating the cumulative distance, we take the
  (100 * `final_neighbors_ratio`)% best neighbors. This can be a cheap way to handle multi-modality.
  - If there are likely two modes evenly distributed in our neighbors, and `final_neighbors_ratio` is 0.5, we will take only the 50% closest neighbors post-cumulative distance function, ideally eliminating one of the two modes.
