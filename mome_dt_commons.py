import numpy as np 
import matplotlib.pyplot as plt 
import os 
plt.style.use('ggplot')



def sample_path(node_fn, ctx, path=None):
    """Randomly walk through the tree using probabilities."""
    if path is None:
        path = []
    desc, utility_val, children = node_fn(ctx)
    path.append(desc)
    
    if children is None:
        # Terminal node
        if isinstance(utility_val, list):  # stochastic payoff
            probs, outcomes, outcome_descs = zip(*utility_val)
            idx = np.random.choice(len(outcomes), p=probs)
            path.append(outcome_descs[idx])
            return path, outcomes[idx]
        else:
            return path, utility_val
    
    if callable(children):
        return sample_path(children, ctx, path)
    
    probs, nodes = zip(*children)
    chosen = np.random.choice(nodes, p=probs)
    return sample_path(chosen, ctx, path)


def enumerate_paths(node_fn, ctx, prob_so_far=1.0, path=None):
    """
    Enumerate all paths in a decision tree, returning a list of dictionaries:
    {
        'path_probs': probability of the path,
        'path_utility': utility value of the path,
        'path': list describing the path taken
    }
    """
    if path is None:
        path = []
    desc, utility_val, children = node_fn(ctx)
    new_path = path + [desc]

    if children is None:
        if isinstance(utility_val, list):  # stochastic payoff
            results = []
            for p, u, outcome_desc in utility_val:
                results.append({
                    'path_prob': prob_so_far * p,
                    'path_utility': u,
                    'path': new_path + [outcome_desc]
                })
            return results
        else:
            return [{
                'path_prob': prob_so_far,
                'path_utility': utility_val,
                'path': new_path
            }]

    if callable(children):
        return enumerate_paths(children, ctx, prob_so_far, new_path)

    results = []
    for p, child in children:
        results.extend(enumerate_paths(child, ctx, prob_so_far * p, new_path))
    return results


def expected_utility(node_fn, ctx):
    """Expected value over all paths."""
    return sum(path['path_prob'] * path['path_utility'] for path in enumerate_paths(node_fn, ctx))



def plot_multiple_outcome_probabilities(data_dict, output_path, figsize=(8, 6)):
    """
    Plots outcome-probability bar plots for multiple datasets on the same figure.

    Args:
        data_dict (dict): Keys are labels, values are data structures (dicts) 
                          containing 'tree_paths' or 'data'->'tree_paths'.
        output_path (str): Path to save the output image.
        figsize (tuple): Figure size.
    """
    plt.figure(figsize=figsize)
    colors = plt.cm.get_cmap('tab10', len(data_dict))  # Get a colormap

    for idx, (label, data) in enumerate(data_dict.items()):
        # Extract tree_paths
        tree_paths = data.get('tree_paths', data.get('data', {}).get('tree_paths', []))
        outcomes = []
        probs = []
        for path_info in tree_paths:
            outcome = path_info.get('path_utility')
            prob = path_info.get('path_prob')
            if outcome is not None and prob is not None:
                outcomes.append(outcome)
                probs.append(prob)
        # Sort by outcome for cleaner plot
        sorted_pairs = sorted(zip(outcomes, probs), key=lambda x: x[0])
        if sorted_pairs:
            outcomes_sorted, probs_sorted = zip(*sorted_pairs)
        else:
            outcomes_sorted, probs_sorted = [], []
        # Offset bars for each dataset to avoid overlap
        bar_width = 0.6 / len(data_dict)
        offsets = np.linspace(-0.3, 0.3, len(data_dict))
        x_positions = np.array(outcomes_sorted) + offsets[idx]
        plt.bar(
            x_positions,
            probs_sorted,
            width=bar_width,
            color=colors(idx),
            edgecolor='black',
            label=label
        )

    plt.xlabel('Outcome')
    plt.ylabel('Probability')
    plt.title('Outcomes vs Probability')
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_cumulative_risk(results_dict, output_path):
    """
    Plots cumulative risk analysis for multiple result sets on the same figure.

    Args:
        results_dict (dict): Dictionary where keys are labels and values are results dicts
                             (each with a "tree_paths" list of dicts containing "path_utility" and "path_prob").
        output_path (str): Path to save the output figure.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(8, 6))

    for label, results in results_dict.items():
        tree_paths = results["tree_paths"]

        # Extract utilities and probabilities
        utilities = []
        probs = []
        for path in tree_paths:
            utility = float(path["path_utility"])
            prob = float(path["path_prob"])
            utilities.append(utility)
            probs.append(prob)

        # Sort by utility (ascending: risk analysis)
        sorted_indices = np.argsort(utilities)
        sorted_utilities = np.array(utilities)[sorted_indices]
        sorted_probs = np.array(probs)[sorted_indices]

        # Compute cumulative probability
        cum_probs = np.cumsum(sorted_probs)

        # Plot for this label
        plt.step(sorted_utilities, cum_probs, where='post', label=label)

    plt.xlabel("Utility")
    plt.ylabel("Cumulative Probability")
    plt.title("Cumulative Risk Analysis")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()