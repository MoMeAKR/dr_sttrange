import matplotlib.patches as mpatches
import networkx as nx
import numpy as np 
import matplotlib.pyplot as plt 
from collections import defaultdict
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




def plot_multiple_outcome_probabilities(
    data_dict,
    output_path,
    figsize= (12, 8),
    cmap_name= "tab20",
    show_values = False,
    value_fmt= "{:.2f}",
    alpha = 0.95,
    min_segment_alpha = 0.4,
) -> None:
    """
    Plots a grouped, stacked bar chart with dynamic segment opacity.

    Behavior:
    - The x-axis is grouped by outcome. If outcomes are numeric, their spacing
      on the axis is proportional to their value. Otherwise, they are spaced evenly.
    - Within each outcome group, there are separate, side-by-side bars for each label.
    - Each individual bar is a stack of segments. The opacity of each segment is
      proportional to its value relative to other segments in the same stack,
      making larger contributors more opaque.

    Args:
        data_dict: A mapping from a label (str) to a data structure.
                   The data structure must contain 'tree_paths', either at the
                   top level or nested under a 'data' key. 'tree_paths' is an
                   iterable of dicts, each with 'path_utility' and 'path_prob'.
        output_path: The file path where the plot image will be saved.
        figsize: The size of the matplotlib figure (width, height) in inches.
        cmap_name: The name of the matplotlib colormap to use for the labels.
        show_values: If True, annotates each bar segment with its numeric value.
        value_fmt: The format string used for the value annotations.
        alpha: The maximum opacity for the largest segment in a stack (0.0 to 1.0).
        min_segment_alpha: The minimum opacity for the smallest segments (0.0 to 1.0).
    """
    # 1. Aggregate probabilities into lists for each (label, outcome) pair.
    datasets_agg = defaultdict(lambda: defaultdict(list))
    all_outcomes = set()

    for label, data in data_dict.items():
        tree_paths = data.get("tree_paths", data.get("data", {}).get("tree_paths", []))
        if not tree_paths:
            continue
        for path_info in tree_paths:
            outcome = path_info.get("path_utility")
            prob = path_info.get("path_prob")
            if outcome is None or prob is None:
                continue
            try:
                datasets_agg[label][outcome].append(float(prob))
                all_outcomes.add(outcome)
            except (TypeError, ValueError):
                continue

    # 2. Handle the no-data case.
    if not all_outcomes:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title("Outcomes vs Probability (No data to display)")
        ax.text(0.5, 0.5, "No valid data provided.", ha="center", va="center")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path)
        plt.close(fig)
        return

    # 3. Prepare data and layout with intelligent x-axis spacing.
    labels = sorted(list(datasets_agg.keys()))
    n_labels = len(labels)
    is_numeric = all(isinstance(x, (int, float, np.number)) for x in all_outcomes)

    if is_numeric:
        sorted_outcomes = sorted(list(all_outcomes))
        x_centers = np.array(sorted_outcomes, dtype=float)
        xtick_labels = [str(o) for o in sorted_outcomes]
        min_dist = np.min(np.diff(sorted_outcomes)) if len(sorted_outcomes) > 1 else 1.0
        total_group_width = min_dist * 0.8
        bar_width = total_group_width / n_labels
    else:
        sorted_outcomes = sorted(map(str, all_outcomes))
        x_centers = np.arange(len(sorted_outcomes))
        xtick_labels = sorted_outcomes
        total_group_width = 0.8
        bar_width = total_group_width / n_labels

    # 4. Create the grouped and stacked bar plot.
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap(cmap_name, max(1, n_labels))

    for i, label in enumerate(labels):
        offset = (i - (n_labels - 1) / 2) * bar_width
        x_positions = x_centers + offset

        for j, outcome in enumerate(sorted_outcomes):
            probs_list = datasets_agg[label].get(outcome, [])
            if not probs_list:
                continue

            max_prob_in_stack = max(probs_list)
            bottom = 0
            
            for prob_segment in probs_list:
                base_color = cmap(i)
                
                # Calculate dynamic alpha for this segment
                segment_alpha = alpha
                if max_prob_in_stack > 1e-9: # Avoid division by zero
                    normalized_prob = prob_segment / max_prob_in_stack
                    segment_alpha = min_segment_alpha + (alpha - min_segment_alpha) * normalized_prob
                
                final_color = (base_color[0], base_color[1], base_color[2], segment_alpha)

                ax.bar(
                    x_positions[j],
                    prob_segment,
                    width=bar_width,
                    bottom=bottom,
                    color=final_color,
                    edgecolor="black",
                    linewidth=0.5,
                )

                if show_values and prob_segment > 1e-6:
                    y_pos = bottom + prob_segment / 2.0
                    ax.text(
                        x_positions[j],
                        y_pos,
                        value_fmt.format(prob_segment),
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if segment_alpha > 0.6 else "black",
                        clip_on=True,
                    )
                
                bottom += prob_segment

    # 5. Finalize and save the plot.
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Probability")
    ax.set_title("Grouped and Stacked Outcome Probabilities by Label")
    ax.set_xticks(x_centers)
    ax.set_xticklabels(xtick_labels, rotation=45 if not is_numeric else 0, ha="right")
    ax.margins(x=0.05, y=0.05)

    legend_patches = [mpatches.Patch(color=cmap(i), label=label, alpha=alpha) for i, label in enumerate(labels)]
    ax.legend(handles=legend_patches, title="Label")
    
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


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



def plot_graph_dot(
    G: nx.DiGraph,
    figsize= (12, 8),
    node_size: int = 1500,
    node_color: str = "lightblue",
    font_size: int = 10,
    edge_font_size: int = 9,
    with_labels: bool = True,
    edge_label_attr= None,
    filename = None
) -> None:
    """
    Plot a NetworkX DiGraph using Graphviz 'dot' layout when available.

    Behavior:
    - Attempts nx.nx_agraph.graphviz_layout (pygraphviz) first, then nx.nx_pydot.graphviz_layout (pydot),
      otherwise falls back to nx.spring_layout with a warning.
    - Node labels use node attributes in order: 'logical_name', 'node_id', 'name', else the node id.
    - Edge labels: if edge_label_attr provided, uses that attribute value; otherwise it prefers 'prob',
      then 'child_name', then no edge labels.
    - If filename is provided, saves the figure; otherwise displays it with plt.show().

    Args:
        G: Directed graph to plot.
        figsize: Figure size in inches.
        node_size: Size for nodes (passed to nx.draw).
        node_color: Color for nodes.
        font_size: Font size for node labels.
        edge_font_size: Font size for edge labels.
        with_labels: Whether to draw node labels.
        edge_label_attr: Specific edge attribute name to display on edges (e.g., 'prob').
        filename: If provided, save image to this path instead of showing.
    """
    # Resolve layout using graphviz if possible
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    # Prepare node labels (prefer logical_name / node_id / name)
    node_labels = {}
    for nid, data in G.nodes(data=True):
        label = None
        if isinstance(data, dict):
            label = data.get("logical_name") or data.get("node_id") or data.get("name")
        if label is None:
            label = str(nid)
        node_labels[nid] = label

    plt.figure(figsize=figsize)
    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle="->", arrowsize=12)

    # Draw node labels if requested
    if with_labels:
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=font_size)

    # Determine and draw edge labels
    # If edge_label_attr explicitly provided, use that; else attempt to auto-select
    chosen_attr = edge_label_attr
    if chosen_attr is None:
        # Inspect a few edges to decide
        attrs = set()
        for _, _, ed in G.edges(data=True):
            if isinstance(ed, dict):
                attrs.update(ed.keys())
        for preferred in ("prob", "child_name"):
            if preferred in attrs:
                chosen_attr = preferred
                break

    if chosen_attr:
        edge_labels = {}
        for u, v, ed in G.edges(data=True):
            val = ed.get(chosen_attr) if isinstance(ed, dict) else None
            # Convert to string for display; skip empty values
            if val is None:
                label = ""
            else:
                label = str(val)
            edge_labels[(u, v)] = label
        # Filter out empty labels for cleaner rendering
        edge_labels = {k: v for k, v in edge_labels.items() if v}
        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=edge_font_size)

    plt.axis("off")
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches="tight")
    else:
        plt.show()
    plt.close()