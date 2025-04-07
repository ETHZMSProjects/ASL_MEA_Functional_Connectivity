import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.lines import Line2D

def plot_comparison_bar_charts(results_dict, title="Inter-Region Comparisons",saveFig=False,dpi=300):
    """
    Creates 4 subplots comparing 44 → 6v vs. 6v → 44 for word and sentence conditions.

    Parameters:
    - results_dict (dict): Dictionary with keys as filenames (without .npy) and values as scalar results.
    - title (str): Title for the entire figure.
    """
    
    # Define subplot order for the 4 inter-region comparisons
    comparisons = [
        ("sup44_gc_sup6v", "sup6v_gc_sup44"),
        ("sup44_gc_inf6v", "sup6v_gc_inf44"),
        ("inf44_gc_sup6v", "inf6v_gc_sup44"),
        ("inf44_gc_inf6v", "inf6v_gc_inf44")
    ]

    # Define word vs. sentence labels
    conditions = ["word", "sent"]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    axes = axes.flatten()
    fig.suptitle(title, fontsize=16)

    for i, (cond_44_to_6v, cond_6v_to_44) in enumerate(comparisons):
        # Extract values from the dictionary
        word_44_to_6v = results_dict.get(f"word_{cond_44_to_6v}_results", np.nan)
        sent_44_to_6v = results_dict.get(f"sent_{cond_44_to_6v}_results", np.nan)
        word_6v_to_44 = results_dict.get(f"word_{cond_6v_to_44}_results", np.nan)
        sent_6v_to_44 = results_dict.get(f"sent_{cond_6v_to_44}_results", np.nan)
        
        # Data for bar chart
        data = [
            [word_44_to_6v, sent_44_to_6v],  # 44 to 6v
            [word_6v_to_44, sent_6v_to_44]   # 6v to 44
        ]

        # Bar plot positions
        x = np.arange(len(conditions))  # [0, 1]
        width = 0.35  # Width of bars
        
        # Plot bars
        axes[i].bar(x - width/2, data[0], width, label="44 → 6v")
        axes[i].bar(x + width/2, data[1], width, label="6v → 44")

        # Formatting
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(["Words", "Sentences"])
        axes[i].set_title(cond_44_to_6v.replace("_gc_", " → ").replace("_", " ").title())
        axes[i].legend()

    # Show plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title
    if saveFig:
        plt.savefig(saveFig,dpi=300)
    plt.show()

def plot_delay_speaking_comparison(results_dict, title='Inter-Region Connectivity During Speaking and Delay Periods',saveFig=False,dpi=300):
    """
    Creates 4 subplots comparing 44 → 6v vs. 6v → 44 for word and sentence conditions.

    Parameters:
    - results_dict (dict): Dictionary with keys as filenames (without .npy) and values as scalar results.
    - title (str): Title for the entire figure.
    """
    
    # Define subplot order for the 4 inter-region comparisons
    comparisons = [
        ("sup44_gc_sup6v", "sup6v_gc_sup44"),
        ("sup44_gc_inf6v", "sup6v_gc_inf44"),
        ("inf44_gc_sup6v", "inf6v_gc_sup44"),
        ("inf44_gc_inf6v", "inf6v_gc_inf44")
    ]

    # Define control vs. regular
    conditions = ["ctrl_", ""]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
    axes = axes.flatten()
    fig.suptitle(title, fontsize=16)

    for i, (cond_44_to_6v, cond_6v_to_44) in enumerate(comparisons):
        # Extract values from the dictionary
        ctrl_44_to_6v = results_dict.get(f"ctrl_{cond_44_to_6v}", np.nan)
        r44_to_6v = results_dict.get(f"{cond_44_to_6v}", np.nan)
        ctrl_6v_to_44 = results_dict.get(f"ctrl_{cond_6v_to_44}", np.nan)
        r6v_to_44 = results_dict.get(f"{cond_6v_to_44}", np.nan)
        
        # Data for bar chart
        data = [
            [ctrl_44_to_6v, r44_to_6v],  # 44 to 6v
            [ctrl_6v_to_44, r6v_to_44]   # 6v to 44
        ]

        # Bar plot positions
        x = np.arange(len(conditions))  # [0, 1]
        width = 0.35  # Width of bars
        
        # Plot bars
        axes[i].bar(x - width/2, data[0], width, label="44 → 6v")
        axes[i].bar(x + width/2, data[1], width, label="6v → 44")

        # Formatting
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(["Delay", "Speaking"])
        axes[i].set_title(cond_44_to_6v.replace("_gc_", " → ").replace("_", " ").title())
        axes[i].legend()

    # Show plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to fit title
    if saveFig:
        plt.savefig(saveFig,dpi=300)
    plt.show()


def plot_causality_flow(results_dict, title="Inter-Region Causality Flow",saveFig=False,dpi=300):
    """
    Visualizes 44 → 6v and 6v → 44 connections as weighted arrows.
    
    Updates:
    - Adds a visual legend for word/sentence (solid/dashed) and direction (blue/red).
    - Adjusts thickness scaling to make differences more visible.
    
    Parameters:
    - results_dict (dict): Dictionary with filenames (without .npy) as keys and values as proportions (0-1).
    - title (str): Title for the figure.
    """
    
    # Define node positions
    nodes = {
        "sup44": (-1, 1),
        "inf44": (-1, -1),
        "sup6v": (1, 1),
        "inf6v": (1, -1)
    }
    
    # Define connections: (source, target, key for results_dict)
    forward_connections = [
        ("sup44", "sup6v", "word_sup44_gc_sup6v_results", "word"),
        ("sup44", "inf6v", "word_sup44_gc_inf6v_results", "word"),
        ("inf44", "sup6v", "word_inf44_gc_sup6v_results", "word"),
        ("inf44", "inf6v", "word_inf44_gc_inf6v_results", "word"),
        ("sup44", "sup6v", "sent_sup44_gc_sup6v_results", "sent"),
        ("sup44", "inf6v", "sent_sup44_gc_inf6v_results", "sent"),
        ("inf44", "sup6v", "sent_inf44_gc_sup6v_results", "sent"),
        ("inf44", "inf6v", "sent_inf44_gc_inf6v_results", "sent"),
    ]
    
    feedback_connections = [
        ("sup6v", "sup44", "word_sup6v_gc_sup44_results", "word"),
        ("inf6v", "sup44", "word_inf6v_gc_sup44_results", "word"),
        ("sup6v", "inf44", "word_sup6v_gc_inf44_results", "word"),
        ("inf6v", "inf44", "word_inf6v_gc_inf44_results", "word"),
        ("sup6v", "sup44", "sent_sup6v_gc_sup44_results", "sent"),
        ("inf6v", "sup44", "sent_inf6v_gc_sup44_results", "sent"),
        ("sup6v", "inf44", "sent_sup6v_gc_inf44_results", "sent"),
        ("inf6v", "inf44", "sent_inf6v_gc_inf44_results", "sent"),
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    
    # Plot nodes
    node_radius = 0.2  # Circle size
    for node, (x, y) in nodes.items():
        ax.scatter(x, y, s=2000, color="white", edgecolors="black", zorder=3)
        ax.text(x, y, node, fontsize=12, ha='center', va='center', weight='bold')

    def add_curved_arrow(src, tgt, color, lw, curvature, offset_dir, linestyle):
        """Draws a curved arrow between two nodes with offset to prevent overlap."""
        x1, y1 = nodes[src]
        x2, y2 = nodes[tgt]
        
        # Compute unit vector
        dx, dy = x2 - x1, y2 - y1
        distance = np.hypot(dx, dy)
        ux, uy = dx / distance, dy / distance  # Unit vector

        # Get perpendicular offset vector (rotate 90 degrees)
        perp_x, perp_y = -uy, ux

        # Offset amount
        offset_amount = 0.1 * offset_dir  # Adjust for separation

        # Adjust start and end points to:
        # 1. Avoid overlap with circles
        # 2. Slightly separate forward and backward arrows
        x1_new, y1_new = x1 + ux * node_radius + perp_x * offset_amount, y1 + uy * node_radius + perp_y * offset_amount
        x2_new, y2_new = x2 - ux * node_radius + perp_x * offset_amount, y2 - uy * node_radius + perp_y * offset_amount

        # Create curved arrow
        arrow = FancyArrowPatch((x1_new, y1_new), (x2_new, y2_new),
                                connectionstyle=f"arc3,rad={curvature}",
                                arrowstyle="->", mutation_scale=20,
                                linewidth=lw, linestyle=linestyle, color=color)
        ax.add_patch(arrow)

    # Adjust thickness scaling
    def scale_line_thickness(value):
        return max(1, value * 20)  # Increased scaling for visibility

    # Draw forward connections (44 → 6v) with upward curvature
    for src, tgt, key, category in forward_connections:
        value = results_dict.get(key, 0)
        lw = scale_line_thickness(value)
        linestyle = "solid" if category == "word" else "dashed"
        curvature = 0.1 if category == "word" else 0.25  # Different curvature for words vs sentences
        add_curved_arrow(src, tgt, "blue", lw, curvature, offset_dir=1, linestyle=linestyle)

    # Draw feedback connections (6v → 44) with downward curvature
    for src, tgt, key, category in feedback_connections:
        value = results_dict.get(key, 0)
        lw = scale_line_thickness(value)
        linestyle = "solid" if category == "word" else "dashed"
        curvature = 0.1 if category == "word" else 0.25  # Different curvature for words vs sentences
        add_curved_arrow(src, tgt, "red", lw, curvature, offset_dir=-1, linestyle=linestyle)

    # Create legend

    legend_patches = [
        Line2D([0], [0], color="blue", lw=3, linestyle="solid", label="Words (44 → 6v)"),
        Line2D([0], [0], color="red", lw=3, linestyle="solid", label="Words (6v → 44)"),
        Line2D([0], [0], color="blue", lw=3, linestyle="dashed", label="Sentences (44 → 6v)"),
        Line2D([0], [0], color="red", lw=3, linestyle="dashed", label="Sentences (6v → 44)")
    ]

    #ax.legend(handles=legend_patches, loc="upper left", fontsize=10, frameon=True)
    ax.legend(handles=legend_patches, loc="lower center", fontsize=10, frameon=True, ncol=2)


    # Title
    ax.set_title(title, fontsize=14)
    if saveFig:
        plt.savefig(saveFig,dpi=300)
    # Show plot
    plt.show()

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np

def plot_causality_flow_speakdelay(results_dict, title="Inter-Region Causality Flow", scalar = 20, saveFig=False, dpi=300):
    """
    Visualizes 44 → 6v and 6v → 44 connections as weighted arrows.

    Updates:
    - Compares speaking vs. delay conditions (solid vs. dashed lines).
    - Adjusts thickness scaling to make differences more visible.
    
    Parameters:
    - results_dict (dict): Dictionary with filenames as keys and values as proportions (0-1).
    - title (str): Title for the figure.
    - saveFig (str or bool): Filename to save the figure if given.
    - dpi (int): Resolution for saving the figure.
    """

    # Define node positions
    nodes = {
        "sup44": (-1, 1),
        "inf44": (-1, -1),
        "sup6v": (1, 1),
        "inf6v": (1, -1)
    }

    # Define all possible area connections
    area_connections = [
        ("sup44", "sup6v"),
        ("sup44", "inf6v"),
        ("inf44", "sup6v"),
        ("inf44", "inf6v"),
        ("sup6v", "sup44"),
        ("inf6v", "sup44"),
        ("sup6v", "inf44"),
        ("inf6v", "inf44"),
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')  # Ensure white background
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # Plot nodes
    node_radius = 0.2  # Circle size
    for node, (x, y) in nodes.items():
        ax.scatter(x, y, s=2000, color="white", edgecolors="black", zorder=3)
        ax.text(x, y, node, fontsize=12, ha='center', va='center', weight='bold')

    def add_curved_arrow(src, tgt, color, lw, curvature, offset_dir, linestyle):
        """Draws a curved arrow between two nodes with offset to prevent overlap."""
        x1, y1 = nodes[src]
        x2, y2 = nodes[tgt]

        # Compute unit vector
        dx, dy = x2 - x1, y2 - y1
        distance = np.hypot(dx, dy)
        ux, uy = dx / distance, dy / distance  # Unit vector

        # Get perpendicular offset vector (rotate 90 degrees)
        perp_x, perp_y = -uy, ux

        # Offset amount
        offset_amount = 0.1 * offset_dir  # Adjust for separation

        # Adjust start and end points to:
        # 1. Avoid overlap with circles
        # 2. Slightly separate forward and backward arrows
        x1_new, y1_new = x1 + ux * node_radius + perp_x * offset_amount, y1 + uy * node_radius + perp_y * offset_amount
        x2_new, y2_new = x2 - ux * node_radius + perp_x * offset_amount, y2 - uy * node_radius + perp_y * offset_amount

        # Create curved arrow
        arrow = FancyArrowPatch((x1_new, y1_new), (x2_new, y2_new),
                                connectionstyle=f"arc3,rad={curvature}",
                                arrowstyle="->", mutation_scale=20,
                                linewidth=lw, linestyle=linestyle, color=color)
        ax.add_patch(arrow)

    # Adjust thickness scaling
    def scale_line_thickness(value):
        return max(1, value * scalar)  # Increased scaling for visibility

    # Draw arrows for speaking and control (delay) conditions
    for src, tgt in area_connections:
        for condition, linestyle, offset_dir in [('speaking', 'solid', 1), ('delay', 'dashed', -1)]:
            key = f"ctrl_{src}_gc_{tgt}" if condition == "delay" else f"{src}_gc_{tgt}"
            if key in results_dict:
                value = results_dict[key]
                lw = scale_line_thickness(value)
                color = "blue" if "6v" in tgt else "red"  # Different colors for 6v and 44 targets
                curvature = 0.1 if condition == "speaking" else 0.25  # Different curvature for conditions
                add_curved_arrow(src, tgt, color, lw, curvature, offset_dir, linestyle)

    # Create legend
    legend_patches = [
        Line2D([0], [0], color="black", lw=3, linestyle="solid", label="Speaking"),
        Line2D([0], [0], color="black", lw=3, linestyle="dashed", label="Delay"),
        Line2D([0], [0], color="blue", lw=3, linestyle="solid", label="6v Target"),
        Line2D([0], [0], color="red", lw=3, linestyle="solid", label="44 Target"),
    ]
    ax.legend(handles=legend_patches, loc="lower center", fontsize=10, frameon=True, ncol=2)

    # Title
    ax.set_title(title, fontsize=14)
    
    # Save figure if requested (ensuring white background)
    if saveFig:
        plt.savefig(saveFig, dpi=dpi, bbox_inches="tight", facecolor='white')

    # Show plot
    plt.show()
