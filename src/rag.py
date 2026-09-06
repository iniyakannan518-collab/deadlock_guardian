"""
rag.py
------
Resource Allocation Graph (RAG) construction and drawing using NetworkX + Matplotlib.

Node types:
    process nodes  -- drawn as circles,   e.g. "P0"
    resource nodes -- drawn as squares,   e.g. "R0"

Edge types:
    assignment edge:  R_j -> P_i   (resource j is allocated to process i)
    request edge:      P_i -> R_j   (process i is waiting for a unit of resource j)

A directed edge P_i -> R_j -> P_k (request then assignment) is how a
cycle -- and therefore a potential deadlock -- shows up visually.
"""

import networkx as nx
import matplotlib.pyplot as plt


def build_rag(allocation, request, process_names=None, resource_names=None):
    """
    allocation: n x m matrix, allocation[i][j] = units of resource j held by process i
    request:    n x m matrix, request[i][j]    = units of resource j process i is waiting for
    Returns a networkx.DiGraph with typed nodes/edges.
    """
    n = len(allocation)
    m = len(allocation[0]) if n else 0
    process_names = process_names or [f"P{i}" for i in range(n)]
    resource_names = resource_names or [f"R{j}" for j in range(m)]

    G = nx.DiGraph()
    for p in process_names:
        G.add_node(p, kind="process")
    for r in resource_names:
        G.add_node(r, kind="resource")

    for i in range(n):
        for j in range(m):
            if allocation[i][j] > 0:
                G.add_edge(resource_names[j], process_names[i],
                           kind="assignment", weight=allocation[i][j])
            if request[i][j] > 0:
                G.add_edge(process_names[i], resource_names[j],
                           kind="request", weight=request[i][j])
    return G


def build_wait_for_graph(rag):
    """
    Collapse a Resource Allocation Graph into a Process-only Wait-For Graph:
    an edge P_i -> P_k exists if P_i is requesting a resource currently held by P_k.
    Useful because "cycle in RAG (single-instance resources)" == "cycle in wait-for graph".
    """
    wfg = nx.DiGraph()
    processes = [n for n, d in rag.nodes(data=True) if d.get("kind") == "process"]
    wfg.add_nodes_from(processes)

    for p in processes:
        for r in rag.successors(p):
            if rag.nodes[r].get("kind") != "resource":
                continue
            if rag[p][r].get("kind") != "request":
                continue
            # who holds resource r? Assignment edges point R -> holder, so
            # the holder is a SUCCESSOR of r (not a predecessor).
            for holder in rag.successors(r):
                if rag.nodes[holder].get("kind") == "process" and holder != p \
                        and rag[r][holder].get("kind") == "assignment":
                    wfg.add_edge(p, holder)
    return wfg


def draw_rag(G, title="Resource Allocation Graph", ax=None, save_path=None, highlight_cycle=None):
    """
    Draw the RAG: circles = processes, squares = resources.
    Assignment edges (resource -> process) drawn in blue,
    request edges (process -> resource) drawn in red/dashed.
    If highlight_cycle (list of nodes forming a cycle) is given, those edges are bolded.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 6))

    processes = [n for n, d in G.nodes(data=True) if d.get("kind") == "process"]
    resources = [n for n, d in G.nodes(data=True) if d.get("kind") == "resource"]

    pos = {}
    pos.update(nx.circular_layout(processes, scale=1.0, center=(-1.4, 0)) if processes else {})
    pos.update(nx.circular_layout(resources, scale=1.0, center=(1.4, 0)) if resources else {})

    nx.draw_networkx_nodes(G, pos, nodelist=processes, node_shape="o",
                            node_color="#8ecae6", node_size=1600, ax=ax, edgecolors="black")
    nx.draw_networkx_nodes(G, pos, nodelist=resources, node_shape="s",
                            node_color="#ffb703", node_size=1600, ax=ax, edgecolors="black")
    nx.draw_networkx_labels(G, pos, ax=ax, font_weight="bold")

    assignment_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "assignment"]
    request_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "request"]

    cycle_edges = set()
    if highlight_cycle:
        for k in range(len(highlight_cycle)):
            cycle_edges.add((highlight_cycle[k], highlight_cycle[(k + 1) % len(highlight_cycle)]))

    def edge_width(u, v):
        return 3.5 if (u, v) in cycle_edges else 1.6

    for (u, v) in assignment_edges:
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax, edge_color="#023e8a",
                                width=edge_width(u, v), arrowsize=20,
                                connectionstyle="arc3,rad=0.08")
    for (u, v) in request_edges:
        color = "#d00000" if (u, v) in cycle_edges else "#e07a5f"
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax, edge_color=color,
                                width=edge_width(u, v), style="dashed", arrowsize=20,
                                connectionstyle="arc3,rad=0.08")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")

    # simple legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#8ecae6", markeredgecolor="black", markersize=14, label="Process"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#ffb703", markeredgecolor="black", markersize=14, label="Resource"),
        Line2D([0], [0], color="#023e8a", lw=2, label="Assignment (R \u2192 P)"),
        Line2D([0], [0], color="#e07a5f", lw=2, linestyle="dashed", label="Request (P \u2192 R)"),
    ]
    if highlight_cycle:
        legend_elems.append(Line2D([0], [0], color="#d00000", lw=3, label="Cycle (deadlock)"))
    ax.legend(handles=legend_elems, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=2, frameon=False, fontsize=9)

    if own_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
    return ax
