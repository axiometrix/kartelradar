"""Birga qatnashish grafi: bir xil tenderlarda ko'p marta birga chiqqan firmalar jamoalari."""
from __future__ import annotations

import itertools

import networkx as nx
import pandas as pd


def cobid_graph(bids: pd.DataFrame, min_weight: int = 3) -> nx.Graph:
    g = nx.Graph()
    for _, grp in bids.groupby("tender_id"):
        firms = sorted(set(grp["firm_inn"]))
        for a, b in itertools.combinations(firms, 2):
            w = g.get_edge_data(a, b, {"weight": 0})["weight"]
            g.add_edge(a, b, weight=w + 1)
    weak = [(a, b) for a, b, d in g.edges(data=True) if d["weight"] < min_weight]
    g.remove_edges_from(weak)
    g.remove_nodes_from([n for n in list(g.nodes) if g.degree(n) == 0])
    return g


def suspicious_groups(g: nx.Graph, min_size: int = 3) -> pd.DataFrame:
    if g.number_of_nodes() == 0:
        return pd.DataFrame(columns=["group_id", "firms", "size", "density", "total_weight"])
    comms = nx.algorithms.community.greedy_modularity_communities(g, weight="weight")
    rows = []
    for i, c in enumerate(comms):
        if len(c) < min_size:
            continue
        sub = g.subgraph(c)
        rows.append({"group_id": i, "firms": sorted(c), "size": len(c), "density": nx.density(sub),
                     "total_weight": sub.size(weight="weight")})
    return pd.DataFrame(rows).sort_values(["density", "total_weight"], ascending=False).reset_index(drop=True)
