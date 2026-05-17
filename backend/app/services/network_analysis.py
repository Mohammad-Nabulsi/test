from __future__ import annotations

from typing import Dict, Tuple

import networkx as nx
import pandas as pd


def build_cooccurrence_network(transactions_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    if transactions_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {"ok": False, "message": "No transactions."}

    G = nx.Graph()
    for _, r in transactions_df.iterrows():
        items = [x for x in str(r.get("items", "")).split("|") if x.strip()]
        items = list(dict.fromkeys(items))  # stable unique
        for it in items:
            if not G.has_node(it):
                G.add_node(it)
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a, b = items[i], items[j]
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)

    if G.number_of_nodes() == 0:
        return pd.DataFrame(), pd.DataFrame(), {"ok": False, "message": "Empty graph."}

    degree_c = nx.degree_centrality(G)
    pagerank = nx.pagerank(G, weight="weight") if G.number_of_edges() > 0 else {n: 0.0 for n in G.nodes()}

    nodes = []
    for n in G.nodes():
        nodes.append(
            {
                "node": n,
                "type": n.split("=", 1)[0] if "=" in n else "misc",
                "degree": int(G.degree(n)),
                "degree_centrality": float(degree_c.get(n, 0.0)),
                "pagerank": float(pagerank.get(n, 0.0)),
            }
        )
    nodes_df = pd.DataFrame(nodes).sort_values("pagerank", ascending=False)

    edges = []
    for u, v, d in G.edges(data=True):
        edges.append({"source": u, "target": v, "weight": int(d.get("weight", 1))})
    edges_df = pd.DataFrame(edges).sort_values("weight", ascending=False)

    # Simple community detection if possible
    communities = []
    try:
        import networkx.algorithms.community as nx_comm

        if G.number_of_edges() > 0 and G.number_of_nodes() >= 10:
            comms = list(nx_comm.greedy_modularity_communities(G, weight="weight"))
            communities = [sorted(list(c)) for c in comms[:10]]
    except Exception:
        communities = []

    summary = {
        "ok": True,
        "nodes": int(G.number_of_nodes()),
        "edges": int(G.number_of_edges()),
        "top_nodes_by_pagerank": nodes_df.head(10)[["node", "pagerank", "degree"]].to_dict(orient="records"),
        "communities": communities,
    }
    return nodes_df.reset_index(drop=True), edges_df.reset_index(drop=True), summary

