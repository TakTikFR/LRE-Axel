import numpy as np
from graphviz import Digraph


def display_graph(parent: np.ndarray, img: np.ndarray, filename: str) -> Digraph:
    """
    Display the parent image (The tree) as graphviz tree.

    :param parent: Parent image.
    :param img: Starting image.
    :param filename: Name of the graphviz image.
    :return: Graphviz tree that represents the parent image.
    """
    rows, cols = parent.shape
    dot = Digraph()

    for y in range(rows):
        for x in range(cols):
            if parent[y][x] is not None:
                node_from = f"({y}, {x}) [{img[y, x]}]"
                node_to = f"({parent[y][x][0]}, {parent[y][x][1]}) [{img[parent[y][x][0], parent[y][x][1]]}]"
                dot.edge(node_from, node_to)

    dot.render(filename, format="png", cleanup=True)
    return dot
