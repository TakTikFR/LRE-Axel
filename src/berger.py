import numpy as np
from typing import Optional


Point = tuple[int, int]


def find_root(node: Point, parent: np.ndarray) -> Optional[Point]:
    """
    This function takes a node and finds the root from this node.

    :param node: The node for which we want to find the root.
    :param parent: A 2D NumPy array where parent[x, y] stores the coordinates of the parent pixel.
    :return: The root associated with this node.
    """

    while parent[node] is not None and node != parent[node]:
        node = parent[node]

    return node


def undef_neighbors(p: Point, parent: np.ndarray) -> list[Point]:
    """
    Find all the 4-neighbors that has been already visited (their parent value is already set) and return them.

    :param p: Pixel reference to find their neighbors.
    :param parent: Parent image.
    :return: List of all defined 4-neighbors (Or empty if no neighbors).
    """
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

    neighbors = []
    rows, cols = parent.shape
    for dx, dy in directions:
        nx = p[1] + dx
        ny = p[0] + dy
        if 0 <= nx < cols and 0 <= ny < rows:
            neighbor = parent[ny][nx]
            if neighbor is not None:
                neighbors.append((ny, nx))
    return neighbors


def reverse_sort(f: np.ndarray) -> list[Point]:
    """
    Return all the coordinates sorted in descending order by the value of the pixel in f.

    :param f: Starting image.
    :return: List of all coordinates sorted in descending order.
    """
    rows, cols = f.shape
    coords_values = [(x, y, f[x, y]) for x in range(rows) for y in range(cols)]
    sorted_coords_values = sorted(coords_values, key=lambda x: x[2], reverse=True)
    sorted_coords_array = [(x, y) for x, y, value in sorted_coords_values]

    return sorted_coords_array


def compute_tree(f: np.ndarray) -> tuple[list[Point], np.ndarray]:
    """
    Fills the parent matrix and thus obtain the complete tree.

    :param f: Starting image.
    :return: The list of pixels sorted in decreasing order and the parent image (The tree) completed
    """
    parent = np.full(f.shape, None, dtype=object)
    R = reverse_sort(f)
    for p in R:
        if f[p] == 0:
            parent[p] = None
        else:
            parent[p] = p
            for n in undef_neighbors(p, parent):
                r = find_root(n, parent)
                if r is not None:
                    parent[r] = p  # if r != p else None

    return R, parent


def compute_tree_AllNodes(f: np.ndarray) -> tuple[list[Point], np.ndarray]:
    """
    Fills the parent matrix and thus obtain the complete tree of all nodes.

    :param f: Starting image.
    :return: The list of pixels sorted in decreasing order and the parent image (The tree) completed
    """
    parent = np.full(f.shape, None, dtype=object)
    R = reverse_sort(f)
    for p in R:
        parent[p] = p
        for n in undef_neighbors(p, parent):
            r = find_root(n, parent)
            if r is not None and r != p:
                parent[r] = p

    return R, parent


def canonize_tree(
    parent: np.ndarray, f: np.ndarray, R: tuple[list[Point]]
) -> np.ndarray:
    """
    Getting a simple and compressed representation of the tree.

    :param R: List of all pixel sorted in decreasing order.
    :param parent: Parent image.
    :param f: Starting image.
    :return: Parent matrix with the canonized representation of the tree.
    """
    for p in reversed(R):
        q = parent[p]
        if q is not None and parent[q] is not None and f[parent[q]] == f[q]:
            parent[p] = parent[q]

    return parent


def canonize_tree_no_order(parent: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Getting a simple and compressed representation of the tree.

    :param parent: Parent image.
    :param f: Starting image.
    :return: Parent matrix with the canonized representation of the tree.
    """
    R = reverse_sort(f)
    for p in reversed(R):
        q = parent[p]
        if q is not None and parent[q] is not None and f[parent[q]] == f[q]:
            parent[p] = parent[q]

    return parent


def compute_area(f: np.ndarray, parent: np.ndarray) -> np.ndarray:
    """
    Create a matrix that associates all nodes with their area.

    :param f: Starting image.
    :param parent: Parent image.
    :return: Area matrix of the tree.
    """
    R = reversed(reverse_sort(f))
    area = np.full(f.shape, 1, dtype=object)
    for p in R:
        area[parent[p]] += area[p]

    return area


def find_peak_root(
    parent: np.ndarray, x: Point, lvl: int, f: np.ndarray
) -> tuple[Point, Point | None]:
    """
    Find the peak root of a given node in the tree.

    :param parent: Parent image.
    :param x: The starting node.
    :param lvl: Reference intensity level
    :param f: Starting image.
    :return: A tuple containing the peak root and its parent (or None if no parent exists).1
    """
    q = parent[x]
    while q != x and lvl <= f[q]:
        old = q
        q = parent[q]
        x = old
    return x, q


def find_level_root(
    parent: np.ndarray, x: Point, f: np.ndarray
) -> tuple[Point, Point | None]:
    return find_peak_root(parent, x, f[x], f)


def order_op(p: Point, q: Point, f: np.ndarray) -> bool:
    """
    Implementation of a total order between pixels.

    :param p: First pixel coordinates.
    :param q: Second pixel coordinates.
    :param f: Starting image.
    :return: Boolean of the total order between the first and second pixel.
    """
    if f[p] < f[q]:
        return True
    elif f[p] == f[q]:
        return p > q

    return False


def connect(
    a: Point, b: Point | None, f: np.ndarray, parent: np.ndarray
) -> None:
    """
    Connect two separate trees.

    :param a: Pixel associated to the first tree.
    :param b: Pixel associated to the second tree.
    :param f: Starting image.
    :param parent: Parent image.
    :return: New parent map that connect the two trees.
    """
    while b is not None:
        if f[b] < f[a]:
            a, b = b, a
        a, _ = find_level_root(parent, a, f)
        b, _ = find_peak_root(parent, b, f[a], f)
        # if order_op(b, a, f):
        #    a, b = b, a
        if a == b:
            return
        assert f[b] >= f[a]
        old = parent[b]
        parent[b] = a
        b = old  # Equivalent to the function exchange(parent[b], a)


def maxtree(f: np.ndarray) -> np.ndarray:
    """
    Create a maxtree using a non-sorting algorithm.

    :param f: Starting image.
    :return: Parent Image.
    """
    parent = np.full(f.shape, None, dtype=object)
    rows, cols = f.shape
    for y in range(rows):
        for x in range(cols):
            p = (y, x)
            parent[p] = p
            for n in undef_neighbors(p, parent):
                connect(p, n, f, parent)

    return parent


def root_none(parent: np.ndarray) -> None:
    """
    Clear all the cycles between the root and itself.

    :param parent: Parent image.
    """
    rows, cols = parent.shape
    for y in range(rows):
        for x in range(cols):
            p = (y, x)
            if parent[p] == p:
                parent[p] = None
