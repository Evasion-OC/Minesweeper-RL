import sys, random, copy, time, logging, os
from itertools import combinations
from collections import defaultdict, Counter, deque
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sympy.combinatorics import Permutation, PermutationGroup
from Graph_Pattern_Generation2 import PATTERNS

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtWidgets import QSizePolicy, QScrollArea
import networkx as nx
try:
    import cudf
    import cugraph
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from pysat.solvers import Glucose3
from ortools.sat.python import cp_model
from networkx.algorithms.isomorphism import GraphMatcher
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


from z3 import Solver, Bool, Sum, If, sat

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Allow ops not yet implemented on Apple's Metal backend to silently fall back to CPU
# instead of raising. Must be set before torch is meaningfully used.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Board-size thresholds for slow / informational analyses. Tunable.
PATTERN_MATCH_MAX_CELLS = 300   # skip pattern_based_solve on boards with more cells
ANALYTICS_MAX_CELLS = 100       # skip O(N^3) automorphism + spectral analytics on first move


def pick_device():
    """Choose the best available accelerator: CUDA > Apple-Silicon MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available() and mps_backend.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def empty_device_cache(device):
    """torch.cuda.empty_cache equivalent that no-ops on non-CUDA devices."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        empty = getattr(mps_backend, "empty_cache", None) if mps_backend is not None else None
        if callable(empty):
            empty()


"START logger"

logging.basicConfig(level=logging.INFO)
if GPU_AVAILABLE:
    logging.info("GPU libraries (cudf and cugraph) found. Using GPU-accelerated graph algorithms.")
else:
    logging.info("GPU libraries not found. Falling back to CPU-based graph algorithms.")

ai_logger = logging.getLogger('ai')
calculation_logger = logging.getLogger('calculation')

ai_logger.setLevel(logging.INFO)
calculation_logger.setLevel(logging.INFO)
ai_logger.propagate = False
calculation_logger.propagate = False

"Design GUI logger QT"

class QTextEditLogger(QObject, logging.Handler):
    log_signal = pyqtSignal(str)
    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        logging.Handler.__init__(self)
        self.text_edit = None
        self._alive = True
    def set_text_edit(self, text_edit):
        self.text_edit = text_edit
        self.log_signal.connect(self.append_log)
    def append_log(self, msg):
        if not self._alive or self.text_edit is None:
            return
        try:
            self.text_edit.append(msg)
            self.text_edit.verticalScrollBar().setValue(
                self.text_edit.verticalScrollBar().maximum()
            )
        except RuntimeError:
            # Underlying Qt object was destroyed (e.g. across a Spyder runfile reload)
            self._alive = False
    def emit(self, record):
        if not self._alive:
            return
        try:
            msg = self.format(record)
            self.log_signal.emit(msg)
        except RuntimeError:
            # C++ side of this QObject is gone - disable so we no-op from now on.
            self._alive = False
    def shutdown(self):
        self._alive = False

def _purge_stale_qtext_handlers(logger):
    """Detach orphaned QTextEditLogger handlers left over from a previous run
    (Spyder's runfile reloads modules but keeps the logging.Logger singletons,
    so dead Qt-backed handlers can survive into the next run and crash on emit).
    Match by class name so this still works after `Reloaded modules` rebinds the class."""
    for h in list(logger.handlers):
        if h.__class__.__name__ == 'QTextEditLogger':
            try:
                h.shutdown()
            except Exception:
                pass
            logger.removeHandler(h)

class LogWindow(QtWidgets.QMainWindow):
    def __init__(self, logger, title="Log Window"):
        super().__init__()
        # Drop any handlers from a prior run before installing ours.
        _purge_stale_qtext_handlers(logger)
        self.setWindowTitle(title)
        self.setGeometry(150, 150, 600, 400)
        self.text_edit = QtWidgets.QTextEdit()
        self.text_edit.setReadOnly(True)
        self.setCentralWidget(self.text_edit)
        self.qtext_handler = QTextEditLogger(self)
        self.qtext_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.qtext_handler.set_text_edit(self.text_edit)
        self.logger = logger
        self.start_logging()
        self.force_close = False
    def start_logging(self):
        if self.qtext_handler not in self.logger.handlers:
            self.logger.addHandler(self.qtext_handler)
            self.logger.info("Logging started.")
    def stop_logging(self):
        if self.qtext_handler in self.logger.handlers:
            self.logger.removeHandler(self.qtext_handler)
            self.logger.info("Logging stopped.")
    def closeEvent(self, event):
        if self.force_close:
            self.stop_logging()
            event.accept()
        else:
            self.stop_logging()
            self.hide()
            event.ignore()
    def force_close_window(self):
        self.force_close = True
        self.close()
        
"END Design GUI logger QT"

"END logger"

"START ORBIT & SYMM"

def get_valid_transformations(n_rows, n_cols):
    transformations = []
    transformations.append(lambda r, c: (r, c))
    transformations.append(lambda r, c: (r, n_cols - 1 - c))
    transformations.append(lambda r, c: (n_rows - 1 - r, c))
    transformations.append(lambda r, c: (n_rows - 1 - r, n_cols - 1 - c))
    if n_rows == n_cols:
        transformations.append(lambda r, c: (c, n_rows - 1 - r))
        transformations.append(lambda r, c: (n_rows - 1 - c, r))
    calculation_logger.info(f"Computed {len(transformations)} valid transformations for board size {n_rows}x{n_cols}.")
    return transformations

def canonical_candidate(candidate_nodes, n_rows, n_cols):
    transforms = get_valid_transformations(n_rows, n_cols)
    candidate_list = list(candidate_nodes)
    all_forms = []
    for transform in transforms:
        transformed = [transform(r, c) for (r, c) in candidate_list]
        transformed_sorted = tuple(sorted(transformed))
        all_forms.append(transformed_sorted)
    canonical = min(all_forms)
    calculation_logger.info(f"Canonical candidate for {candidate_nodes} is {canonical}.")
    return canonical

def compute_node_orbits(board_graph):
    n_rows = board_graph.graph.get("n_rows")
    n_cols = board_graph.graph.get("n_cols")
    if n_rows is None or n_cols is None:
        raise ValueError("Board dimensions not stored in graph attributes")
    transforms = get_valid_transformations(n_rows, n_cols)
    orbits = {}
    for node in board_graph.nodes:
        orbit = set()
        for transform in transforms:
            orbit.add(transform(*node))
        canonical = min(orbit)
        orbits.setdefault(canonical, set()).add(node)
    calculation_logger.info(f"Computed node orbits: {len(orbits)} distinct orbits found.")
    return orbits

def cycle_index_polynomial(transformations, n_rows, n_cols):
    poly_terms = Counter()
    for t in transformations:
        cycles = []
        visited = set()
        for r in range(n_rows):
            for c in range(n_cols):
                if (r, c) in visited:
                    continue
                cycle = []
                current = (r, c)
                while current not in cycle:
                    cycle.append(current)
                    visited.add(current)
                    current = t(*current)
                cycles.append(len(cycle))
        term = tuple(sorted(cycles))
        poly_terms[term] += 1
    group_size = len(transformations)
    cycle_index = {term: count / group_size for term, count in poly_terms.items()}
    calculation_logger.info(f"Cycle index polynomial computed with {len(cycle_index)} terms.")
    return cycle_index

def compute_orbit_and_stabilizer(candidate_nodes, n_rows, n_cols):
    transforms = get_valid_transformations(n_rows, n_cols)
    orbit = set()
    stabilizer = []
    for t in transforms:
        transformed = frozenset(t(*node) for node in candidate_nodes)
        orbit.add(transformed)
        if transformed == frozenset(candidate_nodes):
            stabilizer.append(t)
    orbit_size = len(orbit)
    stabilizer_size = len(stabilizer)
    calculation_logger.info(f"Candidate {candidate_nodes} has orbit size {orbit_size} and stabilizer size {stabilizer_size}.")
    return orbit_size, stabilizer_size

def heuristic_graph_automorphisms(G):
    n_rows = G.graph.get("n_rows")
    n_cols = G.graph.get("n_cols")
    if n_rows is None or n_cols is None:
        raise ValueError("Board dimensions must be stored in graph attributes")
    return get_valid_transformations(n_rows, n_cols)

def invariant_signature(G):
    degrees = sorted([d for n, d in G.degree()])
    clues = sorted([G.nodes[n].get("clue", -1) for n in G.nodes()])
    signature = (tuple(degrees), tuple(clues))
    calculation_logger.info(f"Graph invariant signature: {signature}")
    return signature

def count_fixed_configurations(transformation, n_rows, n_cols, num_mines):
    cycles = []
    visited = set()
    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) in visited:
                continue
            orbit = []
            current = (r, c)
            while current not in orbit:
                orbit.append(current)
                visited.add(current)
                current = transformation(*current)
            cycles.append(len(orbit))
    dp = {0: 1}
    for cycle_size in cycles:
        new_dp = {}
        for s, count in dp.items():
            new_dp[s] = new_dp.get(s, 0) + count
            new_sum = s + cycle_size
            new_dp[new_sum] = new_dp.get(new_sum, 0) + count
        dp = new_dp
    fixed_count = dp.get(num_mines, 0)
    calculation_logger.info(f"Transformation cycles: {cycles} yield {fixed_count} fixed configurations with {num_mines} mines.")
    return fixed_count

def burnside_mine_count(n_rows, n_cols, num_mines):
    transformations = get_valid_transformations(n_rows, n_cols)
    total = 0
    for t in transformations:
        fixed = count_fixed_configurations(t, n_rows, n_cols, num_mines)
        total += fixed
    result = total // len(transformations)
    calculation_logger.info(f"Burnside's lemma: {result} distinct configurations up to symmetry for board size {n_rows}x{n_cols} with {num_mines} mines.")
    return result

def board_cell_index(r, c, n_cols):
    return r * n_cols + c

def index_to_coord(idx, n_cols):
    return (idx // n_cols, idx % n_cols)

def transformation_to_permutation(transform, n_rows, n_cols):
    total_cells = n_rows * n_cols
    mapping = {}
    for r in range(n_rows):
        for c in range(n_cols):
            i = board_cell_index(r, c, n_cols)
            r_new, c_new = transform(r, c)
            j = board_cell_index(r_new, c_new, n_cols)
            mapping[i] = j
    perm_list = [mapping[i] for i in range(total_cells)]
    return Permutation(perm_list)

def compute_automorphism_group_order(n_rows, n_cols):
    transforms = get_valid_transformations(n_rows, n_cols)
    perms = [transformation_to_permutation(t, n_rows, n_cols) for t in transforms]
    group = PermutationGroup(perms)
    order = group.order()
    calculation_logger.info(f"Automorphism group order (via Schreier-Sims): {order}")
    return order, group

def print_board_automorphism_info(board, adj):
    board_graph = board_to_graph(board, adj)
    n_rows = board_graph.graph["n_rows"]
    n_cols = board_graph.graph["n_cols"]
    order, group = compute_automorphism_group_order(n_rows, n_cols)
    print(f"Board automorphism group order: {order}")
    #print("Automorphism group generators:")
    for gen in group.generators:
        print(gen)
        
def canonical_board(board):
    n_rows = len(board)
    n_cols = len(board[0])
    transforms = get_valid_transformations(n_rows, n_cols)
    def board_to_tuple(b):
        return tuple(tuple((cell['clue'], cell['covered'], cell['flagged'], cell['isMine']) for cell in row) for row in b)
    forms = []
    for t in transforms:
        new_board = [[None for _ in range(n_cols)] for _ in range(n_rows)]
        for r in range(n_rows):
            for c in range(n_cols):
                new_r, new_c = t(r, c)
                new_board[new_r][new_c] = board[r][c]
        forms.append(board_to_tuple(new_board))
    canonical = min(forms)
    calculation_logger.info("Canonical board state computed.")
    return canonical

def symmetry_analysis(board, adj, logger):
    n_rows = len(board)
    n_cols = len(board[0])
    transforms = get_valid_transformations(n_rows, n_cols)
    cip = cycle_index_polynomial(transforms, n_rows, n_cols)
    logger.info("Symmetry Analysis:")
    logger.info(f"Cycle Index Polynomial: {cip}")
    order, group = compute_automorphism_group_order(n_rows, n_cols)
    logger.info(f"Automorphism Group Order: {order}")

def symmetry_reduced_moves(board, adj, logger):
    # Log the purpose of symmetry-reduced moves

    # Find covered, unflagged cells
    covered_cells = [(r, c) for r in range(len(board)) for c in range(len(board[0]))
                     if board[r][c]['covered'] and not board[r][c]['flagged']]
    
    if not covered_cells:
        logger.info("No covered, unflagged cells available for symmetry-reduced moves. Skipping.")
        return False

    # Group cells by symmetry (simplified for illustration)
    start_time = time.time()
    n_rows, n_cols = len(board), len(board[0])
    transforms = get_valid_transformations(n_rows, n_cols)
    symmetry_groups = {}
    for r, c in covered_cells:
        # Canonicalize the cell under all transformations
        all_forms = []
        for transform in transforms:
            tr, tc = transform(r, c)
            all_forms.append((tr, tc))
        canonical = tuple(sorted(all_forms)[0])
        symmetry_groups.setdefault(canonical, []).append((r, c))

    # Log the symmetry groups
    logger.info(
        f"Found {len(symmetry_groups)} symmetry groups from {len(covered_cells)} covered cells. "
        f"Group sizes: {[len(group) for group in symmetry_groups.values()]}."
    )

    # Pick a group and a cell to reveal (simplified: pick the largest group)
    if not symmetry_groups:
        logger.info("No symmetry groups identified. Skipping.")
        return False

    largest_group = max(symmetry_groups.values(), key=len)
    r, c = largest_group[0]  # Pick the first cell in the largest group
    end_time = time.time()

    logger.info(
        f"Selected cell at ({r},{c}) from a symmetry group of size {len(largest_group)}. "
        f"Other cells in the group: {largest_group[1:] if len(largest_group) > 1 else 'None'}. "
        f"Time taken: {end_time - start_time:.3f} seconds."
    )

    # Provide insights
    remaining_mines = sum(1 for row in board for cell in row if cell['isMine']) - \
                      sum(1 for row in board for cell in row if cell['flagged'])
    prob = remaining_mines / len(covered_cells) if covered_cells else 0
    logger.info(
        f"Estimated mine probability for this guess: {prob:.3f} "
        f"(based on {remaining_mines} remaining mines and {len(covered_cells)} covered cells)."
    )

    # Reveal the cell
    board[r][c]['covered'] = False
    if board[r][c]['isMine']:
        logger.error(
            f"Symmetry-reduced move hit a mine at ({r},{c}). "
            "This indicates that the symmetry-based guess was unlucky."
        )
    else:
        logger.info(
            f"Successfully revealed cell at ({r},{c}) (not a mine). "
            f"New clue: {board[r][c]['clue']}. This may enable further deductions."
        )
        if board[r][c]['clue'] == 0:
            bfs_expand(board, r, c, adj, logger)

    return True


def get_coset_representatives(board, adj):
    board_graph = board_to_graph(board, adj)
    orbits = compute_node_orbits(board_graph)
    representatives = []
    for rep, nodes in orbits.items():
        for node in nodes:
            r, c = node
            if board[r][c]['covered'] and not board[r][c]['flagged']:
                representatives.append(node)
                break
    calculation_logger.info(f"Coset decomposition: Found {len(representatives)} coset representative moves.")
    return representatives


"END ORBIT & SYMM"

"SPECT GRAPH LAP d = l - a"
def compute_board_laplacian(board_graph):
    nodes = list(board_graph.nodes())
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    L = np.zeros((n, n))
    for node in nodes:
        i = node_index[node]
        degree = board_graph.degree[node]
        L[i, i] = degree
        for neighbor in board_graph.neighbors(node):
            j = node_index[neighbor]
            L[i, j] = -1
    return L, node_index

def spectral_analysis(board_graph, k=10):
    L, node_index = compute_board_laplacian(board_graph)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    #print("Spectral Analysis:")
    
    k = min(k, len(eigenvalues))
    return eigenvalues[:k], eigenvectors[:, :k], node_index



def spectral_clustering(board_graph, num_clusters=2):
    eigenvalues, eigenvectors, node_index = spectral_analysis(board_graph, k=num_clusters)
    if num_clusters == 2:
        fiedler_vector = eigenvectors[:, 1]
        clusters = {0: [], 1: []}
        for node, idx in node_index.items():
            if fiedler_vector[idx] < 0:
                clusters[0].append(node)
            else:
                clusters[1].append(node)
    else:
        from sklearn.cluster import KMeans
        features = eigenvectors[:, 1:num_clusters]
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(features)
        labels = kmeans.labels_
        clusters = {}
        for node, idx in node_index.items():
            label = labels[idx]
            clusters.setdefault(label, []).append(node)
    #print("Spectral Clustering Results:", clusters)
    return clusters




def board_to_graph(board, adj):
    G = nx.Graph()
    for r in range(len(board)):
        for c in range(len(board[0])):
            attrs = {
                'clue': board[r][c]['clue'],
                'covered': board[r][c]['covered'],
                'flagged': board[r][c]['flagged'],
                'isMine': board[r][c]['isMine'],
                'pos': (r, c)
            }
            G.add_node((r, c), **attrs)
            for nr, nc in adj[r][c]:
                G.add_edge((r, c), (nr, nc))
    G.graph["n_rows"] = len(board)
    G.graph["n_cols"] = len(board[0]) if board else 0
    return G

"DESIGNING BFS & RUNNING FLOODFILL ALGORITHM THROUGH IT"

def bfs_expand(board, r, c, adj, logger):
    """Flood-fill from a 0-clue cell. Caller is expected to have already revealed (r, c)."""
    if board[r][c]['clue'] != 0:
        return 0

    revealed_count = 0
    queue = deque([(r, c)])
    visited = set()

    while queue:
        curr_r, curr_c = queue.popleft()
        if (curr_r, curr_c) in visited:
            continue
        visited.add((curr_r, curr_c))

        cell = board[curr_r][curr_c]
        # Reveal if currently covered (and not flagged); count it
        if cell['covered'] and not cell['flagged']:
            cell['covered'] = False
            revealed_count += 1

        # Expand only through 0-clue cells
        if cell['clue'] == 0 and not cell['flagged']:
            for nr, nc in adj[curr_r][curr_c]:
                if (nr, nc) not in visited:
                    queue.append((nr, nc))

    logger.info(f"BFS expanded from ({r}, {c}), revealed {revealed_count} cells")
    return revealed_count

"GRAPH COLORING & DECOMPOSITION, PREF IS USING CUDA, IF NOT AVAIL (cugraph, cudf) WILL SWITCH TO 2D BASED"

def gpu_graph_decomposition_coloring(board_graph, logger):


    
    num_nodes = board_graph.number_of_nodes()
    num_edges = board_graph.number_of_edges()
    logger.info(
        f"Board graph statistics: {num_nodes} nodes (cells), {num_edges} edges (adjacent cell connections)."
    )

    
    node_to_id = {node: i for i, node in enumerate(board_graph.nodes())}
    edges = [(node_to_id[u], node_to_id[v]) for u, v in board_graph.edges()]
    logger.debug(f"Prepared edge list for cuGraph: {len(edges)} edges.")

    
    df = cudf.DataFrame({'src': [u for u, v in edges], 'dst': [v for u, v in edges]})
    G_cu = cugraph.Graph()
    G_cu.from_cudf_edgelist(df, source='src', destination='dst', edge_attr=None)
    logger.debug("Converted board graph to cuGraph format for GPU processing.")

    
    df_cc = cugraph.connected_components(G_cu)
    id_to_node = {v: k for k, v in node_to_id.items()}
    component_dict = {id_to_node[int(row['vertex'])]: int(row['component']) for index, row in df_cc.iterrows()}
    logger.debug(f"Computed connected components using cuGraph: {len(component_dict)} nodes assigned to components.")

    
    component_groups = {}
    for node, comp in component_dict.items():
        component_groups.setdefault(comp, []).append(node)
    num_components = len(component_groups)
    logger.info(
        f"Decomposed the board into {num_components} connected components using GPU acceleration. "
        "Each component represents a region of the board that can be solved independently."
    )

    # If there are no components, log a warning and return an empty color map
    if num_components == 0:
        logger.warning("No connected components found in the board graph. The board may be empty or fully revealed.")
        return {}

    # Analyze component sizes for insights
    component_sizes = [len(nodes) for nodes in component_groups.values()]
    max_size = max(component_sizes) if component_sizes else 0
    min_size = min(component_sizes) if component_sizes else 0
    avg_size = sum(component_sizes) / len(component_sizes) if component_sizes else 0
    logger.info(
        f"Component size summary: Largest component has {max_size} nodes, smallest has {min_size} nodes, "
        f"average size is {avg_size:.2f} nodes."
    )
    if max_size > num_nodes * 0.5:
        logger.info(
            f"Note: The largest component contains {max_size/num_nodes*100:.1f}% of the board's nodes. "
            "This may indicate a highly connected region, which could be computationally challenging to solve."
        )
    if min_size == 1:
        logger.info(
            "Note: Some components have only 1 node. These are isolated cells (e.g., fully surrounded by revealed cells)."
        )

    # Color each component (using CPU-based NetworkX for coloring)
    color_map = {}
    for comp, nodes in sorted(component_groups.items()):
        # Log component details
        subgraph = board_graph.subgraph(nodes)
        num_nodes_in_component = subgraph.number_of_nodes()
        num_edges_in_component = subgraph.number_of_edges()
        logger.info(
            f"Processing Component {comp} (GPU-identified): {num_nodes_in_component} nodes, "
            f"{num_edges_in_component} edges."
        )

        # Perform graph coloring
        coloring_result = nx.algorithms.coloring.greedy_color(subgraph, strategy='largest_first')
        num_colors_used = max(coloring_result.values()) + 1 if coloring_result else 0
        logger.info(
            f"Component {comp}: Colored with {num_colors_used} colors using the 'largest first' strategy (CPU-based)."
        )

        # Summarize color distribution
        if coloring_result:
            color_counts = {}
            for node, color in coloring_result.items():
                color_counts[color] = color_counts.get(color, 0) + 1
            color_summary = ", ".join(
                f"Color {color}: {count} nodes" for color, count in sorted(color_counts.items())
            )
            logger.info(f"Component {comp} color distribution: {color_summary}.")

        # Add to the overall color map
        color_map.update(coloring_result)

        # Log potential insights
        if num_colors_used == 1 and num_nodes_in_component > 1:
            logger.info(
                f"Component {comp}: All {num_nodes_in_component} nodes assigned the same color. "
                "This may indicate a simple structure (e.g., a chain or clique)."
            )
        if num_colors_used > num_nodes_in_component * 0.5:
            logger.info(
                f"Component {comp}: Used {num_colors_used} colors for {num_nodes_in_component} nodes. "
                "A high number of colors may indicate a complex structure, potentially slowing down constraint solving."
            )

    # Log a summary of the entire coloring process
    total_colors = max(color_map.values()) + 1 if color_map else 0
    logger.info(
        f"GPU-accelerated graph decomposition and coloring completed. "
        f"Total unique colors used across all components: {total_colors}. "
        f"Total nodes colored: {len(color_map)}."
    )

    return color_map

def graph_decomposition_coloring(board_graph, logger):
    
    def cpu_decomposition(graph, log_handler):
        """
        CPU-based graph decomposition and coloring.
        """
        # Log the purpose of graph decomposition and coloring
        log_handler.info(
            "Starting CPU-based graph decomposition and coloring using NetworkX. "
            "This step breaks the board into independent regions (connected components) and assigns colors to nodes "
            "to help optimize constraint solving. Each color represents a group of nodes that can be processed independently."
        )

        # Log basic graph statistics
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        log_handler.info(
            f"Board graph statistics: {num_nodes} nodes (cells), {num_edges} edges (adjacent cell connections)."
        )

        # Find connected components
        components = list(nx.connected_components(graph))
        num_components = len(components)
        log_handler.info(
            f"Decomposed the board into {num_components} connected components. "
            "Each component represents a region of the board that can be solved independently."
        )

        # If there are no components, log a warning and return an empty color map
        if num_components == 0:
            log_handler.warning("No connected components found in the board graph. The board may be empty or fully revealed.")
            return {}

        # Analyze component sizes for insights
        component_sizes = [len(component) for component in components]
        max_size = max(component_sizes) if component_sizes else 0
        min_size = min(component_sizes) if component_sizes else 0
        avg_size = sum(component_sizes) / len(component_sizes) if component_sizes else 0
        log_handler.info(
            f"Component size summary: Largest component has {max_size} nodes, smallest has {min_size} nodes, "
            f"average size is {avg_size:.2f} nodes."
        )
        if max_size > num_nodes * 0.5:
            log_handler.info(
                f"Note: The largest component contains {max_size/num_nodes*100:.1f}% of the board's nodes. "
                "This may indicate a highly connected region, which could be computationally challenging to solve."
            )
        if min_size == 1:
            log_handler.info(
                "Note: Some components have only 1 node. These are isolated cells (e.g., fully surrounded by revealed cells)."
            )

        # Color each component
        color_map = {}
        for idx, component in enumerate(components, 1):
            # Log component details
            subgraph = graph.subgraph(component)
            num_nodes_in_component = subgraph.number_of_nodes()
            num_edges_in_component = subgraph.number_of_edges()
            log_handler.info(
                f"Processing Component {idx}/{num_components}: {num_nodes_in_component} nodes, "
                f"{num_edges_in_component} edges."
            )

            # Perform graph coloring
            coloring_result = nx.algorithms.coloring.greedy_color(subgraph, strategy='largest_first')
            num_colors_used = max(coloring_result.values()) + 1 if coloring_result else 0
            log_handler.info(
                f"Component {idx}: Colored with {num_colors_used} colors using the 'largest first' strategy."
            )

            # Summarize color distribution
            if coloring_result:
                color_counts = {}
                for node, color in coloring_result.items():
                    color_counts[color] = color_counts.get(color, 0) + 1
                color_summary = ", ".join(
                    f"Color {color}: {count} nodes" for color, count in sorted(color_counts.items())
                )
                log_handler.info(f"Component {idx} color distribution: {color_summary}.")

            # Add to the overall color map
            color_map.update(coloring_result)

            # Log potential insights
            if num_colors_used == 1 and num_nodes_in_component > 1:
                log_handler.info(
                    f"Component {idx}: All {num_nodes_in_component} nodes assigned the same color. "
                    "This may indicate a simple structure (e.g., a chain or clique)."
                )
            if num_colors_used > num_nodes_in_component * 0.5:
                log_handler.info(
                    f"Component {idx}: Used {num_colors_used} colors for {num_nodes_in_component} nodes. "
                    "A high number of colors may indicate a complex structure, potentially slowing down constraint solving."
                )

        # Log a summary of the entire coloring process
        total_colors = max(color_map.values()) + 1 if color_map else 0
        log_handler.info(
            f"CPU-based graph decomposition and coloring completed. "
            f"Total unique colors used across all components: {total_colors}. "
            f"Total nodes colored: {len(color_map)}."
        )

        return color_map

    # Decide whether to use GPU or CPU
    if GPU_AVAILABLE:
        #logger.info("GPU is available. Attempting GPU-accelerated graph decomposition and coloring.")
        try:
            return gpu_graph_decomposition_coloring(board_graph, logger)
        except Exception as e:
            logger.error(
                f"GPU-accelerated decomposition and coloring failed: {str(e)}. "
                "Falling back to CPU-based implementation."
            )
            return cpu_decomposition(board_graph, logger)
    else:
        #logger.info("GPU is not available. Using CPU-based graph decomposition and coloring.")
        return cpu_decomposition(board_graph, logger)

def graph_decomposition_coloring_wrapper(board_graph, logger):
    """
    Wrapper for graph decomposition and coloring, allowing for GPU or CPU implementation.
    
    Args:
        board_graph (networkx.Graph): The graph representation of the Minesweeper board.
        logger (logging.Logger): Logger for outputting information.
    
    Returns:
        dict: A mapping of nodes to their assigned colors.
    """
    logger.info(
        "Initiating graph decomposition and coloring process. "
        "The implementation (GPU or CPU) will be determined based on availability."
    )
    return graph_decomposition_coloring(board_graph, logger)

"FOR CACHING"

def compute_subgraph_signature(subgraph):
    degree_seq = tuple(sorted([d for n, d in subgraph.degree()]))
    clues = tuple(sorted([data['clue'] for n, data in subgraph.nodes(data=True) if data.get('clue', -1) >= 0]))
    return hash((degree_seq, clues))


"FILTERING RUNS THROUGH PATTERNS DEFINED ABOVE WITH ISOMORPHISM & SUBGRAPH ISOMORPHISM"

"WITH FULL PATTERNS PROCESSING THIS PART WILL BE EXTREMELY COMPUTATIONALLY CHALLENGING AND NEEDS AT LEAST 32GB OF MEMORY"

def generic_candidate_filter(board_graph, pattern, logger):
    candidates = []
    seen = set()
    pattern_node_attributes = pattern.nodes(data=True)
    pattern_num_nodes = pattern.number_of_nodes()
    pattern_degree_sequence = sorted([d for n, d in pattern.degree()])
    pattern_clues = sorted([data['clue'] for n, data in pattern_node_attributes if data.get('clue', -1) >= 0])
    pattern_signature = hash((tuple(pattern_degree_sequence), tuple(pattern_clues)))
    logger.debug(f"Generic filter: Pattern with {pattern_num_nodes} nodes and signature {pattern_signature}")
    n_rows = board_graph.graph["n_rows"]
    n_cols = board_graph.graph["n_cols"]
    invariant_cache = {}
    for node in board_graph.nodes:
        node_clue = board_graph.nodes[node].get('clue', -1)
        if node_clue not in pattern_clues and node_clue != -1:
            continue
        sub_nodes = set([node])
        for neighbor in board_graph.neighbors(node):
            sub_nodes.add(neighbor)
            if len(sub_nodes) == pattern_num_nodes:
                break
        if len(sub_nodes) != pattern_num_nodes:
            continue
        can_form = canonical_candidate(sub_nodes, n_rows, n_cols)
        if can_form in seen:
            continue
        seen.add(can_form)
        subgraph = board_graph.subgraph(sub_nodes)
        if can_form in invariant_cache:
            sub_signature = invariant_cache[can_form]
        else:
            sub_signature = compute_subgraph_signature(subgraph)
            invariant_cache[can_form] = sub_signature
        if sub_signature != pattern_signature:
            continue
        candidates.append(sub_nodes)
    logger.info(f"Generic candidate filter found {len(candidates)} candidates.")
    return candidates

def heuristic_filter_candidates(board_graph, pattern_info, logger):
    if isinstance(pattern_info, dict) and 'candidate_extractor' in pattern_info:
        candidates = pattern_info['candidate_extractor'](board_graph)
        logger.info(f"Custom candidate extractor returned {len(candidates)} candidates.")
        return candidates
    return generic_candidate_filter(board_graph, pattern_info, logger)

def find_pattern_matches_heuristic(board_graph, pattern_info, logger):
    matches = []
    candidates = heuristic_filter_candidates(board_graph, pattern_info, logger)
    for candidate_nodes in candidates:
        subgraph = board_graph.subgraph(candidate_nodes)
        matcher = GraphMatcher(subgraph, 
                               pattern_info['graph'] if isinstance(pattern_info, dict) else pattern_info,
                               node_match=lambda n1, n2: node_match(n1, n2))
        if matcher.is_isomorphic():
            for mapping in matcher.isomorphisms_iter():
                full_mapping = {pattern_node: board_node for pattern_node, board_node in mapping.items()}
                matches.append(full_mapping)
    logger.info(f"Pattern matching found {len(matches)} isomorphic matches.")
    return matches

def node_match(n1_attrs, n2_attrs):
    if n2_attrs.get('isMine', False):
        return n1_attrs.get('isMine', False) or n1_attrs.get('flagged', False)
    if n1_attrs.get('clue', -1) != n2_attrs.get('clue', -1):
        return False
    if n1_attrs.get('covered') != n2_attrs.get('covered'):
        return False
    return True

"SEQUENTIAL PATTERN MATCHING (Python's GIL serializes pure-Python NetworkX work,"
" so a thread pool only added overhead and log-handler contention)"

def parallel_find_pattern_matches(board_graph, patterns, logger, max_workers=None):
    """Run each pattern's matcher in priority order. The `max_workers` parameter
    is kept for API compat but unused - execution is sequential."""
    del max_workers  # ignored; kept for backwards compat with old callers
    sorted_patterns = sorted(
        patterns.items(),
        key=lambda kv: kv[1].get('priority', 100) if isinstance(kv[1], dict) else 100
    )
    results = {}
    start_time = time.time()
    for pname, pinfo in sorted_patterns:
        try:
            results[pname] = find_pattern_matches_heuristic(board_graph, pinfo, logger)
        except Exception as exc:
            logger.error(f"Pattern '{pname}' raised: {exc}")
            results[pname] = []
    elapsed = time.time() - start_time
    total = sum(len(m) for m in results.values())
    logger.info(f"Pattern matching: {len(patterns)} patterns, {total} matches, {elapsed:.2f}s")
    return results

def pattern_based_solve(board, adj, logger):
    n_rows, n_cols = len(board), len(board[0])
    cells = n_rows * n_cols
    if cells > PATTERN_MATCH_MAX_CELLS:
        # Pattern matching is heuristic - on Expert/Extreme it costs many seconds and
        # the SAT/CP-SAT solvers downstream cover the same deductions. Skip it there.
        logger.info(f"Pattern-based solve skipped ({cells} cells > {PATTERN_MATCH_MAX_CELLS}).")
        return False

    start_time = time.time()
    board_graph = board_to_graph(board, adj)
    matches_dict = parallel_find_pattern_matches(board_graph, PATTERNS, logger)
    elapsed = time.time() - start_time
    total_matches = sum(len(m) for m in matches_dict.values())
    logger.info(f"pattern_based_solve: {total_matches} matches in {elapsed:.2f}s")

    changed = False
    for pattern_name, matches in matches_dict.items():
        if not matches:
            continue
        for mapping in matches:
            covered = [n for n in mapping.values() if board_graph.nodes[n]['covered']]
            if len(covered) == 1:
                r, c = covered[0]
                if not board[r][c]['flagged']:
                    board[r][c]['flagged'] = True
                    changed = True
                    logger.info(f"Pattern '{pattern_name}': flagged ({r},{c}) as mine.")

    if changed:
        logger.info(
            f"Pattern-based solve flagged "
            f"{sum(1 for row in board for cell in row if cell['flagged'])} cells total."
        )
    return changed

"LOGIC SOLVERS TO COMPLEMENT THE PATTERN MATCHING"

def _reconstruct_sat_problem(board, adj, var_map):
    solver = Glucose3()
    for r in range(len(board)):
        for c in range(len(board[0])):
            if not board[r][c]['covered'] and board[r][c]['clue'] >= 0:
                neighbors = adj[r][c]
                vars_in_constraint = []
                flagged_count = 0
                for nr, nc in neighbors:
                    if board[nr][nc]['flagged']:
                        flagged_count += 1
                    elif board[nr][nc]['covered']:
                        if (nr, nc) in var_map:
                            vars_in_constraint.append(var_map[(nr, nc)])
                required_mines = board[r][c]['clue'] - flagged_count
                if required_mines < 0 or required_mines > len(vars_in_constraint):
                    solver.delete()
                    return solver, False

                if required_mines > 0:
                    n = len(vars_in_constraint)
                    for comb in combinations(vars_in_constraint, n - required_mines + 1):
                        solver.add_clause(list(comb))
                if required_mines < len(vars_in_constraint):
                    for combo in combinations(vars_in_constraint, required_mines + 1):
                        solver.add_clause([-vv for vv in combo])
    return solver, True


def sat_based_solve(board, adj, logger):
    logger.info("Starting SAT-based solving...")
    solver = Glucose3()
    var_map = {}
    var_counter = 1
    for r in range(len(board)):
        for c in range(len(board[0])):
            if board[r][c]['covered'] and not board[r][c]['flagged']:
                var_map[(r, c)] = var_counter
                var_counter += 1
    for r in range(len(board)):
        for c in range(len(board[0])):
            if not board[r][c]['covered'] and board[r][c]['clue'] >= 0:
                neighbors = adj[r][c]
                vars_in_constraint = []
                flagged_count = 0
                for nr, nc in neighbors:
                    if board[nr][nc]['flagged']:
                        flagged_count += 1
                    elif board[nr][nc]['covered']:
                        vars_in_constraint.append(var_map[(nr, nc)])
                required_mines = board[r][c]['clue'] - flagged_count
                if required_mines < 0 or required_mines > len(vars_in_constraint):
                    continue
                if required_mines > 0:
                    solver.add_clause(vars_in_constraint.copy())
                if required_mines < len(vars_in_constraint):
                    for combo in combinations(vars_in_constraint, required_mines + 1):
                        solver.add_clause([-v for v in combo])
    if not solver.solve():
        logger.error("SAT: No solution exists under current constraints.")
        solver.delete()
        return False
    model = solver.get_model()
    solver.delete()
    changes_made = False
    for cell, var in var_map.items():
        if var in model:
            temp_solver, reconstruct_ok = _reconstruct_sat_problem(board, adj, var_map)
            if not reconstruct_ok:
                temp_solver.delete()
                continue
            temp_solver.add_clause([-var])
            if not temp_solver.solve():
                r, c = cell
                if not board[r][c]['flagged']:
                    board[r][c]['flagged'] = True
                    changes_made = True
                    logger.info(f"SAT: Flagged cell at ({r},{c}) as mine.")
                    calculation_logger.info(f"SAT: Flagged cell at ({r},{c}) as mine.")
            temp_solver.delete()
        else:
            temp_solver, reconstruct_ok = _reconstruct_sat_problem(board, adj, var_map)
            if not reconstruct_ok:
                temp_solver.delete()
                continue
            temp_solver.add_clause([var])
            if not temp_solver.solve():
                r, c = cell
                if board[r][c]['covered'] and not board[r][c]['flagged']:
                    board[r][c]['covered'] = False
                    changes_made = True
                    logger.info(f"SAT: Revealed cell at ({r},{c}) as safe.")
                    calculation_logger.info(f"SAT: Revealed cell at ({r},{c}) as safe.")
                    if board[r][c]['clue'] == 0:
                        bfs_expand(board, r, c, adj, logger)
            temp_solver.delete()
    logger.info("SAT-based solving completed.")
    return changes_made

def smt_based_solve(board, adj, logger):
    logger.info("Starting SMT-based solving using Z3...")
    solver = Solver()
    var_map = {}
    for r in range(len(board)):
        for c in range(len(board[0])):
            if board[r][c]['covered'] and not board[r][c]['flagged']:
                var_map[(r, c)] = Bool(f"cell_{r}_{c}")
    for r in range(len(board)):
        for c in range(len(board[0])):
            if not board[r][c]['covered'] and board[r][c]['clue'] >= 0:
                neighbors = adj[r][c]
                covered_vars = []
                flagged_count = 0
                for nr, nc in neighbors:
                    if board[nr][nc]['flagged']:
                        flagged_count += 1
                    elif board[nr][nc]['covered']:
                        if (nr, nc) in var_map:
                            covered_vars.append(var_map[(nr, nc)])
                required = board[r][c]['clue'] - flagged_count
                if required < 0 or required > len(covered_vars):
                    continue
                solver.add(Sum([If(v, 1, 0) for v in covered_vars]) == required)
    if solver.check() != sat:
        logger.error("SMT: No solution exists under current constraints.")
        return False
    changes_made = False
    for cell, var in var_map.items():
        s1 = Solver()
        s1.add(solver.assertions())
        s1.add(var == False)
        if s1.check() != sat:
            r, c = cell
            if not board[r][c]['flagged']:
                board[r][c]['flagged'] = True
                changes_made = True
                logger.info(f"SMT: Flagged cell at ({r},{c}) as mine.")
        s2 = Solver()
        s2.add(solver.assertions())
        s2.add(var == True)
        if s2.check() != sat:
            r, c = cell
            if board[r][c]['covered'] and not board[r][c]['flagged']:
                board[r][c]['covered'] = False
                changes_made = True
                logger.info(f"SMT: Revealed cell at ({r},{c}) as safe.")
                if board[r][c]['clue'] == 0:
                    bfs_expand(board, r, c, adj, logger)
    logger.info("SMT-based solving completed.")
    return changes_made

def apply_constraint_logic(board, adj, logger):
    changed = False
    rows, cols = len(board), len(board[0])
    for r in range(rows):
        for c in range(cols):
            if board[r][c]['covered'] or board[r][c]['isMine']:
                continue
            clue = board[r][c]['clue']
            neighbors = adj[r][c]
            flagged_count = 0
            covered_list = []
            for nr, nc in neighbors:
                if board[nr][nc]['flagged']:
                    flagged_count += 1
                elif board[nr][nc]['covered']:
                    covered_list.append((nr, nc))
            if flagged_count == clue and covered_list:
                for (nr, nc) in covered_list:
                    board[nr][nc]['covered'] = False
                    changed = True
                    logger.info(f"Constraint logic: Revealed cell at ({nr},{nc})")
                    calculation_logger.info(f"Constraint logic: Revealed cell at ({nr},{nc})")
                    if board[nr][nc]['clue'] == 0:
                        bfs_expand(board, nr, nc, adj, logger)
            elif flagged_count + len(covered_list) == clue and covered_list:
                for (nr, nc) in covered_list:
                    if not board[nr][nc]['flagged']:
                        board[nr][nc]['flagged'] = True
                        changed = True
                        logger.info(f"Constraint logic: Flagged cell at ({nr},{nc})")
                        calculation_logger.info(f"Constraint logic: Flagged cell at ({nr},{nc})")
    return changed

def advanced_constraint_solving(board, adj, logger):
    constraints = []
    variables = {}
    var_count = 0
    for r in range(len(board)):
        for c in range(len(board[0])):
            if not board[r][c]['covered'] and board[r][c]['clue'] > 0:
                neighbors = [(nr, nc) for nr, nc in adj[r][c] if board[nr][nc]['covered'] and not board[nr][nc]['flagged']]
                if neighbors:
                    clue = board[r][c]['clue'] - sum(1 for nr, nc in adj[r][c] if board[nr][nc]['flagged'])
                    var_indices = []
                    for cell in neighbors:
                        if cell not in variables:
                            variables[cell] = var_count
                            var_count += 1
                        var_indices.append(variables[cell])
                    constraints.append((var_indices, clue))
    solution = {}
    if constraints:
        var_map = {v: k for k, v in variables.items()}
        remaining_mines = sum(board[r][c]['covered'] and not board[r][c]['flagged']
                              for r in range(len(board))
                              for c in range(len(board[0])))
        if remaining_mines <= 15:
            for combo in combinations(variables.keys(), remaining_mines):
                valid = True
                for vars_list, clue in constraints:
                    count = sum(1 for var in vars_list if var_map[var] in [variables[c] for c in combo])
                    if count != clue:
                        valid = False
                        break
                if valid:
                    for cell in variables:
                        solution[cell] = 1 if cell in combo else 0
                    break
    changed = False
    if solution:
        for cell, value in solution.items():
            r, c = cell
            if value == 1 and not board[r][c]['flagged']:
                board[r][c]['flagged'] = True
                changed = True
                logger.info(f"Advanced Constraint: Flagged cell at ({r}, {c})")
                calculation_logger.info(f"Advanced Constraint: Flagged cell at ({r}, {c})")
            elif value == 0 and board[r][c]['covered']:
                board[r][c]['covered'] = False
                changed = True
                logger.info(f"Advanced Constraint: Revealed cell at ({r}, {c})")
                calculation_logger.info(f"Advanced Constraint: Revealed cell at ({r}, {c})")
                if board[r][c]['clue'] == 0:
                    bfs_expand(board, r, c, adj, logger)
    return changed

def cp_sat_solve(board, adj, logger):
    model = cp_model.CpModel()
    var_map = {}
    for r in range(len(board)):
        for c in range(len(board[0])):
            if board[r][c]['covered'] and not board[r][c]['flagged']:
                var_map[(r, c)] = model.NewBoolVar(f"cell_{r}_{c}")
    for r in range(len(board)):
        for c in range(len(board[0])):
            if not board[r][c]['covered'] and board[r][c]['clue'] >= 0:
                flagged_count = 0
                vars_in_constraint = []
                for nr, nc in adj[r][c]:
                    if board[nr][nc]['flagged']:
                        flagged_count += 1
                    elif board[nr][nc]['covered']:
                        if (nr, nc) in var_map:
                            vars_in_constraint.append(var_map[(nr, nc)])
                required = board[r][c]['clue'] - flagged_count
                if required < 0 or required > len(vars_in_constraint):
                    continue
                model.Add(sum(vars_in_constraint) == required)
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status not in [cp_model.FEASIBLE, cp_model.OPTIMAL]:
        logger.error("CP-SAT: No solution found.")
        return False
    changes_made = False
    for cell, var in var_map.items():
        model_copy = cp_model.CpModel()
        var_map_copy = {}
        for r in range(len(board)):
            for c in range(len(board[0])):
                if board[r][c]['covered'] and not board[r][c]['flagged']:
                    var_map_copy[(r, c)] = model_copy.NewBoolVar(f"cell_{r}_{c}")
        for r in range(len(board)):
            for c in range(len(board[0])):
                if not board[r][c]['covered'] and board[r][c]['clue'] >= 0:
                    flagged_count = 0
                    vars_in_constraint = []
                    for nr, nc in adj[r][c]:
                        if board[nr][nc]['flagged']:
                            flagged_count += 1
                        elif board[nr][nc]['covered']:
                            if (nr, nc) in var_map_copy:
                                vars_in_constraint.append(var_map_copy[(nr, nc)])
                    required = board[r][c]['clue'] - flagged_count
                    if required < 0 or required > len(vars_in_constraint):
                        continue
                    model_copy.Add(sum(vars_in_constraint) == required)
        if cell in var_map_copy:
            model_copy.Add(var_map_copy[cell] == 0)
            solver2 = cp_model.CpSolver()
            status2 = solver2.Solve(model_copy)
            if status2 == cp_model.INFEASIBLE:
                r, c = cell
                if not board[r][c]['flagged']:
                    board[r][c]['flagged'] = True
                    changes_made = True
                    logger.info(f"CP-SAT: Flagged cell at ({r}, {c}) as mine.")
            else:
                model_copy = cp_model.CpModel()
                var_map_copy = {}
                for r in range(len(board)):
                    for c in range(len(board[0])):
                        if board[r][c]['covered'] and not board[r][c]['flagged']:
                            var_map_copy[(r, c)] = model_copy.NewBoolVar(f"cell_{r}_{c}")
                for r in range(len(board)):
                    for c in range(len(board[0])):
                        if not board[r][c]['covered'] and board[r][c]['clue'] >= 0:
                            flagged_count = 0
                            vars_in_constraint = []
                            for nr, nc in adj[r][c]:
                                if board[nr][nc]['flagged']:
                                    flagged_count += 1
                                elif board[nr][nc]['covered']:
                                    if (nr, nc) in var_map_copy:
                                        vars_in_constraint.append(var_map_copy[(nr, nc)])
                            required = board[r][c]['clue'] - flagged_count
                            if required < 0 or required > len(vars_in_constraint):
                                continue
                            model_copy.Add(sum(vars_in_constraint) == required)
                model_copy.Add(var_map_copy[cell] == 1)
                solver3 = cp_model.CpSolver()
                status3 = solver3.Solve(model_copy)
                if status3 == cp_model.INFEASIBLE:
                    r, c = cell
                    if board[r][c]['covered']:
                        board[r][c]['covered'] = False
                        changes_made = True
                        logger.info(f"CP-SAT: Revealed cell at ({r}, {c}) as safe.")
                        if board[r][c]['clue'] == 0:
                            bfs_expand(board, r, c, adj, logger)
    return changes_made

def endgame_solver(board, adj, remaining_mines, logger):
    frontier = [(r, c) for r in range(len(board))
                for c in range(len(board[0]))
                if board[r][c]['covered'] and not board[r][c]['flagged']]
    if len(frontier) == 0 or remaining_mines == 0:
        return False
    if len(frontier) == remaining_mines:
        for r, c in frontier:
            board[r][c]['flagged'] = True
            logger.info(f"Flagged cell at ({r}, {c}) as it's a remaining mine")
            calculation_logger.info(f"Flagged cell at ({r}, {c}) as it's a remaining mine")
        return True
    changed = False
    for cell in frontier:
        # If forcing this cell to BE a mine produces a contradiction, the cell is provably safe.
        tb_mine = copy.deepcopy(board)
        tb_mine[cell[0]][cell[1]]['flagged'] = True
        if not is_valid_partial_config(tb_mine, adj, remaining_mines - 1):
            r, c = cell
            board[r][c]['covered'] = False
            changed = True
            logger.info(f"Revealed cell at ({r}, {c}) as it's safe in endgame solver")
            calculation_logger.info(f"Revealed cell at ({r}, {c}) as it's safe in endgame solver")
            if board[r][c]['clue'] == 0:
                bfs_expand(board, r, c, adj, logger)
            continue
        # If forcing this cell to NOT be a mine produces a contradiction, the cell must be a mine.
        tb_safe = copy.deepcopy(board)
        tb_safe[cell[0]][cell[1]]['covered'] = False
        tb_safe[cell[0]][cell[1]]['isMine'] = False
        tb_safe[cell[0]][cell[1]]['clue'] = 0
        if not is_valid_partial_config(tb_safe, adj, remaining_mines):
            r, c = cell
            board[r][c]['flagged'] = True
            changed = True
            logger.info(f"Flagged cell at ({r}, {c}) as mine in endgame solver")
            calculation_logger.info(f"Flagged cell at ({r}, {c}) as mine in endgame solver")
    return changed

def is_valid_partial_config(board, adj, remaining_mines):
    """Return False if the current partial assignment is provably impossible."""
    if remaining_mines < 0:
        return False
    total_covered_unflagged = 0
    for r in range(len(board)):
        for c in range(len(board[0])):
            if board[r][c]['covered'] and not board[r][c]['flagged']:
                total_covered_unflagged += 1
            if not board[r][c]['covered'] and board[r][c]['clue'] >= 0:
                clue = board[r][c]['clue']
                flagged_count = 0
                covered_unflagged = 0
                for nr, nc in adj[r][c]:
                    if board[nr][nc]['flagged']:
                        flagged_count += 1
                    elif board[nr][nc]['covered']:
                        covered_unflagged += 1
                if flagged_count > clue:
                    return False
                if flagged_count + covered_unflagged < clue:
                    return False
    if remaining_mines > total_covered_unflagged:
        return False
    return True

def belief_propagation_probabilities(board, adj, logger, total_mines, iterations=10):
    covered_cells = [(r, c) for r in range(len(board)) for c in range(len(board[0]))
                     if board[r][c]['covered'] and not board[r][c]['flagged']]
    flagged_count = sum(cell['flagged'] for row in board for cell in row)
    remaining_mines = max(0, total_mines - flagged_count)
    if not covered_cells:
        return {}
    uniform_prob = remaining_mines / len(covered_cells)
    beliefs = {cell: uniform_prob for cell in covered_cells}
    for _ in range(iterations):
        new_beliefs = beliefs.copy()
        for r in range(len(board)):
            for c in range(len(board[0])):
                if not board[r][c]['covered'] and board[r][c]['clue'] >= 0:
                    nb = adj[r][c]
                    flagged_ = sum(board[nr][nc]['flagged'] for nr, nc in nb)
                    covered_ = [(nr, nc) for nr, nc in nb if board[nr][nc]['covered'] and not board[nr][nc]['flagged']]
                    if covered_:
                        needed = board[r][c]['clue'] - flagged_
                        if needed >= 0:
                            val = needed / len(covered_)
                            for ncell in covered_:
                                new_beliefs[ncell] = (beliefs[ncell] + val) / 2
        beliefs = new_beliefs
    return beliefs

def factor_graph_belief_propagation_solver(board, adj, logger, iterations=10):
    logger.info("Starting Factor Graph Belief Propagation Solver...")
    variables = {}
    factors = []
    for r in range(len(board)):
        for c in range(len(board[0])):
            if board[r][c]['covered'] and not board[r][c]['flagged']:
                variables[(r, c)] = {'belief': 0.5}
    for r in range(len(board)):
        for c in range(len(board[0])):
            if not board[r][c]['covered'] and board[r][c]['clue'] >= 0:
                nb = adj[r][c]
                var_neighbors = [ (nr, nc) for nr, nc in nb if (nr, nc) in variables ]
                flagged = sum(1 for nr, nc in nb if board[nr][nc]['flagged'])
                required = board[r][c]['clue'] - flagged
                if var_neighbors:
                    factors.append({'neighbors': var_neighbors, 'required': required})
    messages = {}
    for factor in factors:
        for var in factor['neighbors']:
            messages[(('factor', tuple(factor['neighbors']), factor['required']), var)] = 0.5
    for _ in range(iterations):
        for factor in factors:
            neighbors = factor['neighbors']
            required = factor['required']
            for var in neighbors:
                others = [v for v in neighbors if v != var]
                sum_others = sum(variables[v]['belief'] for v in others)
                message = min(1.0, max(0.0, (required - (len(others) - sum_others)) / (sum_others + 1e-5)))
                messages[(('factor', tuple(neighbors), required), var)] = message
        for var in variables:
            incoming = [messages[(key, var)] for key in { key for (key, v) in messages if (key, var) in messages }]
            if incoming:
                variables[var]['belief'] = sum(incoming) / len(incoming)
    changed = False
    for var, data in variables.items():
        r, c = var
        if data['belief'] < 0.1 and board[r][c]['covered']:
            board[r][c]['covered'] = False
            changed = True
            logger.info(f"Factor BP: Revealed cell at ({r},{c}) with belief {data['belief']:.2f}")
            if board[r][c]['clue'] == 0:
                bfs_expand(board, r, c, adj, logger)
        elif data['belief'] > 0.9 and not board[r][c]['flagged']:
            board[r][c]['flagged'] = True
            changed = True
            logger.info(f"Factor BP: Flagged cell at ({r},{c}) with belief {data['belief']:.2f}")
    logger.info("Factor Graph Belief Propagation Solver completed.")
    return changed

"monte carlo tree search"

def mcts_move(board, adj, logger, simulations=50):
    candidates = get_coset_representatives(board, adj)
    if not candidates:
        candidates = [(r, c) for r in range(len(board)) for c in range(len(board[0]))
                      if board[r][c]['covered'] and not board[r][c]['flagged']]
    if not candidates:
        return False
    total_cells = sum(1 for row in board for cell in row if not cell['isMine'])
    best_move = None
    best_score = float('-inf')
    for move in candidates:
        score_total = 0.0
        for _ in range(simulations):
            sim_board = copy.deepcopy(board)
            mr, mc = move
            sim_board[mr][mc]['covered'] = False
            if sim_board[mr][mc]['isMine']:
                score_total -= 1.0
                continue
            if sim_board[mr][mc]['clue'] == 0:
                bfs_expand(sim_board, mr, mc, adj, logger)
            revealed_safe = sum(1 for row in sim_board for cell in row
                                if not cell['covered'] and not cell['isMine'])
            score_total += revealed_safe / max(total_cells, 1)
        score = score_total / simulations
        if score > best_score:
            best_score = score
            best_move = move
    if best_move and best_score > float('-inf'):
        br, bc = best_move
        board[br][bc]['covered'] = False
        logger.info(f"MCTS: Revealed cell at ({br},{bc}) with avg score {best_score:.3f}")
        if board[br][bc]['clue'] == 0:
            bfs_expand(board, br, bc, adj, logger)
        return True
    return False


class MineProbabilityCalculator:
    def __init__(self, total_mines, board_size):
        self.total_mines = total_mines
        self.board_size = board_size  # (rows, cols)
        self.pattern_weights = { 
            "1_1_shared_mine": 0.95,
            "1_2_1": 0.88,
            "edge_2": 0.85,
            "corner_3": 0.83,
            "2_2_parallel": 0.80,
            "1_2_diagonal": 0.78,
            "t_shape_3": 0.76,
            "1_1_1_triangle": 0.75,
            "3_2_3_line": 0.72,
            "4_center_cross": 0.70,
            "edge_pattern": 0.68,
            "1_2_2_1": 0.67,
            "l_shape_1_2": 0.76,
            "split_1_2": 0.72,
            "checkerboard_1s": 0.74,
            "5_center_star": 0.70,
            "1_1_1_1_square": 0.72,
            "1_2_Split": 0.71,
            "2_3_2_wall": 0.70,
            "mirror_2s": 0.72,
            "5_edge": 0.65,
            "island_1": 0.60,
            "double_l_shape": 0.66,
            "3_4_3_line": 0.58,
            "snake_1_2_1_2": 0.62,
            "4_corner": 0.55,
            "false_edge_5": 0.50,
            "diagonal_2_bridge": 0.52,
            "1_gap_1": 0.65,
            "mine_less_cross": 0.66,
            "2_2_2_line": 0.60,
            "corner_1_2_trap": 0.48,
            "double_bluff_cluster": 0.55,
            "spiral_progression": 0.58,
            "50_50_forcer": 0.49,
            "triple_3_cluster": 0.52,
            "edge_1_2_gap": 0.50,
            "3_corner_trap": 0.45,
            "false_2_cluster": 0.54,
            "double_3_mirror": 0.53,
            "1_1_1_line": 0.68,
            "asymmetric_cluster": 0.60,
            "spiral_pattern": 0.58,
            "checkerboard_expansion": 0.62,
            "cross_overlap": 0.64,
            "corner_trap": 0.43,
            "diagonal_chain": 0.56,
            "nested_corner": 0.52,
            "cross_high_clues": 0.66,
            "double_diamond": 0.61,
            "extended_2_3_2_line": 0.59,
            "4_2_4_Sandwich": 0.48,
            "2_5_2_trap": 0.47,
            "floating_4_edge": 0.50,
            "edge_1_3_1_edge": 0.49,
            "3_1_parallel": 0.53,
            "mirror_3s_diagonal": 0.51,
            "false_3_cluster": 0.43,
            "mine_pocket_1_2_2_1": 0.58,
            "rotational_2_quad": 0.54,
            "axial_1_3_mirror": 0.56,
            "twin_tunnels": 0.52,
            "false_2_choice": 0.45,
            "triangular_2_trap": 0.52,
            "prog_1_wave": 0.57,
            "spiderweb_3": 0.60,
            "corner_2_1_2_L": 0.46,
            "decoy_2_line": 0.49,
            "1_4_1_v": 0.48,
            "pseudo_1_trap": 0.43,
            "diagonal_33_bridge": 0.44,
            "split_4_deception": 0.42,
            "triple_2_cluster": 0.45,
            "edge_locked_5": 0.47,
            "131_sandwich": 0.50,
            "cross_reducer": 0.58,
            "false_edge_trap": 0.46,
            "mine_less_cross_verification": 0.68,
            "double_bluff": 0.49,
            "spiral_minefield": 0.54,
            "1_gap_2_ambiguity": 0.52,
            "corner_4_verification": 0.56,
            "3_way_intersection": 0.51,
            "progressive_wave": 0.48
        }
        self.prob_cache = {}
        
    def calculate_probabilities(self, board, matched_patterns):
        rows, cols = self.board_size
        covered = sum(1 for r in range(rows) for c in range(cols) if board[r][c]['covered'])
        base_prob = self.total_mines / covered if covered > 0 else 0
        prob_grid = defaultdict(lambda: base_prob)
        self._apply_pattern_probs(prob_grid, matched_patterns, covered)
        self._apply_spatial_adjustments(prob_grid, board)
        return prob_grid

    def _apply_pattern_probs(self, prob_grid, matched_patterns, covered_cells):
        pattern_boost = 1.2
        conflict_penalty = 0.7
        for pattern_name, cells in matched_patterns:
            base_weight = self.pattern_weights.get(pattern_name, 0.5)
            for cell in cells:
                current_prob = prob_grid[cell]
                pattern_prob = self._get_pattern_base_prob(pattern_name, cells)
                weighted_prob = (current_prob * (1 - base_weight) + pattern_prob * base_weight)
                if self._has_conflicting_patterns(cell, matched_patterns):
                    prob_grid[cell] = weighted_prob * conflict_penalty
                else:
                    prob_grid[cell] = weighted_prob * pattern_boost
                prob_grid[cell] = max(0.01, min(0.99, prob_grid[cell]))
                
    def _get_pattern_base_prob(self, pattern_name, pattern_cells):
        known_probs = { 
            "1_1_shared_mine": 0.95,
            "1_2_1": 0.88,
            "edge_2": 0.85,
            "corner_3": 0.83,
            "2_2_parallel": 0.80,
            "1_2_diagonal": 0.78,
            "t_shape_3": 0.76,
            "1_1_1_triangle": 0.75,
            "3_2_3_line": 0.72,
            "4_center_cross": 0.70,
            "edge_pattern": 0.68,
            "1_2_2_1": 0.67,
            "l_shape_1_2": 0.76,
            "split_1_2": 0.72,
            "checkerboard_1s": 0.74,
            "5_center_star": 0.70,
            "1_1_1_1_square": 0.72,
            "1_2_Split": 0.71,
            "2_3_2_wall": 0.70,
            "mirror_2s": 0.72,
            "5_edge": 0.65,
            "island_1": 0.60,
            "double_l_shape": 0.66,
            "3_4_3_line": 0.58,
            "snake_1_2_1_2": 0.62,
            "4_corner": 0.55,
            "false_edge_5": 0.50,
            "diagonal_2_bridge": 0.52,
            "1_gap_1": 0.65,
            "mine_less_cross": 0.66,
            "2_2_2_line": 0.60,
            "corner_1_2_trap": 0.48,
            "double_bluff_cluster": 0.55,
            "spiral_progression": 0.58,
            "50_50_forcer": 0.49,
            "triple_3_cluster": 0.52,
            "edge_1_2_gap": 0.50,
            "3_corner_trap": 0.45,
            "false_2_cluster": 0.54,
            "double_3_mirror": 0.53,
            "1_1_1_line": 0.68,
            "asymmetric_cluster": 0.60,
            "spiral_pattern": 0.58,
            "checkerboard_expansion": 0.62,
            "cross_overlap": 0.64,
            "corner_trap": 0.43,
            "diagonal_chain": 0.56,
            "nested_corner": 0.52,
            "cross_high_clues": 0.66,
            "double_diamond": 0.61,
            "extended_2_3_2_line": 0.59,
            "4_2_4_Sandwich": 0.48,
            "2_5_2_trap": 0.47,
            "floating_4_edge": 0.50,
            "edge_1_3_1_edge": 0.49,
            "3_1_parallel": 0.53,
            "mirror_3s_diagonal": 0.51,
            "false_3_cluster": 0.43,
            "mine_pocket_1_2_2_1": 0.58,
            "rotational_2_quad": 0.54,
            "axial_1_3_mirror": 0.56,
            "twin_tunnels": 0.52,
            "false_2_choice": 0.45,
            "triangular_2_trap": 0.52,
            "prog_1_wave": 0.57,
            "spiderweb_3": 0.60,
            "corner_2_1_2_L": 0.46,
            "decoy_2_line": 0.49,
            "1_4_1_v": 0.48,
            "pseudo_1_trap": 0.43,
            "diagonal_33_bridge": 0.44,
            "split_4_deception": 0.42,
            "triple_2_cluster": 0.45,
            "edge_locked_5": 0.47,
            "131_sandwich": 0.50,
            "cross_reducer": 0.58,
            "false_edge_trap": 0.46,
            "mine_less_cross_verification": 0.68,
            "double_bluff": 0.49,
            "spiral_minefield": 0.54,
            "1_gap_2_ambiguity": 0.52,
            "corner_4_verification": 0.56,
            "3_way_intersection": 0.51,
            "progressive_wave": 0.48
        }
        if pattern_name in known_probs:
            return known_probs[pattern_name]
        mine_count = sum(1 for cell in pattern_cells if cell['isMine'])
        return mine_count / len(pattern_cells)

    def _apply_spatial_adjustments(self, prob_grid, board):
        for (r, c), cell_prob in prob_grid.items():
            if not board[r][c]['covered']:
                continue
            neighbors = self._get_neighbors(r, c)
            revealed_neighbors = [n for n in neighbors if not board[n[0]][n[1]]['covered']]
            if revealed_neighbors:
                total_clues = sum(board[nr][nc]['clue'] for nr, nc in revealed_neighbors)
                flagged = sum(1 for nr, nc in neighbors if board[nr][nc]['flagged'])
                remaining_mines = total_clues - flagged
                if remaining_mines > 0:
                    adjustment = remaining_mines / len(neighbors)
                    prob_grid[(r, c)] = max(prob_grid[(r, c)], adjustment)

    def _get_neighbors(self, r, c):
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.board_size[0] and 0 <= nc < self.board_size[1]:
                    neighbors.append((nr, nc))
        return neighbors

    def _has_conflicting_patterns(self, cell, matched_patterns):
        patterns = [p[0] for p in matched_patterns if cell in p[1]]
        safe_patterns = {"mine_less_cross_verification", "progressive_wave"}
        danger_patterns = {"triple_3_cluster", "4_2_4_sandwich"}
        return len(safe_patterns.intersection(patterns)) > 0 and len(danger_patterns.intersection(patterns)) > 0

    def burnside_baseline(self, board):
        n_rows, n_cols = self.board_size
        total_configs = burnside_mine_count(n_rows, n_cols, self.total_mines)
        calculation_logger.info(f"Burnside baseline: {total_configs} computed.")
        return total_configs

class ProbabilisticSolver:
    def __init__(self, total_mines, board_size):
        self.prob_calculator = MineProbabilityCalculator(total_mines, board_size)
        self.pattern_cache = {}
        
    def make_decision(self, board, adj):
        board_graph = self._board_to_graph(board, adj)
        matched_patterns = self._find_pattern_matches(board_graph)
        prob_grid = self.prob_calculator.calculate_probabilities(board, matched_patterns)
        safest = min(prob_grid, key=prob_grid.get)
        riskiest = max(prob_grid, key=prob_grid.get)
        return {
            'reveal': safest,
            'flag': riskiest if prob_grid[riskiest] > 0.85 else None,
            'probabilities': prob_grid
        }

    def _find_pattern_matches(self, board_graph):
        matched = []
        for pattern_name, pattern_info in PATTERNS.items():
            gm = nx.isomorphism.GraphMatcher(
                board_graph, pattern_info['graph'] if isinstance(pattern_info, dict) else pattern_info,
                node_match=lambda n1, n2: (
                    n1.get('clue', -1) == n2.get('clue', -2) and
                    n1['covered'] == n2['covered'] and
                    n1['flagged'] == n2['flagged']
                )
            )
            if gm.subgraph_is_isomorphic():
                matched.append((pattern_name, [n for n in (pattern_info['graph'].nodes() if isinstance(pattern_info, dict) else pattern_info.nodes()) if board_graph.nodes[n]['covered']]))
        return matched

    def _board_to_graph(self, board, adj):
        G = nx.Graph()
        for r in range(len(board)):
            for c in range(len(board[0])):
                props = {
                    'clue': board[r][c]['clue'],
                    'covered': board[r][c]['covered'],
                    'flagged': board[r][c]['flagged'],
                    'isMine': board[r][c]['isMine']
                }
                G.add_node((r, c), **props)
                for nr, nc in adj[r][c]:
                    G.add_edge((r, c), (nr, nc))
        G.graph["n_rows"] = len(board)
        G.graph["n_cols"] = len(board[0]) if board else 0
        return G

"INTEGRATION"

def generate_random_board(n_rows, n_cols, num_mines, first_move=None):
    cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    if first_move:
        dirs = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
        exset = set([first_move])
        fr, fc = first_move
        for dr, dc in dirs:
            rr, cc = fr + dr, fc + dc
            if 0 <= rr < n_rows and 0 <= cc < n_cols:
                exset.add((rr, cc))
        available = list(set(cells) - exset)
    else:
        available = cells[:]
    if num_mines > len(available):
        raise ValueError("Too many mines for the available cells.")
    mines = set(random.sample(available, num_mines))
    grid = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            row.append('*' if (r, c) in mines else '.')
        grid.append(row)
    return grid

def compute_clues(grid):
    dirs = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    n_rows, n_cols = len(grid), len(grid[0])
    for r in range(n_rows):
        for c in range(n_cols):
            if grid[r][c] == '*':
                continue
            count = 0
            for dr, dc in dirs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < n_rows and 0 <= cc < n_cols and grid[rr][cc] == '*':
                    count += 1
            grid[r][c] = str(count)

def create_board(grid):
    board = []
    for row in grid:
        rowdata = []
        for ch in row:
            if ch == '*':
                rowdata.append({'isMine': True, 'clue': -1, 'covered': True, 'flagged': False})
            else:
                rowdata.append({'isMine': False, 'clue': int(ch), 'covered': True, 'flagged': False})
        board.append(rowdata)
    return board

def build_adjacency(board):
    n_rows, n_cols = len(board), len(board[0])
    dirs = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    adj = []
    for r in range(n_rows):
        row = []
        for c in range(n_cols):
            neighbors = []
            for dr, dc in dirs:
                rr, cc = r + dr, c + dc
                if 0 <= rr < n_rows and 0 <= cc < n_cols:
                    neighbors.append((rr, cc))
            row.append(neighbors)
        adj.append(row)
    return adj


def build_state(board, n_rows, n_cols, num_mines):
    """Canonical RL state. Used by both the training env and GUI inference so the
    network sees the same encoding at train and test time.

    Reveal-only: there is no flag action, so no flag plane is needed.

    Returns (spatial, scalars):
      spatial[0]: covered mask  (1 if cell is covered, 0 otherwise)
      spatial[1]: clue / 8      (0..1 if revealed, 0 if covered)
      scalars: [covered_ratio, mine_density]  (game-progress + difficulty cues
        for the dueling value head)
    """
    spatial = np.zeros((2, n_rows, n_cols), dtype=np.float32)
    covered_count = 0
    for r in range(n_rows):
        for c in range(n_cols):
            cell = board[r][c]
            if cell['covered']:
                spatial[0, r, c] = 1.0
                covered_count += 1
            else:
                spatial[1, r, c] = cell['clue'] / 8.0
    cells = max(n_rows * n_cols, 1)
    scalars = np.array([covered_count / cells, num_mines / cells], dtype=np.float32)
    return spatial, scalars


"UI DESIGN"

class DifficultyDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Difficulty")
        layout = QtWidgets.QVBoxLayout(self)
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems(["Beginner", "Intermediate", "Expert", "Extreme"])
        layout.addWidget(self.combo)
        btn_ok = QtWidgets.QPushButton("OK")
        btn_ok.clicked.connect(self.accept)
        layout.addWidget(btn_ok)
        self.selected_difficulty = None
    def accept(self):
        self.selected_difficulty = self.combo.currentText().lower()
        super().accept()

class CellButton(QtWidgets.QPushButton):
    def __init__(self, row, col, parent=None):
        super().__init__(parent)
        self.row = row
        self.col = col
        self.setFixedSize(35,35)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setStyleSheet("""
            QPushButton {
                background-color: #BDBDBD;
                border: 1px solid #999;
                font-weight: bold;
            }
            QPushButton:disabled {
                background-color: white;
                border: 1px solid #CCC;
            }
        """)
        self.setFont(QtGui.QFont("Arial", 14))

def get_difficulty_params(difficulty):
    params = {
        'beginner': (8,8,10),
        'intermediate': (16,16,40),
        'expert': (16,30,99),
        'extreme': (25,50,375) #EXTRA DIFF FOR STABILITY TEST will be removed for the final submission code
    }
    return params.get(difficulty, (8,8,10))

"AGENT DESIGN"

class MinesweeperEnv:
    """Reveal-only Minesweeper environment with guaranteed-safe first click.

    Action space is `n_rows * n_cols` reveal indices. There is no flag action - the
    win condition `all(covered == isMine)` only requires every safe cell to be
    revealed, and including flag actions encouraged a degenerate "flag everything"
    fixed point during DQN training. The GUI's auto-flag-on-win helper covers the
    cosmetic side.

    Reward (bounded):
      step penalty           -0.05
      reveal safe (per cell)  +1.0
      cascade per BFS cell    +0.2
      hit a mine (terminal)  -10.0
      win (terminal)         +20.0
    """

    def __init__(self, difficulty):
        self.difficulty = difficulty
        self.n_rows, self.n_cols, self.num_mines = get_difficulty_params(difficulty)
        self.first_reveal_done = False
        self.reset()

    def reset(self):
        # Lay down a placeholder board; it gets regenerated around the agent's
        # first reveal so the opening click is always safe.
        grid = generate_random_board(self.n_rows, self.n_cols, self.num_mines)
        compute_clues(grid)
        self.board = create_board(grid)
        self.adj = build_adjacency(self.board)
        self.done = False
        self.step_count = 0
        self.first_reveal_done = False
        return self._get_state()

    def _get_state(self):
        return build_state(self.board, self.n_rows, self.n_cols, self.num_mines)

    def _ensure_safe_first_reveal(self, r, c):
        grid = generate_random_board(self.n_rows, self.n_cols, self.num_mines, first_move=(r, c))
        compute_clues(grid)
        self.board = create_board(grid)
        self.adj = build_adjacency(self.board)
        self.first_reveal_done = True

    def step(self, action):
        r = action // self.n_cols
        c = action % self.n_cols
        cell = self.board[r][c]

        # Safety net: action masking should prevent picking already-revealed cells.
        if not cell['covered']:
            self.step_count += 1
            return self._get_state(), -0.5, self.done, {'reason': 'invalid_move'}

        if not self.first_reveal_done:
            self._ensure_safe_first_reveal(r, c)
            cell = self.board[r][c]

        self.step_count += 1
        info = {}
        reward = -0.05  # small step penalty to nudge toward shorter games

        cell['covered'] = False
        if cell['isMine']:
            self.done = True
            reward += -10.0
            info['reason'] = 'hit_mine'
        else:
            reward += 1.0
            if cell['clue'] == 0:
                cascade = bfs_expand(self.board, r, c, self.adj, calculation_logger)
                reward += 0.2 * cascade

        if self.check_win():
            self.done = True
            reward += 20.0
            info['reason'] = 'win'

        return self._get_state(), reward, self.done, info

    def check_win(self):
        return all(cell['covered'] == cell['isMine'] for row in self.board for cell in row)

class DQN(nn.Module):
    """Dueling DQN. Reveal-only action space: one Q value per cell.

    Inputs:
        spatial: (B, 2, H, W) - covered mask, clue plane
        scalars: (B, num_scalars) - covered_ratio, mine_density

    Output:
        Q values of shape (B, H*W), row-major to match the env's action encoding.
    """

    def __init__(self, height, width, num_actions, num_scalars=2):
        super().__init__()
        assert num_actions == height * width, (
            f"num_actions ({num_actions}) must equal height * width ({height * width})")
        self.height = height
        self.width = width
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # Dueling: advantage head - one Q per cell
        self.advantage_conv = nn.Conv2d(64, 1, kernel_size=1)
        # Dueling: value head from globally-pooled features + scalars
        self.value_fc1 = nn.Linear(64 + num_scalars, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, spatial, scalars):
        x = F.relu(self.conv1(spatial))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        adv = self.advantage_conv(x)              # (B, 1, H, W)
        adv = adv.flatten(1)                      # (B, H*W)
        adv = adv - adv.mean(dim=1, keepdim=True) # zero-mean (dueling)

        v = x.mean(dim=[2, 3])                    # (B, 64) global avg pool
        v = torch.cat([v, scalars], dim=1)
        v = F.relu(self.value_fc1(v))
        v = self.value_fc2(v)                     # (B, 1)

        return v + adv

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.7):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.alpha = alpha
    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size, beta=1.0):
        if len(self.buffer) == 0:
            return [], [], []
        prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = float(priority)
    def __len__(self):
        return len(self.buffer)

def evaluate_model(model, env, eval_episodes, device, eval_epsilon=0.01):
    """Run greedy (with small ε for tie-breaking) eval. Returns (avg_reward, win_rate)."""
    if eval_episodes <= 0:
        return 0.0, 0.0
    model.eval()
    total_reward = 0.0
    win_count = 0
    max_steps = max(2 * env.n_rows * env.n_cols, 200)
    try:
        for episode in range(eval_episodes):
            spatial, scalars = env.reset()
            done = False
            episode_reward = 0.0
            episode_won = False
            for _ in range(max_steps):
                if done:
                    break
                valid_actions = []
                for i in range(env.n_rows * env.n_cols):
                    r, c = i // env.n_cols, i % env.n_cols
                    if env.board[r][c]['covered']:
                        valid_actions.append(i)
                if not valid_actions:
                    break
                if random.random() < eval_epsilon:
                    action = random.choice(valid_actions)
                else:
                    spatial_t = torch.from_numpy(spatial).unsqueeze(0).to(device)
                    scalars_t = torch.from_numpy(scalars).unsqueeze(0).to(device)
                    with torch.no_grad():
                        q_values = model(spatial_t, scalars_t)
                    q_valid = q_values[0, valid_actions]
                    action = valid_actions[q_valid.argmax().item()]
                next_state, reward, done, info = env.step(action)
                spatial, scalars = next_state
                episode_reward += reward
                if info.get('reason') == 'win':
                    episode_won = True
            total_reward += episode_reward
            if episode_won:
                win_count += 1
        avg_reward = total_reward / eval_episodes
        win_rate = win_count / eval_episodes
        print(f"Eval over {eval_episodes} episodes: avg_reward={avg_reward:.2f}, "
              f"win_rate={win_rate:.2%} ({win_count}/{eval_episodes})")
    finally:
        model.train()
    return avg_reward, win_rate

def train_in_batches(difficulty="beginner",
                     episodes_per_batch=1000,
                     num_batches=2,
                     eval_episodes=200,
                     batch_size=512,
                     gamma=0.99,
                     learning_rate=1e-4,
                     epsilon_start=1.0,
                     epsilon_end=0.1,
                     epsilon_decay_steps=80_000,
                     learning_starts=1_000,
                     target_sync_steps=10_000,
                     beta_start=0.4,
                     beta_end=0.5,
                     replay_capacity=100_000,
                     early_stop_patience=3):
    """Double + Dueling DQN with prioritized replay, Huber loss, env-step-paced targets/eps.

    Reveal-only action space (`num_actions = n_rows * n_cols`). Including flag actions
    creates a degenerate "flag everything" attractor for risk-averse Q functions.

    Returns (best_model_state, all_batch_rewards, all_batch_epsilons,
             all_batch_eval_scores, all_batch_wins, all_batch_eval_win_rates).
    """
    device = pick_device()
    print(f"\nTraining on: {device}\n")
    empty_device_cache(device)
    env = MinesweeperEnv(difficulty)
    n_rows, n_cols, _ = get_difficulty_params(difficulty)
    num_actions = n_rows * n_cols
    policy_net = DQN(n_rows, n_cols, num_actions).to(device)
    target_net = DQN(n_rows, n_cols, num_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate, weight_decay=1e-5)
    replay_buffer = PrioritizedReplayBuffer(capacity=replay_capacity, alpha=0.6)

    best_eval_win_rate = -1.0
    best_model_state = None
    bad_batches = 0
    env_steps = 0
    epsilon = epsilon_start
    all_batch_rewards = []
    all_batch_epsilons = []
    all_batch_wins = []
    all_batch_eval_scores = []
    all_batch_eval_win_rates = []
    max_steps = max(2 * n_rows * n_cols, 200)

    for batch_idx in range(1, num_batches + 1):
        batch_rewards = []
        batch_epsilons = []
        batch_wins = []
        print(f"\n=== Batch {batch_idx}/{num_batches} ===")
        print(f"env_steps so far: {env_steps} | initial epsilon: {epsilon:.4f}")

        for episode in range(episodes_per_batch):
            spatial, scalars = env.reset()
            done = False
            total_reward = 0.0
            won = False

            for _ in range(max_steps):
                if done:
                    break
                env_steps += 1

                # Linear eps decay over env steps
                progress = min(1.0, env_steps / max(1, epsilon_decay_steps))
                epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress

                valid_actions = []
                for i in range(n_rows * n_cols):
                    r, c = i // n_cols, i % n_cols
                    if env.board[r][c]['covered']:
                        valid_actions.append(i)
                if not valid_actions:
                    break

                if env_steps < learning_starts or random.random() < epsilon:
                    action = random.choice(valid_actions)
                else:
                    spatial_t = torch.from_numpy(spatial).unsqueeze(0).to(device)
                    scalars_t = torch.from_numpy(scalars).unsqueeze(0).to(device)
                    with torch.no_grad():
                        q_values = policy_net(spatial_t, scalars_t)
                    q_valid = q_values[0, valid_actions]
                    action = valid_actions[q_valid.argmax().item()]

                next_state, reward, done, info = env.step(action)
                next_spatial, next_scalars = next_state
                total_reward += reward
                if info.get('reason') == 'win':
                    won = True
                replay_buffer.push((spatial, scalars), action, reward,
                                   (next_spatial, next_scalars), done)
                spatial, scalars = next_spatial, next_scalars

                # Learning step (after warmup)
                if env_steps >= learning_starts and len(replay_buffer) >= batch_size:
                    beta = beta_start + (beta_end - beta_start) * progress
                    samples, indices, weights = replay_buffer.sample(batch_size, beta=beta)
                    states_b, actions_b, rewards_b, next_states_b, dones_b = zip(*samples)
                    sp_b = np.stack([s[0] for s in states_b])
                    sc_b = np.stack([s[1] for s in states_b])
                    nsp_b = np.stack([s[0] for s in next_states_b])
                    nsc_b = np.stack([s[1] for s in next_states_b])

                    sp_t = torch.from_numpy(sp_b).to(device)
                    sc_t = torch.from_numpy(sc_b).to(device)
                    nsp_t = torch.from_numpy(nsp_b).to(device)
                    nsc_t = torch.from_numpy(nsc_b).to(device)
                    actions_t = torch.tensor(actions_b, dtype=torch.long, device=device)
                    rewards_t = torch.tensor(rewards_b, dtype=torch.float32, device=device)
                    dones_t = torch.tensor(dones_b, dtype=torch.float32, device=device)
                    weights_t = torch.tensor(weights, dtype=torch.float32, device=device)

                    q_pred = policy_net(sp_t, sc_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        # Double DQN: select with policy_net, evaluate with target_net
                        next_actions = policy_net(nsp_t, nsc_t).argmax(1, keepdim=True)
                        next_q = target_net(nsp_t, nsc_t).gather(1, next_actions).squeeze(1)
                        td_target = rewards_t + gamma * next_q * (1.0 - dones_t)
                    td_errors = q_pred - td_target
                    huber = F.smooth_l1_loss(q_pred, td_target, reduction='none')
                    loss = (weights_t * huber).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
                    optimizer.step()

                    new_priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
                    replay_buffer.update_priorities(indices, new_priorities)

                # Periodic target sync, paced by env steps
                if env_steps % target_sync_steps == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            batch_rewards.append(total_reward)
            batch_epsilons.append(epsilon)
            batch_wins.append(1 if won else 0)
            if (episode + 1) % 100 == 0:
                recent_r = batch_rewards[-100:]
                recent_w = batch_wins[-100:]
                print(f"[Batch {batch_idx}] Ep {episode+1}/{episodes_per_batch} | "
                      f"avg_reward={np.mean(recent_r):.2f} | "
                      f"win_rate={np.mean(recent_w):.2%} | "
                      f"epsilon={epsilon:.3f} | "
                      f"env_steps={env_steps}")

        all_batch_rewards.append(batch_rewards)
        all_batch_epsilons.append(batch_epsilons)
        all_batch_wins.append(batch_wins)
        print(f"Batch {batch_idx} completed. Starting evaluation...")

        try:
            avg_reward, win_rate = evaluate_model(policy_net, env, eval_episodes, device)
            all_batch_eval_scores.append(avg_reward)
            all_batch_eval_win_rates.append(win_rate)
        except Exception as e:
            print(f"Evaluation failed in Batch {batch_idx}: {str(e)}")
            raise

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = f"model_batch{batch_idx}_{timestamp}.pt"
            torch.save(policy_net.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            if win_rate > best_eval_win_rate:
                best_eval_win_rate = win_rate
                best_model_state = {k: v.detach().cpu().clone() for k, v in policy_net.state_dict().items()}
                best_path = f"best_model_batch{batch_idx}_{timestamp}.pt"
                torch.save(best_model_state, best_path)
                print(f"New best (win_rate={win_rate:.2%}): {best_path}")
                bad_batches = 0
            else:
                bad_batches += 1
                print(f"No improvement (best={best_eval_win_rate:.2%}, this={win_rate:.2%}), "
                      f"strike {bad_batches}/{early_stop_patience}.")
        except Exception as e:
            print(f"Checkpoint saving failed in Batch {batch_idx}: {str(e)}")
            raise

        if bad_batches >= early_stop_patience:
            print(f"Early stop: eval win_rate did not beat best={best_eval_win_rate:.2%} "
                  f"for {early_stop_patience} batches. Restoring best weights.")
            if best_model_state is not None:
                policy_net.load_state_dict(best_model_state)
            break

        print(f"Batch {batch_idx} fully processed. Moving to next batch...")

    print(f"\nTraining Complete! Best eval win rate: {best_eval_win_rate:.2%}")
    return (best_model_state, all_batch_rewards, all_batch_epsilons,
            all_batch_eval_scores, all_batch_wins, all_batch_eval_win_rates)


def plot_batch_comparisons(all_batch_rewards, all_batch_epsilons,
                           all_batch_eval_scores, all_batch_wins,
                           all_batch_eval_win_rates, window_size=500):
    """Three-panel training plot: epsilon, reward, win rate. Uses recorded win flags
    instead of inferring wins from reward thresholds, which broke under reward shaping."""
    all_batch_rewards = [np.asarray(b) for b in all_batch_rewards]
    all_batch_epsilons = [np.asarray(b) for b in all_batch_epsilons]
    all_batch_wins = [np.asarray(b) for b in all_batch_wins]
    all_batch_eval_scores = np.asarray(all_batch_eval_scores)
    all_batch_eval_win_rates = np.asarray(all_batch_eval_win_rates)

    num_batches = len(all_batch_rewards)
    if num_batches == 0:
        print("plot_batch_comparisons: no data.")
        return
    episodes_per_batch = len(all_batch_rewards[0])
    total_episodes = num_batches * episodes_per_batch

    all_rewards = np.concatenate(all_batch_rewards)
    all_epsilons = np.concatenate(all_batch_epsilons)
    all_wins = np.concatenate(all_batch_wins).astype(np.float32)
    episodes_total = np.arange(total_episodes)

    win_rate_rolling = np.zeros(total_episodes)
    reward_rolling = np.zeros(total_episodes)
    half = window_size // 2
    for i in range(total_episodes):
        lo = max(0, i - half)
        hi = min(total_episodes, i + half + 1)
        win_rate_rolling[i] = float(np.mean(all_wins[lo:hi]))
        reward_rolling[i] = float(np.mean(all_rewards[lo:hi]))

    batch_ends = np.array([i * episodes_per_batch - 1 for i in range(1, num_batches + 1)])

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(episodes_total, all_epsilons, color='blue')
    axes[0].set_ylabel(r'$\epsilon$')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 1.05)

    axes[1].plot(episodes_total, all_rewards, color='lightgray', alpha=0.5, label='Episode reward')
    axes[1].plot(episodes_total, reward_rolling, color='darkgreen', linewidth=2,
                 label=f'Rolling avg (w={window_size})')
    axes[1].scatter(batch_ends, all_batch_eval_scores, color='magenta',
                    marker='*', s=120, zorder=5, label='Eval avg reward')
    axes[1].set_ylabel('Reward')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best')

    axes[2].plot(episodes_total, win_rate_rolling, color='purple', linewidth=2,
                 label=f'Rolling win rate (w={window_size})')
    axes[2].scatter(batch_ends, all_batch_eval_win_rates, color='red',
                    marker='*', s=120, zorder=5, label='Eval win rate')
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Win rate')
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='best')

    plt.suptitle('Training Progress')
    plt.tight_layout()
    plt.savefig('minesweeper_training_plot.png', dpi=200, bbox_inches='tight')
    print("Plot saved as 'minesweeper_training_plot.png'")
    plt.show()


class MinesweeperWindow(QtWidgets.QMainWindow):
    
    def __init__(self, difficulty, mode="solver"):
        super().__init__()
        self.difficulty = difficulty
        self.mode = mode  # "solver" or "trained"
        self.n_rows, self.n_cols, self.num_mines = get_difficulty_params(difficulty)
        self.init_game()
        self.ai_timer = QtCore.QTimer()
        self.ai_timer.timeout.connect(self.ai_step)
        self.proof_metrics = []
        self.probabilities = {}
        self.fail_count = 0
        self.solved = False
        self.first_move_made = False
        
        if self.mode == "trained":
            device = pick_device()
            self.trained_net = DQN(self.n_rows, self.n_cols, self.n_rows * self.n_cols).to(device)
            self.device = device
            # Track whether load_trained_model() succeeded; until then `trained_net`
            # is just random-init and would play randomly.
            self.model_loaded = False
            self.loaded_model_path = None

        self.init_ui()
        self.resize(min(self.n_cols * 40 + 50, 1280), min(self.n_rows * 40 + 150, 720))

    def init_game(self):
        grid = generate_random_board(self.n_rows, self.n_cols, self.num_mines)
        compute_clues(grid)
        self.board = create_board(grid)
        self.adj = build_adjacency(self.board)
        self.game_over = False
        self.first_move_made = False

    def init_ui(self):
        self.setWindowTitle(f"Minesweeper AI - {self.difficulty.capitalize()}")
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10,10,10,10)
        main_layout.setSpacing(10)
        
        control_panel = QtWidgets.QHBoxLayout()
        
        self.ai_button = QtWidgets.QPushButton("Start AI")
        self.ai_button.clicked.connect(self.toggle_ai)
        control_panel.addWidget(self.ai_button)
        reset_btn = QtWidgets.QPushButton("New Game")
        reset_btn.clicked.connect(lambda: self.reset_game(manual=True))
        control_panel.addWidget(reset_btn)
        
        if self.mode == "trained":
            load_model_btn = QtWidgets.QPushButton("Load Trained Model")
            load_model_btn.clicked.connect(self.load_trained_model)
            control_panel.addWidget(load_model_btn)
            self.model_status_label = QtWidgets.QLabel()
            control_panel.addWidget(self.model_status_label)
            self._refresh_model_status_label()

        self.ai_log_checkbox = QtWidgets.QCheckBox("Show AI Logs")
        self.ai_log_checkbox.stateChanged.connect(self.toggle_ai_logging)
        control_panel.addWidget(self.ai_log_checkbox)
        
        
        self.calculation_log_checkbox = QtWidgets.QCheckBox("Show Calculation Logs")
        self.calculation_log_checkbox.stateChanged.connect(self.toggle_calculation_logging)
        control_panel.addWidget(self.calculation_log_checkbox)
        
        
        control_panel.addStretch()
        main_layout.addLayout(control_panel)
        
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        grid_widget = QtWidgets.QWidget()
        scroll.setWidget(grid_widget)
        self.grid_layout = QtWidgets.QGridLayout(grid_widget)
        self.grid_layout.setSpacing(1)
        self.grid_layout.setContentsMargins(2,2,2,2)
        self.buttons = []
        for r in range(self.n_rows):
            row_buttons = []
            for c in range(self.n_cols):
                btn = CellButton(r, c, grid_widget)
                btn.clicked.connect(lambda _, rr=r, cc=c: self.reveal(rr, cc))
                row_buttons.append(btn)
                self.grid_layout.addWidget(btn, r, c)
            self.buttons.append(row_buttons)
        main_layout.addWidget(scroll)
        
        self.ai_log_window = LogWindow(ai_logger, "AI Logs")
        self.calculation_log_window = LogWindow(calculation_logger, "Calculation Logs")
        self.moves_made = 0
        self.flags_placed = 0
        self.time_elapsed = 0
        self.moves_label = QtWidgets.QLabel("Moves Made: 0")
        self.flags_label = QtWidgets.QLabel("Flags Placed: 0")
        self.mines_left_label = QtWidgets.QLabel(f"Mines Left: {self.num_mines}")
        self.time_label = QtWidgets.QLabel("Time Elapsed: 0s")
        control_panel.addWidget(self.moves_label)
        control_panel.addWidget(self.flags_label)
        control_panel.addWidget(self.mines_left_label)
        control_panel.addWidget(self.time_label)
        self.game_timer = QtCore.QTimer()
        self.game_timer.timeout.connect(self.update_time)
        self.game_timer.start(1000)
        self.update_ui()

    def toggle_ai_logging(self, state):
        if state == QtCore.Qt.Checked:
            self.ai_log_window.show()
            self.ai_log_window.start_logging()
        else:
            self.ai_log_window.stop_logging()
            self.ai_log_window.hide()

    def toggle_calculation_logging(self, state):
        if state == QtCore.Qt.Checked:
            self.calculation_log_window.show()
            self.calculation_log_window.start_logging()
        else:
            self.calculation_log_window.stop_logging()
            self.calculation_log_window.hide()

    def update_counters(self):
        self.flags_placed = sum(cell['flagged'] for row in self.board for cell in row)
        self.flags_label.setText(f"Flags Placed: {self.flags_placed}")
        mines_left = max(self.num_mines - self.flags_placed, 0)
        self.mines_left_label.setText(f"Mines Left: {mines_left}")

    def update_ui(self):
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                btn = self.buttons[r][c]
                cell = self.board[r][c]
                btn.setEnabled(not self.game_over and not cell['flagged'])
                btn.setDisabled(cell['flagged'] or not cell['covered'])
                if cell['flagged']:
                    btn.setText('🚩')
                    btn.setStyleSheet("background-color: #BDBDBD; color: black;")
                elif not cell['covered']:
                    if cell['isMine']:
                        btn.setText('💣')
                        btn.setStyleSheet("background-color: red; color: white;")
                    else:
                        text = str(cell['clue']) if cell['clue'] > 0 else ''
                        btn.setText(text)
                        btn.setStyleSheet("background-color: white; color: black;")
                else:
                    btn.setText('')
                    btn.setStyleSheet("background-color: #BDBDBD; color: black;")
                btn.repaint()
        self.update_counters()

    def reveal(self, r, c):
        if self.game_over or self.board[r][c]['flagged']:
            return
        self.board[r][c]['covered'] = False
        self.moves_made += 1
        self.moves_label.setText(f"Moves Made: {self.moves_made}")
        self.update_counters()
        is_mine = self.board[r][c]['isMine']
        clue = self.board[r][c]['clue']
        if (not is_mine) and clue == 0:
            bfs_expand(self.board, r, c, self.adj, ai_logger)
        if self.check_loss():
            self.game_over = True
            self.reveal_all()
            QtWidgets.QMessageBox.critical(self, "Game Over", "You hit a mine!")
        elif self.check_win():
            self.game_over = True
            self._mark_remaining_mines_as_flagged()
            QtWidgets.QMessageBox.information(self, "Victory", "All mines cleared!")
        self.update_ui()

    def check_loss(self):
        return any((not cell['covered'] and cell['isMine']) for row in self.board for cell in row)

    def check_win(self):
        return all((cell['covered'] == cell['isMine']) for row in self.board for cell in row)

    def reveal_all(self):
        for row in self.board:
            for cell in row:
                cell['covered'] = False
        self.update_ui()

    def _mark_remaining_mines_as_flagged(self):
        """On a win, auto-flag any mine the solver hadn't already flagged. Without this,
        a follow-up reveal_all() would un-cover those mines and update_ui would render
        them as 💣 on red - visually identical to an exploded mine, even though the
        player won by revealing every safe cell."""
        for row in self.board:
            for cell in row:
                if cell['isMine'] and not cell['flagged']:
                    cell['flagged'] = True

    def _refresh_model_status_label(self):
        if self.mode != "trained" or not hasattr(self, 'model_status_label'):
            return
        if self.model_loaded:
            name = os.path.basename(self.loaded_model_path or '')
            self.model_status_label.setText(f"Model: {name}")
            self.model_status_label.setStyleSheet("color: #2e7d32; font-weight: bold;")
            self.model_status_label.setToolTip(self.loaded_model_path or '')
        else:
            self.model_status_label.setText("Model: NOT LOADED (random)")
            self.model_status_label.setStyleSheet("color: #c62828; font-weight: bold;")
            self.model_status_label.setToolTip(
                "Click 'Load Trained Model' to use a .pt checkpoint. "
                "Until then the network is at random initialization and plays randomly."
            )

    def toggle_ai(self):
        if self.ai_timer.isActive():
            self.ai_timer.stop()
            self.ai_button.setText("Start AI")
        else:
            if self.mode == "trained":
                # Trained mode without weights would just play random. Warn before starting.
                if not getattr(self, 'model_loaded', False):
                    reply = QtWidgets.QMessageBox.warning(
                        self,
                        "No trained model loaded",
                        "The neural network has not been loaded from a checkpoint, "
                        "so it will play randomly. Start anyway?",
                        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                        QtWidgets.QMessageBox.No,
                    )
                    if reply != QtWidgets.QMessageBox.Yes:
                        return
                    ai_logger.warning("Starting AI in trained mode without a loaded model - playing randomly.")
                else:
                    ai_logger.info(f"Starting AI in trained mode with: {os.path.basename(self.loaded_model_path)}")
            else:
                ai_logger.info("Starting AI in solver mode (SAT/CP-SAT/pattern pipeline).")
            self.ai_timer.start(100)
            self.ai_button.setText("Stop AI")
        self.update_ui()

    def reset_game(self, manual=True):
        if manual:
            self.fail_count = 0
            self.solved = False
            self.ai_timer.stop()
            self.ai_button.setText("Start AI")
        self.init_game()
        self.game_over = False
        self.first_move_made = False
        self.probabilities = {}
        self.moves_made = 0
        self.time_elapsed = 0
        self.moves_label.setText("Moves Made: 0")
        self.flags_label.setText("Flags Placed: 0")
        self.mines_left_label.setText(f"Mines Left: {self.num_mines}")
        self.update_counters()
        self.update_ui()

    def update_time(self):
        if not self.game_over:
            self.time_elapsed += 1
            self.time_label.setText(f"Time Elapsed: {self.time_elapsed}s")

    def ai_step(self):
        if self.game_over:
            if self.check_win():
                self.solved = True
                self._mark_remaining_mines_as_flagged()
                QtWidgets.QMessageBox.information(self, "Victory", f"AI solved it! Attempts: {self.fail_count}")
                self.ai_timer.stop()
            return
        try:
            # Handle first move to ensure a safe start for bfs as the initial method of the solver
            if not self.first_move_made:
                r, c = random.randint(0, self.n_rows-1), random.randint(0, self.n_cols-1)
                self.first_move_made = True
                grid = generate_random_board(self.n_rows, self.n_cols, self.num_mines, first_move=(r, c))
                compute_clues(grid)
                self.board = create_board(grid)
                self.adj = build_adjacency(self.board)
                # The automorphism + spectral analyses below are informational only and
                # cost O(N^3) (eigendecomposition) / sympy Schreier-Sims; on Expert and
                # Extreme they cause multi-second hangs at game start. Run on small boards only.
                if self.n_rows * self.n_cols <= ANALYTICS_MAX_CELLS:
                    print_board_automorphism_info(self.board, self.adj)
                    symmetry_analysis(self.board, self.adj, calculation_logger)
                    board_graph = board_to_graph(self.board, self.adj)
                    spectral_analysis(board_graph, k=10)
                    spectral_clustering(board_graph, num_clusters=2)
                self.reveal(r, c)
                return
            else:
                #new changes for full solve
                if self.mode == "trained":
                    # Same canonical state the env produces; reveal-only action space.
                    spatial, scalars = build_state(self.board, self.n_rows, self.n_cols, self.num_mines)
                    spatial_t = torch.from_numpy(spatial).unsqueeze(0).to(self.device)
                    scalars_t = torch.from_numpy(scalars).unsqueeze(0).to(self.device)

                    valid_actions = []
                    for r in range(self.n_rows):
                        for c in range(self.n_cols):
                            if self.board[r][c]['covered'] and not self.board[r][c]['flagged']:
                                valid_actions.append(r * self.n_cols + c)

                    if not valid_actions:
                        if self.check_win():
                            ai_logger.info("Win condition met.")
                        else:
                            ai_logger.info(
                                "No valid actions and game not won. "
                                f"covered={sum(1 for row in self.board for cell in row if cell['covered'])}"
                            )
                        self.game_over = True
                        return

                    with torch.no_grad():
                        qvals = self.trained_net(spatial_t, scalars_t)
                    q_valid = qvals[0, valid_actions]
                    action = valid_actions[q_valid.argmax().item()]

                    r = action // self.n_cols
                    c = action % self.n_cols

                    if self.board[r][c]['covered'] and not self.board[r][c]['flagged']:
                        self.board[r][c]['covered'] = False
                        ai_logger.info(f"AI revealed cell at ({r}, {c})")
                        if self.board[r][c]['clue'] == 0:
                            bfs_expand(self.board, r, c, self.adj, ai_logger)
                        self.moves_made += 1
                        self.moves_label.setText(f"Moves Made: {self.moves_made}")
                        self.update_counters()
                else:
                    changed = False
                    remaining_mines = self.num_mines - sum(cell['flagged'] for row in self.board for cell in row)
                    changed |= sat_based_solve(self.board, self.adj, ai_logger)
                    if not changed:
                        changed |= smt_based_solve(self.board, self.adj, ai_logger)
                    if not changed:
                        changed |= pattern_based_solve(self.board, self.adj, ai_logger)
                    if not changed:
                        changed |= apply_constraint_logic(self.board, self.adj, ai_logger)
                    if not changed:
                        changed |= advanced_constraint_solving(self.board, self.adj, ai_logger)
                    if not changed:
                        changed |= cp_sat_solve(self.board, self.adj, ai_logger)
                    if not changed and remaining_mines <= 10:
                        changed |= endgame_solver(self.board, self.adj, remaining_mines, ai_logger)
                    if not changed:
                        bp_probs = belief_propagation_probabilities(self.board, self.adj, calculation_logger, self.num_mines, iterations=50)
                        if bp_probs:
                            best_cell = min(bp_probs, key=bp_probs.get)
                            br, bc = best_cell
                            if self.board[br][bc]['covered'] and not self.board[br][bc]['flagged']:
                                self.board[br][bc]['covered'] = False
                                if self.board[br][bc]['clue'] == 0:
                                    bfs_expand(self.board, br, bc, self.adj, ai_logger)
                                changed = True
                    if not changed:
                        changed |= factor_graph_belief_propagation_solver(self.board, self.adj, ai_logger)
                    if not changed:
                        changed |= mcts_move(self.board, self.adj, ai_logger, simulations=500)
                    if not changed:
                        if symmetry_reduced_moves(self.board, self.adj, ai_logger):
                            self.moves_made += 1
                            self.moves_label.setText(f"Moves Made: {self.moves_made}")
                            self.update_counters()
                            changed = True
                    if not changed:
                        cand = [(rr, cc) for rr in range(self.n_rows) for cc in range(self.n_cols)
                                if self.board[rr][cc]['covered'] and not self.board[rr][cc]['flagged']]
                        if cand:
                            rr, cc = random.choice(cand)
                            self.board[rr][cc]['covered'] = False
                            self.moves_made += 1
                            self.moves_label.setText(f"Moves Made: {self.moves_made}")
                            self.update_counters()
                            if self.board[rr][cc]['clue'] == 0:
                                bfs_expand(self.board, rr, cc, self.adj, ai_logger)
            # check game state
            if self.check_loss():
                self.fail_count += 1
                self.reset_game(manual=False)
                ai_logger.info(f"Game lost. Attempt {self.fail_count}. Retrying...")
                if not self.ai_timer.isActive():
                    self.ai_timer.start(100)
            elif self.check_win():
                self.game_over = True
                self.solved = True
                self._mark_remaining_mines_as_flagged()
                ai_logger.info(f"Game won! Fail attempts: {self.fail_count}")
                QtWidgets.QMessageBox.information(self, "Victory", f"AI solved! Fail attempts: {self.fail_count}")
                self.ai_timer.stop()
            else:
                ai_logger.info(f"Game state: {sum(1 for r in range(self.n_rows) for c in range(self.n_cols) if not self.board[r][c]['covered'])} safe cells revealed, {sum(1 for r in range(self.n_rows) for c in range(self.n_cols) if self.board[r][c]['flagged'])} cells flagged.")
            self.update_ui()
        except Exception as e:
            import traceback
            
            ai_logger.error(f"AI error: {str(e)}")
            ai_logger.error(traceback.format_exc())
            calculation_logger.error(f"AI error: {str(e)}")
            calculation_logger.error(traceback.format_exc())
            self.reset_game(manual=False)
            self.update_ui()
            
    def load_trained_model(self):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.ReadOnly
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
        self,
        "Select Trained Model File",
        "",
        "PyTorch Model Files (*.pt)",
        options=options)
        if file_name:
            try:
                state_dict = torch.load(file_name, map_location=self.device)
                self.trained_net.load_state_dict(state_dict)
                self.trained_net.eval()
                self.model_loaded = True
                self.loaded_model_path = file_name
                self._refresh_model_status_label()
                self.setWindowTitle(
                    f"Minesweeper AI - {self.difficulty.capitalize()} - "
                    f"Model: {os.path.basename(file_name)}"
                )
                ai_logger.info(f"Trained model loaded from {file_name}")
                QtWidgets.QMessageBox.information(
                    self,
                    "Model Loaded",
                    f"Successfully loaded model from:\n{file_name}"
                )
            except Exception as e:
                self.model_loaded = False
                self.loaded_model_path = None
                self._refresh_model_status_label()
                ai_logger.error(f"Failed to load trained model from {file_name}: {e}")
                QtWidgets.QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to load model:\n{str(e)}"
                )

"MAIN ENTRY TO CHANGE HYPERVALS FOR TRAINING, BE CAREFUL WITH BATCH_SIZE AS IT'S TIED TO THE GPU's MEMORY."

device = pick_device()
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        difficulty = "beginner"
        print(f"Training on difficulty: {difficulty}")
        result = train_in_batches(
            difficulty=difficulty,
            episodes_per_batch=10_000,
            num_batches=10,
            eval_episodes=200,
            batch_size=512,
            gamma=0.99,
            learning_rate=1e-4,        # was 5e-4, too aggressive - caused Q-value drift
            epsilon_start=1.0,
            epsilon_end=0.1,           # was 0.05 - keep some exploration so replay stays diverse
            epsilon_decay_steps=80_000,
            learning_starts=1_000,
            target_sync_steps=10_000,  # was 2_000 - slow target moves, breaks oscillation
            beta_start=0.4,
            beta_end=0.5,              # was 1.0 - keep PER's IS correction moderate
            replay_capacity=100_000,
            early_stop_patience=3,     # restore best weights if win_rate stagnates 3 batches
        )
        (best_model_state, all_batch_rewards, all_batch_eps,
         all_batch_evals, all_batch_wins, all_batch_eval_wins) = result
        plot_batch_comparisons(all_batch_rewards, all_batch_eps, all_batch_evals,
                               all_batch_wins, all_batch_eval_wins)
    else:
        mode = "trained" if (len(sys.argv) > 1 and sys.argv[1] == "trained") else "solver"
        def main_gui():
            app = QtWidgets.QApplication(sys.argv)
            diff_dialog = DifficultyDialog()
            if diff_dialog.exec_() == QtWidgets.QDialog.Accepted:
                difficulty = diff_dialog.selected_difficulty
                window = MinesweeperWindow(difficulty, mode=mode)
                window.show()
                sys.exit(app.exec_())
            else:
                app.quit()
        main_gui()
