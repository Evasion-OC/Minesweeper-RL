import networkx as nx


def create_1_1_pattern():
    G = nx.Graph()
    G.add_node((0, 0), clue=1, covered=False, flagged=False, isMine=False)
    G.add_node((0, 1), clue=1, covered=False, flagged=False, isMine=False)
    G.add_node((1, 0), clue=-1, covered=True, flagged=False, isMine=True)
    G.add_edges_from([((0,0), (0,1)), ((0,0), (1,0)), ((0,1), (1,0))])
    return G

def create_1_2_1_pattern():
    G = nx.Graph()
    G.add_node((0, 0), clue=1, covered=False)
    G.add_node((0, 1), clue=2, covered=False)
    G.add_node((0, 2), clue=1, covered=False)
    G.add_node((1, 1), covered=True, isMine=True)
    G.add_edges_from([
        ((0,0), (0,1)), ((0,1), (0,2)),
        ((1,1), (0,0)), ((1,1), (0,1)), ((1,1), (0,2))
    ])
    return G

def create_edge_2_pattern():
    G = nx.Graph()
    G.add_node((0, 0), clue=2, covered=False)
    G.add_node((1, 0), clue=3, covered=False)
    G.add_node((1, 1), covered=True, isMine=True)
    G.add_edges_from([((0,0), (1,0)), ((1,0), (1,1)), ((0,0), (1,1))])
    return G

def create_edge_pattern():
    G = nx.Graph()
    G.add_node((0, 0), clue=1, covered=False)
    G.add_node((0, 1), covered=True, isMine=True)
    G.add_node((1, 0), covered=True, isMine=False)
    G.add_edges_from([((0,0), (0,1)), ((0,0), (1,0))])
    return G

def create_2_3_2_wall():
    G = nx.Graph()
    G.add_nodes_from([
        ((0,1), {'clue':2, 'covered':False}),
        ((1,1), {'clue':3, 'covered':False}),
        ((2,1), {'clue':2, 'covered':False}),
        ((1,0), {'covered':True, 'isMine':True}),
        ((1,2), {'covered':True, 'isMine':True}),
        ((2,0), {'covered':True, 'isMine':True}),
        ((2,2), {'covered':True, 'isMine':True})
    ])
    G.add_edges_from([
        ((0,1),(1,0)), ((0,1),(1,1)), ((0,1),(1,2)),
        ((1,1),(1,0)), ((1,1),(1,2)), ((1,1),(2,0)),
        ((1,1),(2,1)), ((1,1),(2,2)),
        ((2,1),(2,0)), ((2,1),(2,2))
    ])
    return G

def create_false_edge_5():
    G = nx.Graph()
    G.add_nodes_from([
        ((0,1), {'clue':5, 'covered':False}),
        ((0,0), {'covered':True, 'isMine':True}),
        ((0,2), {'covered':True, 'isMine':True}),
        ((1,0), {'covered':True, 'isMine':True}),
        ((1,1), {'covered':True, 'isMine':True}),
        ((1,2), {'covered':True, 'isMine':True})
    ])
    G.add_edges_from([((0,1), n) for n in [(0,0),(0,2),(1,0),(1,1),(1,2)]])
    return G

def create_1_2_pattern():
    G = nx.Graph()
    G.add_node((0, 0), clue=1, covered=False)
    G.add_node((0, 1), clue=2, covered=False)
    G.add_node((1, 0), covered=True, isMine=False)
    G.add_node((1, 1), covered=True, isMine=True)
    G.add_edges_from([
        ((0,0), (0,1)),
        ((0,0), (1,0)), ((0,1), (1,0)),
        ((0,1), (1,1))
    ])
    return G

def create_2_2_pattern():
    G = nx.Graph()
    G.add_node((0, 0), clue=2, covered=False)
    G.add_node((0, 1), clue=2, covered=False)
    G.add_node((1, 0), covered=True, isMine=True)
    G.add_node((1, 1), covered=True, isMine=True)
    G.add_edges_from([
        ((0,0), (0,1)),
        ((0,0), (1,0)), ((0,1), (1,0)),
        ((0,0), (1,1)), ((0,1), (1,1))
    ])
    return G

def create_corner_1_pattern():
    G = nx.Graph()
    G.add_node((0, 0), clue=1, covered=False)
    G.add_node((0, 1), covered=True, isMine=True)
    G.add_node((1, 0), covered=True, isMine=False)
    G.add_node((1, 1), covered=True, isMine=False)
    G.add_edges_from([
        ((0,0), (0,1)),
        ((0,0), (1,0)),
        ((0,0), (1,1))
    ])
    return G

def create_1_1_1_pattern():
    G = nx.Graph()
    G.add_node((0, 0), clue=1, covered=False)
    G.add_node((0, 1), clue=1, covered=False)
    G.add_node((0, 2), clue=1, covered=False)
    G.add_node((1, 0), covered=True, isMine=False)
    G.add_node((1, 1), covered=True, isMine=True)
    G.add_edges_from([
        ((0,0), (0,1)), ((0,1), (0,2)),
        ((0,0), (1,0)), ((0,1), (1,0)),
        ((0,1), (1,1)), ((0,2), (1,1))
    ])
    return G

def create_triangle_1_1_1_pattern():
    G = nx.Graph()
    G.add_node((0, 0), clue=1, covered=False)
    G.add_node((0, 1), clue=1, covered=False)
    G.add_node((1, 0), clue=1, covered=False)
    G.add_edges_from([
        ((0,0), (0,1)),
        ((0,0), (1,0)),
        ((0,1), (1,0))
    ])
    return G


def candidate_extractor_1_1_shared_mine(board_graph):
    candidates = []
    for node in board_graph.nodes:
        if board_graph.nodes[node].get('clue', -1) != 1 or board_graph.nodes[node].get('covered', True):
            continue
        for neighbor in board_graph.neighbors(node):
            if board_graph.nodes[neighbor].get('clue', -1) != 1 or board_graph.nodes[neighbor].get('covered', True):
                continue
            node_neighbors = set(board_graph.neighbors(node))
            neighbor_neighbors = set(board_graph.neighbors(neighbor))
            shared_neighbors = node_neighbors.intersection(neighbor_neighbors)
            covered_shared = [
                shared_node for shared_node in shared_neighbors
                if board_graph.nodes[shared_node].get('covered', True) and not board_graph.nodes[shared_node].get('flagged', False)
            ]
            if len(covered_shared) == 1:
                candidate = {node, neighbor, covered_shared[0]}
                candidates.append(candidate)
    return candidates

def candidate_extractor_1_2_1(board_graph):
    candidates = []
    n_rows = board_graph.graph["n_rows"]
    n_cols = board_graph.graph["n_cols"]
    for r in range(n_rows):
        for c in range(n_cols - 2):
            node1 = (r, c)
            node2 = (r, c + 1)
            node3 = (r, c + 2)
            if not (board_graph.nodes[node1].get('clue', -1) == 1 and not board_graph.nodes[node1].get('covered', True) and
                    board_graph.nodes[node2].get('clue', -1) == 2 and not board_graph.nodes[node2].get('covered', True) and
                    board_graph.nodes[node3].get('clue', -1) == 1 and not board_graph.nodes[node3].get('covered', True)):
                continue
            neighbors1 = set(board_graph.neighbors(node1))
            neighbors2 = set(board_graph.neighbors(node2))
            neighbors3 = set(board_graph.neighbors(node3))
            shared_neighbors = neighbors1.intersection(neighbors2).intersection(neighbors3)
            covered_shared = [
                shared_node for shared_node in shared_neighbors
                if board_graph.nodes[shared_node].get('covered', True) and not board_graph.nodes[shared_node].get('flagged', False)
            ]
            if len(covered_shared) == 1:
                candidate = {node1, node2, node3, covered_shared[0]}
                candidates.append(candidate)
    for c in range(n_cols):
        for r in range(n_rows - 2):
            node1 = (r, c)
            node2 = (r + 1, c)
            node3 = (r + 2, c)
            if not (board_graph.nodes[node1].get('clue', -1) == 1 and not board_graph.nodes[node1].get('covered', True) and
                    board_graph.nodes[node2].get('clue', -1) == 2 and not board_graph.nodes[node2].get('covered', True) and
                    board_graph.nodes[node3].get('clue', -1) == 1 and not board_graph.nodes[node3].get('covered', True)):
                continue
            neighbors1 = set(board_graph.neighbors(node1))
            neighbors2 = set(board_graph.neighbors(node2))
            neighbors3 = set(board_graph.neighbors(node3))
            shared_neighbors = neighbors1.intersection(neighbors2).intersection(neighbors3)
            covered_shared = [
                shared_node for shared_node in shared_neighbors
                if board_graph.nodes[shared_node].get('covered', True) and not board_graph.nodes[shared_node].get('flagged', False)
            ]
            if len(covered_shared) == 1:
                candidate = {node1, node2, node3, covered_shared[0]}
                candidates.append(candidate)
    return candidates

def candidate_extractor_edge_2(board_graph):
    candidates = []
    n_rows = board_graph.graph["n_rows"]
    n_cols = board_graph.graph["n_cols"]
    for node in board_graph.nodes:
        r, c = node
        if not (r in [0, n_rows - 1] or c in [0, n_cols - 1]):
            continue
        clue = board_graph.nodes[node].get('clue', -1)
        if clue not in [2, 3] or board_graph.nodes[node].get('covered', True):
            continue
        for neighbor in board_graph.neighbors(node):
            neighbor_clue = board_graph.nodes[neighbor].get('clue', -1)
            if board_graph.nodes[neighbor].get('covered', True):
                continue
            if (clue == 2 and neighbor_clue == 3) or (clue == 3 and neighbor_clue == 2):
                node_neighbors = set(board_graph.neighbors(node))
                neighbor_neighbors = set(board_graph.neighbors(neighbor))
                shared_neighbors = node_neighbors.intersection(neighbor_neighbors)
                covered_shared = [
                    shared_node for shared_node in shared_neighbors
                    if board_graph.nodes[shared_node].get('covered', True) and not board_graph.nodes[shared_node].get('flagged', False)
                ]
                if len(covered_shared) == 1:
                    candidate = {node, neighbor, covered_shared[0]}
                    candidates.append(candidate)
    return candidates

def candidate_extractor_edge_pattern(board_graph):
    candidates = []
    n_rows = board_graph.graph["n_rows"]
    n_cols = board_graph.graph["n_cols"]
    for node in board_graph.nodes:
        r, c = node
        if not (r in [0, n_rows - 1] or c in [0, n_cols - 1]):
            continue
        if board_graph.nodes[node].get('clue', -1) != 1 or board_graph.nodes[node].get('covered', True):
            continue
        neighbors = list(board_graph.neighbors(node))
        covered_neighbors = [
            neighbor for neighbor in neighbors
            if board_graph.nodes[neighbor].get('covered', True) and not board_graph.nodes[neighbor].get('flagged', False)
        ]
        if len(covered_neighbors) == 2:
            candidate = {node}.union(covered_neighbors)
            candidates.append(candidate)
        elif len(covered_neighbors) == 3:
            for i in range(len(covered_neighbors)):
                for j in range(i + 1, len(covered_neighbors)):
                    candidate = {node, covered_neighbors[i], covered_neighbors[j]}
                    candidates.append(candidate)
    return candidates

def candidate_extractor_2_3_2_wall(board_graph):
    candidates = []
    n_rows = board_graph.graph["n_rows"]
    n_cols = board_graph.graph["n_cols"]
    for c in range(n_cols):
        for r in range(n_rows - 2):
            node1 = (r, c)
            node2 = (r + 1, c)
            node3 = (r + 2, c)
            if not (board_graph.nodes[node1].get('clue', -1) == 2 and not board_graph.nodes[node1].get('covered', True) and
                    board_graph.nodes[node2].get('clue', -1) == 3 and not board_graph.nodes[node2].get('covered', True) and
                    board_graph.nodes[node3].get('clue', -1) == 2 and not board_graph.nodes[node3].get('covered', True)):
                continue
            covered_neighbors = []
            for pos in [(r + 1, c - 1), (r + 1, c + 1), (r + 2, c - 1), (r + 2, c + 1)]:
                if pos in board_graph.nodes and board_graph.nodes[pos].get('covered', True) and not board_graph.nodes[pos].get('flagged', False):
                    covered_neighbors.append(pos)
            if len(covered_neighbors) >= 3:
                candidate = {node1, node2, node3}.union(covered_neighbors)
                candidates.append(candidate)
    return candidates

def candidate_extractor_false_edge_5(board_graph):
    candidates = []
    n_rows = board_graph.graph["n_rows"]
    n_cols = board_graph.graph["n_cols"]
    for node in board_graph.nodes:
        r, c = node
        if not (r in [0, n_rows - 1] or c in [0, n_cols - 1]):
            continue
        if board_graph.nodes[node].get('clue', -1) != 5 or board_graph.nodes[node].get('covered', True):
            continue
        neighbors = list(board_graph.neighbors(node))
        if len(neighbors) != 5:
            continue
        covered_neighbors = [
            neighbor for neighbor in neighbors
            if board_graph.nodes[neighbor].get('covered', True) and not board_graph.nodes[neighbor].get('flagged', False)
        ]
        if len(covered_neighbors) == 5:
            candidate = {node}.union(covered_neighbors)
            candidates.append(candidate)
    return candidates

def candidate_extractor_1_2(board_graph):
    candidates = []
    for node in board_graph.nodes:
        if board_graph.nodes[node].get('clue', -1) != 1 or board_graph.nodes[node].get('covered', True):
            continue
        for neighbor in board_graph.neighbors(node):
            if board_graph.nodes[neighbor].get('clue', -1) != 2 or board_graph.nodes[neighbor].get('covered', True):
                continue
            node_neighbors = set(board_graph.neighbors(node))
            neighbor_neighbors = set(board_graph.neighbors(neighbor))
            shared_neighbors = node_neighbors.intersection(neighbor_neighbors)
            covered_shared = [
                shared_node for shared_node in shared_neighbors
                if board_graph.nodes[shared_node].get('covered', True) and not board_graph.nodes[shared_node].get('flagged', False)
            ]
            if len(covered_shared) != 1:
                continue
            unique_neighbors = neighbor_neighbors - node_neighbors
            covered_unique = [
                unique_node for unique_node in unique_neighbors
                if board_graph.nodes[unique_node].get('covered', True) and not board_graph.nodes[unique_node].get('flagged', False)
            ]
            if len(covered_unique) == 1:
                candidate = {node, neighbor, covered_shared[0], covered_unique[0]}
                candidates.append(candidate)
    return candidates

def candidate_extractor_2_2(board_graph):
    candidates = []
    for node in board_graph.nodes:
        if board_graph.nodes[node].get('clue', -1) != 2 or board_graph.nodes[node].get('covered', True):
            continue
        for neighbor in board_graph.neighbors(node):
            if board_graph.nodes[neighbor].get('clue', -1) != 2 or board_graph.nodes[neighbor].get('covered', True):
                continue
            node_neighbors = set(board_graph.neighbors(node))
            neighbor_neighbors = set(board_graph.neighbors(neighbor))
            shared_neighbors = node_neighbors.intersection(neighbor_neighbors)
            covered_shared = [
                shared_node for shared_node in shared_neighbors
                if board_graph.nodes[shared_node].get('covered', True) and not board_graph.nodes[shared_node].get('flagged', False)
            ]
            if len(covered_shared) == 2:
                candidate = {node, neighbor}.union(covered_shared)
                candidates.append(candidate)
    return candidates

def candidate_extractor_corner_1(board_graph):
    candidates = []
    n_rows = board_graph.graph["n_rows"]
    n_cols = board_graph.graph["n_cols"]
    corners = [(0, 0), (0, n_cols - 1), (n_rows - 1, 0), (n_rows - 1, n_cols - 1)]
    for node in corners:
        if node not in board_graph.nodes:
            continue
        if board_graph.nodes[node].get('clue', -1) != 1 or board_graph.nodes[node].get('covered', True):
            continue
        neighbors = list(board_graph.neighbors(node))
        covered_neighbors = [
            neighbor for neighbor in neighbors
            if board_graph.nodes[neighbor].get('covered', True) and not board_graph.nodes[neighbor].get('flagged', False)
        ]
        if len(covered_neighbors) == len(neighbors) and len(neighbors) in [3, 4]:
            candidate = {node}.union(covered_neighbors)
            candidates.append(candidate)
    return candidates

def candidate_extractor_1_1_1(board_graph):
    candidates = []
    n_rows = board_graph.graph["n_rows"]
    n_cols = board_graph.graph["n_cols"]
    for r in range(n_rows):
        for c in range(n_cols - 2):
            node1 = (r, c)
            node2 = (r, c + 1)
            node3 = (r, c + 2)
            if not (board_graph.nodes[node1].get('clue', -1) == 1 and not board_graph.nodes[node1].get('covered', True) and
                    board_graph.nodes[node2].get('clue', -1) == 1 and not board_graph.nodes[node2].get('covered', True) and
                    board_graph.nodes[node3].get('clue', -1) == 1 and not board_graph.nodes[node3].get('covered', True)):
                continue
            neighbors1 = set(board_graph.neighbors(node1))
            neighbors2 = set(board_graph.neighbors(node2))
            neighbors3 = set(board_graph.neighbors(node3))
            shared_1_2 = neighbors1.intersection(neighbors2)
            covered_1_2 = [
                shared_node for shared_node in shared_1_2
                if board_graph.nodes[shared_node].get('covered', True) and not board_graph.nodes[shared_node].get('flagged', False)
            ]
            shared_2_3 = neighbors2.intersection(neighbors3)
            covered_2_3 = [
                shared_node for shared_node in shared_2_3
                if board_graph.nodes[shared_node].get('covered', True) and not board_graph.nodes[shared_node].get('flagged', False)
            ]
            covered_neighbors = set(covered_1_2).union(covered_2_3)
            if len(covered_neighbors) == 2:
                candidate = {node1, node2, node3}.union(covered_neighbors)
                candidates.append(candidate)
    for c in range(n_cols):
        for r in range(n_rows - 2):
            node1 = (r, c)
            node2 = (r + 1, c)
            node3 = (r + 2, c)
            if not (board_graph.nodes[node1].get('clue', -1) == 1 and not board_graph.nodes[node1].get('covered', True) and
                    board_graph.nodes[node2].get('clue', -1) == 1 and not board_graph.nodes[node2].get('covered', True) and
                    board_graph.nodes[node3].get('clue', -1) == 1 and not board_graph.nodes[node3].get('covered', True)):
                continue
            neighbors1 = set(board_graph.neighbors(node1))
            neighbors2 = set(board_graph.neighbors(node2))
            neighbors3 = set(board_graph.neighbors(node3))
            shared_1_2 = neighbors1.intersection(neighbors2)
            covered_1_2 = [
                shared_node for shared_node in shared_1_2
                if board_graph.nodes[shared_node].get('covered', True) and not board_graph.nodes[shared_node].get('flagged', False)
            ]
            shared_2_3 = neighbors2.intersection(neighbors3)
            covered_2_3 = [
                shared_node for shared_node in shared_2_3
                if board_graph.nodes[shared_node].get('covered', True) and not board_graph.nodes[shared_node].get('flagged', False)
            ]
            covered_neighbors = set(covered_1_2).union(covered_2_3)
            if len(covered_neighbors) == 2:
                candidate = {node1, node2, node3}.union(covered_neighbors)
                candidates.append(candidate)
    return candidates

def candidate_extractor_triangle_1_1_1(board_graph):
    candidates = []
    for node1 in board_graph.nodes:
        if board_graph.nodes[node1].get('clue', -1) != 1 or board_graph.nodes[node1].get('covered', True):
            continue
        neighbors1 = set(board_graph.neighbors(node1))
        for node2 in neighbors1:
            if board_graph.nodes[node2].get('clue', -1) != 1 or board_graph.nodes[node2].get('covered', True):
                continue
            neighbors2 = set(board_graph.neighbors(node2))
            shared_neighbors = neighbors1.intersection(neighbors2)
            for node3 in shared_neighbors:
                if node3 == node1 or node3 == node2:
                    continue
                if board_graph.nodes[node3].get('clue', -1) != 1 or board_graph.nodes[node3].get('covered', True):
                    continue
                if node1 in board_graph.neighbors(node3) and node2 in board_graph.neighbors(node3):
                    candidate = {node1, node2, node3}
                    candidates.append(candidate)
    return candidates


PATTERNS = {
    "1_1_shared_mine": {
        'graph': create_1_1_pattern(),
        'priority': 1,
        'candidate_extractor': candidate_extractor_1_1_shared_mine
    },
    "1_2_1": {
        'graph': create_1_2_1_pattern(),
        'priority': 2,
        'candidate_extractor': candidate_extractor_1_2_1
    },
    "edge_2": {
        'graph': create_edge_2_pattern(),
        'priority': 3,
        'candidate_extractor': candidate_extractor_edge_2
    },
    "edge_pattern": {
        'graph': create_edge_pattern(),
        'priority': 4,
        'candidate_extractor': candidate_extractor_edge_pattern
    },
    "2_3_2_wall": {
        'graph': create_2_3_2_wall(),
        'priority': 5,
        'candidate_extractor': candidate_extractor_2_3_2_wall
    },
    "false_edge_5": {
        'graph': create_false_edge_5(),
        'priority': 6,
        'candidate_extractor': candidate_extractor_false_edge_5
    },
    "1_2": {
        'graph': create_1_2_pattern(),
        'priority': 7,
        'candidate_extractor': candidate_extractor_1_2
    },
    "2_2": {
        'graph': create_2_2_pattern(),
        'priority': 8,
        'candidate_extractor': candidate_extractor_2_2
    },
    "corner_1": {
        'graph': create_corner_1_pattern(),
        'priority': 9,
        'candidate_extractor': candidate_extractor_corner_1
    },
    "1_1_1": {
        'graph': create_1_1_1_pattern(),
        'priority': 10,
        'candidate_extractor': candidate_extractor_1_1_1
    },
    "triangle_1_1_1": {
        'graph': create_triangle_1_1_1_pattern(),
        'priority': 11,
        'candidate_extractor': candidate_extractor_triangle_1_1_1
    },
}
