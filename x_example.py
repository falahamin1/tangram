import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ppl
from ppl import Variable, C_Polyhedron, Constraint_System
import gym
from gym import spaces
import torch
import os

class XTangramEnv:
    def __init__(self):
        self.x, self.y = Variable(0), Variable(1)
        # Define the Goal (Silhouette)
        self.target_pieces = [
            self._create_square(4, 3),       # Target P0 (Red)
            self._create_square(4, 5),       # Target P1 (Orange)
            self._create_triangle(2, 2, "SE"), # Target P2 (Green)
            self._create_triangle(6, 2, "SW"), # Target P3 (Teal)
            self._create_triangle(2, 7, "NE"), # Target P4 (Blue)
            self._create_triangle(6, 7, "NW")  # Target P5 (Purple)
        ]
        
        # PRE-CALCULATE target areas for normalization
        self.target_areas = [self._calculate_area(tp) for tp in self.target_pieces]
        # PRE-CALCULATE target centroids for potential-based shaping (cheap, no PPL)
        self.target_centroids = [self._poly_centroid(tp) for tp in self.target_pieces]
        self.reset()

    def _create_square(self, x, y):
        cs = Constraint_System()
        cs.insert(self.x >= x); cs.insert(self.x <= x + 2)
        cs.insert(self.y >= y); cs.insert(self.y <= y + 2)
        return C_Polyhedron(cs)

    def _create_triangle(self, x, y, type="NW"):
        cs = Constraint_System()
        cs.insert(self.x >= x); cs.insert(self.x <= x + 2)
        cs.insert(self.y >= y); cs.insert(self.y <= y + 2)
        if type == "SE": cs.insert(self.y <= -self.x + (x + y + 2)) 
        if type == "SW": cs.insert(self.y <= self.x + (y - x + 2))  
        if type == "NE": cs.insert(self.y >= self.x + (y - x))      
        if type == "NW": cs.insert(self.y >= -self.x + (x + y))    
        return C_Polyhedron(cs)

    def reset(self):
        self.pieces = [
            self._create_square(0, 0),       # Piece 0
            self._create_square(8, 0),       # Piece 1
            self._create_triangle(0, 8, "SE"),
            self._create_triangle(2, 9, "SW"),
            self._create_triangle(7, 9, "NE"),
            self._create_triangle(9, 8, "NW")
        ]
        self.locked = [False] * len(self.pieces)

    def _poly_centroid(self, poly):
        """Return the centroid of a polytope as a numpy array [x, y]."""
        verts = [(float(g.coefficient(self.x)), float(g.coefficient(self.y)))
                 for g in poly.generators() if g.is_point()]
        return np.array(verts).mean(axis=0) if verts else np.zeros(2)

    def _calculate_area(self, poly):
        if poly.is_empty(): return 0.0
        verts = []
        for g in poly.generators():
            if g.is_point():
                verts.append((float(g.coefficient(self.x)), float(g.coefficient(self.y))))
        if len(verts) < 3: return 0.0
        centroid = np.mean(verts, axis=0)
        verts.sort(key=lambda p: np.arctan2(p[1]-centroid[1], p[0]-centroid[0]))
        x, y = zip(*verts)
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _is_in_target(self, piece_idx):
        """True when the piece fully covers its target silhouette (>= 99% overlap)."""
        overlap = C_Polyhedron(self.pieces[piece_idx])
        overlap.intersection_assign(self.target_pieces[piece_idx])
        if overlap.is_empty():
            return False
        return self._calculate_area(overlap) / self.target_areas[piece_idx] >= 0.99

    def move_piece(self, piece_idx, dx, dy):
        """Move piece unconditionally (overlaps with other pieces are allowed).

        Returns (normalized_overlap_with_own_target, message).
        Locked pieces are rejected; callers should rely on the action mask.
        """
        if self.locked[piece_idx]:
            return 1.0, "Locked"

        new_poly = C_Polyhedron(self.pieces[piece_idx])
        new_poly.affine_image(self.x, self.x + dx)
        new_poly.affine_image(self.y, self.y + dy)
        self.pieces[piece_idx] = new_poly

        overlap = C_Polyhedron(self.pieces[piece_idx])
        overlap.intersection_assign(self.target_pieces[piece_idx])
        normalized_overlap = self._calculate_area(overlap) / self.target_areas[piece_idx]

        if self._is_in_target(piece_idx):
            self.locked[piece_idx] = True

        return normalized_overlap, f"Success: {normalized_overlap:.2f}"

    def render(self, save_path):
        """Render the current state and save to save_path (must include extension)."""
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_title(os.path.basename(save_path))
        for tp in self.target_pieces:
            self._plot_poly(ax, tp, color='gray', alpha=0.1, linestyle='--')
        colors = ['#FF5733', '#FFBD33', '#33FF57', '#33FFBD', '#3357FF', '#BD33FF']
        for i, p in enumerate(self.pieces):
            self._plot_poly(ax, p, color=colors[i], alpha=0.6)
        plt.xlim(-1, 11); plt.ylim(-1, 11)
        dir_name = os.path.dirname(save_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        plt.savefig(save_path)
        plt.close()

    def _plot_poly(self, ax, poly, **kwargs):
        verts = []
        for g in poly.generators():
            if g.is_point():
                verts.append((float(g.coefficient(self.x)), float(g.coefficient(self.y))))
        if len(verts) >= 3:
            centroid = np.mean(verts, axis=0)
            verts.sort(key=lambda p: np.arctan2(p[1]-centroid[1], p[0]-centroid[0]))
            ax.add_patch(patches.Polygon(verts, **kwargs))

class SymbolicTangramGym(gym.Env):
    def __init__(self):
        super().__init__()
        self.inner = XTangramEnv()
        self.num_pieces = len(self.inner.pieces)
        self.max_steps = 300
        self.step_count = 0
        self.gamma = 0.99
        
        # Action Space: (Piece_Index * 4) + Direction
        self.action_space = spaces.Discrete(self.num_pieces * 4)
        
        # RESTORED: Observation Space as a Dictionary
        self.observation_space = spaces.Dict({
            "v_rep": spaces.Box(low=0, high=1, shape=(self.num_pieces, 4, 2)), 
            "h_rep": spaces.Box(low=-1, high=1, shape=(self.num_pieces, 5, 3)),
            "adj": spaces.Box(low=0, high=1, shape=(self.num_pieces, 5, 5)) 
        })

    def _get_obs(self):
        """Returns the dictionary observation required by the training script."""
        h_rep_data = self._extract_h_rep()
        v_rep_data = self._extract_v_rep()
        graph_data = self._build_graph_adj()
        
        # If your training script uses the GNN graph logic, 
        # you can call your _build_graph_rep(h_rep_data) here.
        return {
            "v_rep": v_rep_data,
            "h_rep": h_rep_data,
            "adj": graph_data # Returning h_rep as node features for the graph
        }

    def _extract_h_rep(self):
        h_rep = []
        board_max = 11.0
        for p in self.inner.pieces:
            constraints = []
            for c in p.minimized_constraints():
                a1 = -float(c.coefficient(self.inner.x))
                a2 = -float(c.coefficient(self.inner.y))
                b = float(c.inhomogeneous_term())
                norm = np.sqrt(a1**2 + a2**2) if (a1**2 + a2**2) > 0 else 1.0
                # Normalize b by board size to keep values in a similar range
                constraints.append([a1/norm, a2/norm, (b/norm) / board_max])
            
            while len(constraints) < 5: 
                constraints.append([0.0, 0.0, 0.0])
            h_rep.append(constraints[:5])
        return np.array(h_rep, dtype=np.float32)

    def _extract_v_rep(self):
        v_rep = []
        board_max = 11.0
        for p in self.inner.pieces:
            verts = []
            for g in p.generators():
                if g.is_point():
                    x_val = float(g.coefficient(self.inner.x)) / board_max
                    y_val = float(g.coefficient(self.inner.y)) / board_max
                    verts.append([x_val, y_val])
            while len(verts) < 4: 
                verts.append([0.0, 0.0]) 
            v_rep.append(verts[:4])
        return np.array(v_rep, dtype=np.float32)
    
    def _build_graph_adj(self):
        """
        Builds adjacency matrices for each piece.
        An edge exists between two half-spaces if they share a common vertex.
        """
        all_adj = []
        epsilon = 1e-5 # Tolerance for floating point checks

        for p_idx, p in enumerate(self.inner.pieces):
            constraints = list(p.minimized_constraints())
            vertices = [g for g in p.generators() if g.is_point()]
            
            # Initialize 5x5 adjacency matrix (padded to max constraints)
            adj = np.zeros((5, 5), dtype=np.float32)
            
            # We only care about the first 5 constraints (matching h_rep padding)
            num_c = min(len(constraints), 5)
            
            # Check every pair of constraints (i, j)
            for i in range(num_c):
                for j in range(i + 1, num_c):
                    # Do these two constraints share a vertex?
                    shared = False
                    for v in vertices:
                        # Evaluate constraint i and j at vertex v
                        # v satisfies constraint c exactly if its boundary value is 0
                        val_i = self._eval_constraint_at_vertex(constraints[i], v)
                        val_j = self._eval_constraint_at_vertex(constraints[j], v)
                        
                        if abs(val_i) < epsilon and abs(val_j) < epsilon:
                            shared = True
                            break
                    
                    if shared:
                        adj[i, j] = 1.0
                        adj[j, i] = 1.0
            
            all_adj.append(adj)
            
        return np.array(all_adj, dtype=np.float32)

    def _eval_constraint_at_vertex(self, constraint, vertex):
        """Helper to evaluate PPL constraint at a PPL vertex."""
        x_val = float(vertex.coefficient(self.inner.x)) / vertex.divisor()
        y_val = float(vertex.coefficient(self.inner.y)) / vertex.divisor()
        
        # a1*x + a2*y + b
        a1 = -float(constraint.coefficient(self.inner.x))
        a2 = -float(constraint.coefficient(self.inner.y))
        b = float(constraint.inhomogeneous_term())
        
        return a1 * x_val + a2 * y_val + b

    def _potential(self):
        """Φ(s) = -(mean centroid distance to own target), higher = better.

        Locked pieces contribute 0 distance. Used for potential-based shaping:
        shaped reward = γΦ(s') - Φ(s) = mean_dist(s) - γ·mean_dist(s'),
        which is positive whenever the agent moves pieces closer to their targets.
        """
        total_dist = 0.0
        for i in range(self.num_pieces):
            if self.inner.locked[i]:
                continue
            c = self.inner._poly_centroid(self.inner.pieces[i])
            total_dist += np.linalg.norm(c - self.inner.target_centroids[i])
        return -total_dist / self.num_pieces

    def _get_total_completion(self):
        """Mean per-piece normalized overlap with own target, in [0, 1]. Used for logging."""
        total_normalized = 0
        for i in range(self.num_pieces):
            if self.inner.locked[i]:
                total_normalized += 1.0
                continue
            overlap = C_Polyhedron(self.inner.pieces[i])
            overlap.intersection_assign(self.inner.target_pieces[i])
            total_normalized += self.inner._calculate_area(overlap) / self.inner.target_areas[i]
        return total_normalized / self.num_pieces

    def step(self, action):
        self.step_count += 1
        piece_idx = action // 4
        direction = action % 4
        dx, dy = [(0, 1), (0, -1), (-1, 0), (1, 0)][direction]

        phi_before = self._potential()/10 # Scale potential to keep it in a similar range as the step penalty and bonuses
        locked_before = list(self.inner.locked)

        _, msg = self.inner.move_piece(piece_idx, dx, dy)

        phi_after = self._potential()/10 

        # Step penalty
        reward = -0.01
        # Potential-based shaping (Ng et al. 1999): γΦ(s') - Φ(s)
        # With Φ = -distance, this equals dist(s) - γ·dist(s'):
        # positive when pieces move closer to their targets.
        reward += self.gamma * phi_after - phi_before
        # Bonus for each newly locked piece
        for i in range(self.num_pieces):
            if self.inner.locked[i] and not locked_before[i]:
                reward += 1.0
        # Completion bonus when every piece is in its target
        done = False
        if all(self.inner.locked):
            reward += 10.0
            done = True
        elif self.step_count >= self.max_steps:
            done = True

        completion = sum(self.inner.locked) / self.num_pieces
        return self._get_obs(), reward, done, {"message": msg, "completion": completion}

    def reset(self):
        self.step_count = 0
        self.inner.reset()
        return self._get_obs()
    
    def get_action_mask(self):
        """Boolean mask of valid actions.

        An action is invalid if:
          - the piece is locked (already in its target silhouette), or
          - the move would push any vertex outside the board boundary [0, 14].
        Overlaps between pieces are intentionally allowed.
        """
        mask = np.ones(self.action_space.n, dtype=bool)

        for action in range(self.action_space.n):
            piece_idx = action // 4

            # Locked pieces are sink states — no further moves allowed.
            if self.inner.locked[piece_idx]:
                mask[action] = False
                continue

            direction = action % 4
            dx, dy = [(0, 1), (0, -1), (-1, 0), (1, 0)][direction]

            new_poly = C_Polyhedron(self.inner.pieces[piece_idx])
            new_poly.affine_image(self.inner.x, self.inner.x + dx)
            new_poly.affine_image(self.inner.y, self.inner.y + dy)

            for g in new_poly.generators():
                if g.is_point():
                    x_val = float(g.coefficient(self.inner.x))
                    y_val = float(g.coefficient(self.inner.y))
                    if not (0 <= x_val <= 14 and 0 <= y_val <= 14):
                        mask[action] = False
                        break

        return mask
    
