import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ppl
from ppl import Variable, C_Polyhedron, Constraint_System
import gym
import torch
from typing import List, Tuple

class MiniTangramEnv:
    def __init__(self):
        self.x, self.y = Variable(0), Variable(1)
        # Define a 4x2 Silhouette Target at x:[2,6], y:[2,4]
        cs_target = Constraint_System()
        cs_target.insert(self.x >= 2); cs_target.insert(self.x <= 6)
        cs_target.insert(self.y >= 2); cs_target.insert(self.y <= 4)
        self.target_poly = C_Polyhedron(cs_target)
        self.reset()

    def reset(self):
        # Piece 0: 2x2 Square at (0,2) -> Perfectly fits left half of target
        # Piece 1: 2x2 Square at (7,2) -> Outside to the right
        self.pieces = [
            self._create_square(0, 2, 2),
            self._create_square(8, 2, 2)
        ]
        return self.pieces

    def _create_square(self, x, y, side):
        cs = Constraint_System()
        cs.insert(self.x >= x); cs.insert(self.x <= x + side)
        cs.insert(self.y >= y); cs.insert(self.y <= y + side)
        return C_Polyhedron(cs)

    def _get_area(self, poly):
        """Calculates the area of a 2D convex polygon using the Shoelace formula."""
        if poly.is_empty(): return 0.0
        # Get vertices and sort them counter-clockwise
        verts = []
        for g in poly.generators():
            if g.is_point():
                verts.append((float(g.coefficient(self.x)), float(g.coefficient(self.y))))
        
        if len(verts) < 3: return 0.0 # Lines/Points have no area
        
        # Sort by angle from centroid
        centroid = np.mean(verts, axis=0)
        verts.sort(key=lambda p: np.arctan2(p[1]-centroid[1], p[0]-centroid[0]))
        
        # Shoelace formula
        x, y = zip(*verts)
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def _poly_to_constraints(self, poly: C_Polyhedron) -> List[Tuple[List[float], float]]:
        """Extract H-representation from polytope.

        Args:
            poly: C_Polyhedron object

        Returns:
            List of (coefficients, bound) tuples for constraints
        """
        constraints = []

        for constraint in poly.minimized_constraints():
            coeffs = []
            bound = float(constraint.inhomogeneous_term())

            # Extract coefficients for x and y
            coeff_x = float(constraint.coefficient(self.x))
            coeff_y = float(constraint.coefficient(self.y))

            coeffs = [-coeff_x, -coeff_y]  # Negate because PPL uses >= 0 form

            # For equality constraints, add both <= and >= constraints
            if constraint.is_equality():
                constraints.append((coeffs, bound))
                constraints.append(([-c for c in coeffs], -bound))
            else:
                # For inequality constraints (<= form)
                constraints.append((coeffs, bound))

        return constraints

    def _poly_to_vertices(self, poly: C_Polyhedron) -> List[Tuple[float, float]]:
        """Extract V-representation (vertices) from polytope.

        Args:
            poly: C_Polyhedron object

        Returns:
            List of (x, y) vertex coordinates
        """
        vertices = []

        try:
            for gen in poly.generators():
                if gen.is_point():
                    coords = []
                    for i in range(2):
                        var = Variable(i)
                        coords.append(float(gen.coefficient(var)))
                    vertices.append(tuple(coords))
        except Exception as e:
            print(f"Warning: Could not extract vertices: {e}")

        return vertices

    def _build_constraint_graph(self, h_constraints, v_vertices, tolerance=1e-6):
        """Build constraint graph from H-representation and V-representation.

        Args:
            h_constraints: list of (coeffs, bound) tuples - normalized constraints
            v_vertices: list of (x, y) tuples - vertex coordinates
            tolerance: numerical tolerance for constraint satisfaction

        Returns:
            dict with graph components:
            - "nodes": tensor [num_constraints, 3] (a1, a2, b)
            - "edges": list of (i,j) tuples for connectivity
            - "edge_attr": tensor [num_edges, 2] (vertex_x, vertex_y)
        """
        if len(h_constraints) == 0:
            # Empty graph
            return {
                "nodes": torch.zeros(1, 3),
                "edges": [],
                "edge_attr": torch.zeros(0, 2)
            }

        # Node features: constraint parameters
        node_features = []
        for coeffs, bound in h_constraints:
            node_features.append(coeffs + [bound])
        nodes = torch.tensor(node_features, dtype=torch.float32)

        # Build edges based on vertices
        edges = []
        edge_attr = []

        for vertex in v_vertices:
            vx, vy = vertex
            # Find constraints satisfied at this vertex (equality within tolerance)
            active_constraints = []
            for i, (coeffs, bound) in enumerate(h_constraints):
                a1, a2 = coeffs
                constraint_value = a1 * vx + a2 * vy
                if abs(constraint_value - bound) < tolerance:
                    active_constraints.append(i)

            # Create edges between all pairs of active constraints
            for i in range(len(active_constraints)):
                for j in range(i + 1, len(active_constraints)):
                    c1_idx = active_constraints[i]
                    c2_idx = active_constraints[j]

                    # Add bidirectional edges
                    edges.extend([(c1_idx, c2_idx), (c2_idx, c1_idx)])
                    edge_attr.extend([vertex, vertex])  # Same vertex for both directions

        # Convert to tensors
        if len(edge_attr) > 0:
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        else:
            edge_attr = torch.zeros(0, 2, dtype=torch.float32)

        return {
            "nodes": nodes,
            "edges": edges,
            "edge_attr": edge_attr
        }

    def step(self, piece_idx, dx, dy):
        current_poly = self.pieces[piece_idx]
        new_poly = C_Polyhedron(current_poly)
        new_poly.affine_image(self.x, self.x + dx)
        new_poly.affine_image(self.y, self.y + dy)
        
        # Collision Detection
        other_pieces = [p for i, p in enumerate(self.pieces) if i != piece_idx]
        overlap = False
        for other in other_pieces:
            intersection = C_Polyhedron(new_poly)
            intersection.intersection_assign(other)
            if not intersection.is_empty():
                overlap = True
                break
        
        if not overlap:
            self.pieces[piece_idx] = new_poly
            # Calculate reward: Area of pieces inside target
            inter_area = 0
            for p in self.pieces:
                it = C_Polyhedron(p)
                it.intersection_assign(self.target_poly)
                inter_area += self._get_area(it)
            return inter_area, "Move Successful"
        else:
            return -1, "Collision Detected! Move Blocked."

    def render(self, title="Current State", show=True):
        fig, ax = plt.subplots(figsize=(8, 5))
        # Plot Target
        self._plot_poly(ax, self.target_poly, color='gray', alpha=0.2, label='Target (Goal)')
        # Plot Pieces
        colors = ['red', 'blue']
        for i, p in enumerate(self.pieces):
            self._plot_poly(ax, p, color=colors[i], alpha=0.5, label=f'Piece {i}')
        
        plt.title(title)
        plt.xlim(-1, 10); plt.ylim(0, 6)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        if show:
            plt.show()
        else:
            plt.close(fig)

    def _plot_poly(self, ax, poly, **kwargs):
        verts = []
        for g in poly.generators():
            if g.is_point():
                verts.append((float(g.coefficient(self.x)), float(g.coefficient(self.y))))
        if len(verts) >= 3:
            centroid = np.mean(verts, axis=0)
            verts.sort(key=lambda p: np.arctan2(p[1]-centroid[1], p[0]-centroid[0]))
            ax.add_patch(patches.Polygon(verts, **kwargs))


class TangramGymEnv(gym.Env):
    """Gym-compatible wrapper for tangram puzzle environment."""

    def __init__(self, step_size=1.0):
        """Initialize environment."""
        super().__init__()
        self.inner = MiniTangramEnv()
        self.step_size = step_size
        # Actions: 0-3: piece 0 (left, right, up, down)
        # 4-7: piece 1 (left, right, up, down)
        self.action_space = gym.spaces.Discrete(8)
        self.observation_space = None  # Dict space
        self.target_area = 8.0  # 4x2 target

    def reset(self):
        """Reset environment."""
        self.inner.reset()
        return self._get_obs()

    def _get_obs(self):
        """Get current observation with all representations."""
        # For pieces
        vertices = []
        constraints = []
        graphs = []
        for piece in self.inner.pieces:
            verts = self.inner._poly_to_vertices(piece)
            cons = self.inner._poly_to_constraints(piece)
            graph = self.inner._build_constraint_graph(cons, verts)
            vertices.append(verts)
            constraints.extend(cons)  # Concatenate all constraints
            graphs.append(graph)

        # For target
        target_vertices = self.inner._poly_to_vertices(self.inner.target_poly)
        target_constraints = self.inner._poly_to_constraints(self.inner.target_poly)
        target_graph = self.inner._build_constraint_graph(target_constraints, target_vertices)

        return {
            "location": "zone0",  # Dummy location for compatibility
            "vertices": vertices,
            "constraints": constraints,  # Concatenated
            "graph": graphs,
            "target_vertices": target_vertices,
            "target_constraints": target_constraints,
            "target_graph": target_graph,
        }

    def step(self, action):
        """Execute one step in environment."""
        assert self.action_space.contains(action), "Invalid action"

        # Decode action
        piece_idx = action // 4
        direction = action % 4
        dx, dy = 0, 0
        if direction == 0:  # left
            dx = -self.step_size
        elif direction == 1:  # right
            dx = self.step_size
        elif direction == 2:  # up
            dy = self.step_size
        elif direction == 3:  # down
            dy = -self.step_size

        # Execute move
        reward, msg = self.inner.step(piece_idx, dx, dy)

        # Check if done
        done = reward >= self.target_area - 1e-6  # Close enough

        info = {
            "msg": msg,
            "piece_moved": piece_idx,
            "dx": dx,
            "dy": dy,
        }

        return self._get_obs(), reward, done, info

    def render(self, mode='human'):
        """Render current state."""
        self.inner.render("Gym Environment State", show=(mode == 'human'))

    def evaluate(self, agent, episodes=10, max_steps=100, render=False):
        """Evaluate agent performance."""
        eval_rewards = []

        for ep in range(episodes):
            obs = self.reset()
            ep_reward = 0

            for step in range(max_steps):
                action = agent.select_action(obs, training=False)
                next_obs, reward, done, info = self.step(action)

                ep_reward += reward
                obs = next_obs

                if render:
                    print(f"  Step {step+1}: action={action}, reward={reward}, done={done}")

                if done:
                    break

            eval_rewards.append(ep_reward)
            if render:
                print(f"Episode {ep+1}: Total Reward = {ep_reward}\n")

        return eval_rewards

# --- Execution ---
env = MiniTangramEnv()
env.render("Initial Position (Reward: 0)")

# Action 1: Move Piece 0 (Red) onto the left side of the target
reward, msg = env.step(0, 2, 0)
env.render(f"Action: Move Red Right | Reward: {reward} | {msg}")

# Action 2: Move Piece 1 (Blue) onto the right side of the target
reward, msg = env.step(1, -3, 0)
env.render(f"Action: Move Blue Left | Reward: {reward} | {msg}")

# Action 3: Try to move Blue into Red (Collision)
reward, msg = env.step(1, -2, 0)
env.render(f"Action: Move Blue into Red | Reward: {reward} | {msg}")