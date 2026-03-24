import hashlib
import random
import numpy as np
from scipy.signal import convolve2d
from scipy.sparse import lil_matrix
import networkx as nx


class Grid:
    # Define constants at class level
    DIVISION_MORPHOGEN_INDEX = 0
    DIFFERENTIATION_MORPHOGEN_INDEX = 1
    AXON_GUIDANCE_MORPHOGEN_INDEX = 2
    AXON_CONNECT_MORPHOGEN_INDEX = 3
    WEIGHT_ADJUSTMENT_MORPHOGEN_INDEX = 4

    def __init__(self, genome):
        # Initialize random number generator with deterministic hash from genome bytes
        genome_bytes = genome.to_bytes()
        hash_value = int.from_bytes(hashlib.md5(genome_bytes).digest(), "big")
        self.rng = random.Random(hash_value)

        self.size_x = genome.size_x
        self.size_y = genome.size_y
        self.diffusion_rate = genome.diffusion_rate
        self.num_morphogens = genome.num_morphogens
        self.division_threshold = genome.division_threshold
        self.cell_differentiation_threshold = genome.cell_differentiation_threshold
        self.axon_growth_threshold = genome.axon_growth_threshold
        self.max_axon_length = genome.max_axon_length
        self.axon_connect_threshold = genome.axon_connect_threshold
        self.self_connect_isolated_neurons_fraction = (
            genome.self_connect_isolated_neurons_fraction
        )
        self.progenitor_secretion_rates = genome.progenitor_secretion_rates
        self.neuron_secretion_rates = genome.neuron_secretion_rates
        self.inhibition_matrix = genome.inhibition_matrix
        self.diffusion_patterns = genome.diffusion_patterns
        self.max_growth_steps = genome.max_growth_steps

        # Add new genome parameters for weight adjustment
        self.weight_adjustment_target = genome.weight_adjustment_target
        self.weight_adjustment_rate = genome.weight_adjustment_rate

        # Initialize remaining attributes as before
        self._M = np.zeros((self.num_morphogens, self.size_x, self.size_y))
        self._division_morphogen = self._M[self.DIVISION_MORPHOGEN_INDEX]
        self._differentiation_morphogen = self._M[self.DIFFERENTIATION_MORPHOGEN_INDEX]
        self._axon_guidance_morphogen = self._M[self.AXON_GUIDANCE_MORPHOGEN_INDEX]
        self._axon_connect_morphogen = self._M[
            self.AXON_CONNECT_MORPHOGEN_INDEX
            if self.num_morphogens > 3
            else self.AXON_GUIDANCE_MORPHOGEN_INDEX
        ]
        self._weight_adjustment_morphogen = self._M[
            self.WEIGHT_ADJUSTMENT_MORPHOGEN_INDEX
            if self.num_morphogens > 4
            else (
                self.AXON_CONNECT_MORPHOGEN_INDEX
                if self.num_morphogens == 4
                else self.AXON_GUIDANCE_MORPHOGEN_INDEX
            )
        ]

        self.iteration = 0
        self._neurons = np.zeros((self.size_x, self.size_y), dtype=int)
        self._progenitors = np.zeros((self.size_x, self.size_y), dtype=int)
        self._cell_positions = {}
        self._axons = {}
        self._max_cell_id = 0
        self.neuron_connections = lil_matrix(
            (self.size_x * self.size_y, self.size_x * self.size_y), dtype=float
        )
        self.listeners = []

    def get_graph(self):
        """Get the graph of the neural network with connection weights."""
        G = nx.DiGraph()

        # Add nodes for all neurons
        neuron_ids = self.get_neuron_ids()
        for cell_id in neuron_ids:
            G.add_node(cell_id)

        # Add edges from connectivity matrix with their weights
        source_indices, target_indices = self.neuron_connections.nonzero()
        weights = (
            self.neuron_connections[source_indices, target_indices].toarray().flatten()
        )

        for source, target, weight in zip(
            source_indices + 1, target_indices + 1, weights
        ):
            if source in G.nodes() and target in G.nodes():
                G.add_edge(source, target, weight=float(weight))

        return G

    def add_cell(self, position, cell_type="progenitor"):
        """Add a new cell at the given position."""
        if self._neurons[position] == 0 and self._progenitors[position] == 0:
            cell_id = self._max_cell_id + 1
            self._max_cell_id = cell_id

            if cell_type == "neuron":
                self._neurons[position] = cell_id
                self._axons[cell_id] = [position]  # Initialize axon for neurons
            else:  # progenitor
                self._progenitors[position] = cell_id

            self._cell_positions[cell_id] = position

    def diffuse(self):
        # Update each morphogen independently with its own kernel
        for i in range(self.num_morphogens):
            # Convolve the matrix with the morphogen's specific kernel
            diffused = convolve2d(
                self._M[i], self.diffusion_patterns[i], mode="same", boundary="wrap"
            )

            # Update the matrix using the diffusion rate
            self._M[i] = (1 - self.diffusion_rate) * self._M[
                i
            ] + self.diffusion_rate * diffused

        self.iteration += 1

    def inhibit_morphogens(self):
        """Apply morphogen inhibition effects based on the inhibition matrix."""
        # Iterate over each morphogen pair to apply inhibition
        for i in range(self.num_morphogens):
            for j in range(self.num_morphogens):
                if i != j and self.inhibition_matrix[i, j] > 0:
                    # Reduce morphogen `i` by the inhibition effect of morphogen `j`
                    self._M[i] *= 1 - self.inhibition_matrix[i, j] * self._M[j]

    def get_neighbors(self, x, y):
        """Get all neighboring positions around the given coordinates."""
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        # Update wrapping to use separate dimensions
        return [((x + dx) % self.size_x, (y + dy) % self.size_y) for dx, dy in offsets]

    def divide_cells(self):
        """Optimized vectorized cell division using matrix operations."""
        occupied = (self._progenitors != 0) | (self._neurons != 0)
        empty = ~occupied

        # Identify eligible progenitor cells for division
        eligible_mask = (self._division_morphogen > self.division_threshold) & (
            self._progenitors != 0
        )
        eligible_positions = np.argwhere(eligible_mask)

        if eligible_positions.size == 0:
            return

        # Define neighbor offsets
        offsets = np.array(
            [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        )

        # Precompute all neighbor positions and wrap around grid edges
        neighbors = (eligible_positions[:, None, :] + offsets) % [
            self.size_x,
            self.size_y,
        ]

        # Extract morphogen concentrations and empty status for neighbors
        neighbor_morphogens = self._division_morphogen[
            neighbors[..., 0], neighbors[..., 1]
        ]
        neighbor_empty = empty[neighbors[..., 0], neighbors[..., 1]]

        # Mask out non-empty positions
        neighbor_morphogens[~neighbor_empty] = -np.inf

        # Find the best neighbor for each eligible progenitor
        best_indices = np.argmax(neighbor_morphogens, axis=1)
        best_neighbors = neighbors[np.arange(len(eligible_positions)), best_indices]

        # Filter positions where a valid neighbor was found
        valid_divisions = neighbor_morphogens.max(axis=1) > -np.inf
        new_cells = best_neighbors[valid_divisions]

        # Update the grid and add new progenitor cells
        for new_x, new_y in new_cells:
            self.add_cell((new_x, new_y), cell_type="progenitor")
            empty[new_x, new_y] = False

    def secrete_morphogens(self):
        """Vectorized secretion of morphogens."""
        if not self._progenitors.any() and not self._neurons.any():
            return

        # Boolean masks for progenitors and neurons
        progenitor_mask = self._progenitors != 0
        neuron_mask = self._neurons != 0

        # Vectorized secretion for each morphogen
        for i in range(self.num_morphogens):
            secretion = (
                progenitor_mask * self.progenitor_secretion_rates[i]
                + neuron_mask * self.neuron_secretion_rates[i]
            )
            self._M[i] = np.minimum(1.0, self._M[i] + secretion)

    def differentiate_cells(self):
        """Vectorized cell differentiation based on the differentiation morphogen concentration."""
        if not self._progenitors.any():
            return

        # Create a mask for cells that meet the differentiation criteria
        differentiation_mask = (
            self._differentiation_morphogen > self.cell_differentiation_threshold
        ) & (self._progenitors != 0)

        # Identify positions where differentiation occurs
        differentiation_positions = np.argwhere(differentiation_mask)

        for x, y in differentiation_positions:
            cell_id = self._progenitors[x, y]
            self._progenitors[x, y] = 0
            self._neurons[x, y] = cell_id

            # Update the cell data and initialize axon
            self._axons[cell_id] = [self._cell_positions[cell_id]]

    def initialize_weight(self, source_id, target_id):
        """Initialize weight between two neurons based on morphogen concentration and distance."""
        x, y = self._cell_positions[target_id]
        source_x, source_y = self._cell_positions[source_id]

        # Calculate distance-based scaling
        distance = np.linalg.norm(np.array([source_x, source_y]) - np.array([x, y]))
        weight = max(0.01, self._weight_adjustment_morphogen[x, y] / (1 + distance))

        # Update weight in the connections matrix
        self.neuron_connections[source_id - 1, target_id - 1] = weight

    def update_weight(self, source_id, target_id):
        """Update weight between neurons using competitive scaling."""
        x, y = self._cell_positions[target_id]
        neighbors = self.get_neighbors(x, y)

        # Compute total morphogen concentration in the neighborhood
        total_morphogen = sum(
            self._weight_adjustment_morphogen[nx, ny] for nx, ny in neighbors
        )

        if total_morphogen > 0:
            new_weight = max(
                0.01, self._weight_adjustment_morphogen[x, y] / total_morphogen
            )
            self.neuron_connections[source_id - 1, target_id - 1] = new_weight

    def adjust_all_weights(self):
        """Vectorized homeostatic adjustment of all weights in the network.
        Weights will tend towards the local morphogen concentration, scaled by the adjustment rate."""
        MIN_WEIGHT = 0.01

        # Get non-zero indices once
        source_indices, target_indices = self.neuron_connections.nonzero()
        if len(source_indices) == 0:
            return

        # Get all target positions at once
        target_positions = np.array(
            [self._cell_positions[tid + 1] for tid in target_indices]
        )

        # Get morphogen values for all target positions at once
        target_morphogens = self._weight_adjustment_morphogen[
            target_positions[:, 0], target_positions[:, 1]
        ]

        # Get current weights
        current_weights = (
            self.neuron_connections[source_indices, target_indices].toarray().flatten()
        )

        # Calculate adjustments - weights will move towards the morphogen concentration
        # scaled by weight_adjustment_target (as maximum) and weight_adjustment_rate (as speed)
        target_weights = target_morphogens * self.weight_adjustment_target
        adjustments = self.weight_adjustment_rate * (target_weights - current_weights)

        # Update weights with new adjustments
        new_weights = np.clip(current_weights + adjustments, MIN_WEIGHT, 1.0)

        # Update weights one by one in the sparse matrix
        for i in range(len(source_indices)):
            self.neuron_connections[source_indices[i], target_indices[i]] = new_weights[
                i
            ]

    def grow_axon(self, cell_id, morphogen_concentration):
        """Grow the axon of a neuron cell using precomputed morphogen concentrations."""
        cell_pos = self._cell_positions[cell_id]

        # Check if the axon has reached its maximum length
        if len(self._axons[cell_id]) >= self.max_axon_length:
            return

        current_tip = self._axons[cell_id][-1] if self._axons[cell_id] else cell_pos

        # Get neighboring positions of the current axon tip
        x, y = current_tip
        neighbors = self.get_neighbors(x, y)

        # Filter out positions already part of the axon
        available_positions = [
            pos for pos in neighbors if pos not in self._axons[cell_id]
        ]

        if available_positions:
            # Get morphogen concentrations at available positions
            concentrations = [
                morphogen_concentration[pos] for pos in available_positions
            ]

            # Filter positions based on the axon growth threshold
            valid_positions = [
                (pos, conc)
                for pos, conc in zip(available_positions, concentrations)
                if conc >= self.axon_growth_threshold
            ]

            if valid_positions:  # Only proceed if there are valid positions
                # Find the position with the highest morphogen concentration
                max_pos, _ = max(valid_positions, key=lambda x: x[1])

                # Add the new position to the axon
                if not self._axons[cell_id]:
                    self._axons[cell_id] = [cell_pos, max_pos]
                else:
                    self._axons[cell_id].append(max_pos)

    def connect_neurons(self, source_id, target_pos):
        """Connect neurons if conditions are met and initialize weight."""
        target_id = self._neurons[target_pos]

        if target_id in self._cell_positions:
            if self._axon_connect_morphogen[target_pos] >= self.axon_connect_threshold:
                # Update weight initialization
                self.initialize_weight(source_id, target_id)

                # Reset the axon to just the cell position
                self._axons[source_id] = [self._cell_positions[source_id]]
                return True
        return False

    def grow_axons(self):
        """Optimized axon growth and connection for all neurons."""
        if not self._neurons.any():
            return

        morphogen_concentration = self._axon_guidance_morphogen

        for cell_id in self.get_neuron_ids():
            if len(self._axons[cell_id]) > 1:  # Axon exists and has grown
                axon_tip = self._axons[cell_id][-1]
                if self.connect_neurons(cell_id, axon_tip):
                    # Update weight after successful connection
                    target_id = self._neurons[axon_tip]
                    self.update_weight(cell_id, target_id)
                    continue

            self.grow_axon(cell_id, morphogen_concentration)

    def add_listener(self, listener):
        """Add a listener that will be notified after each step."""
        self.listeners.append(listener)

    def step(self):
        """Perform one step of the simulation."""
        self.secrete_morphogens()
        self.inhibit_morphogens()
        self.diffuse()
        self.divide_cells()
        self.differentiate_cells()
        self.grow_axons()
        # Homeostatic adjustment of all weights
        # self.adjust_all_weights()

        # Notify all listeners
        for listener in self.listeners:
            listener.on_step()

    def final_step(self):
        """Perform the final step of the simulation."""
        if self.self_connect_isolated_neurons_fraction > 0:
            self.self_connect_fraction_of_no_input_neurons()

    def no_input_neurons(self):
        """Get the IDs of all neurons with no input connections."""
        # Find columns with no non-zero values (no inputs)
        # and convert to 1-based cell IDs
        return [
            i + 1
            for i in range(self.neuron_connections.shape[1])
            if self.neuron_connections[:, i].nnz == 0 and self.is_neuron(i + 1)
        ]

    def self_connect_fraction_of_no_input_neurons(self):
        """Self connect a fraction of "no input" neurons."""
        no_input_ids = self.no_input_neurons()
        target_count = int(
            len(no_input_ids) * self.self_connect_isolated_neurons_fraction
        )

        # Use the seeded RNG to randomly sort the neurons
        selected_ids = self.rng.sample(no_input_ids, len(no_input_ids))

        # Take only the fraction we need
        for cell_id in selected_ids[:target_count]:
            self.connect_neurons(cell_id, self._cell_positions[cell_id])

    def get_morphogen_sum(self, morphogen_index):
        """Get the total sum of a specific morphogen."""
        return np.sum(self._M[morphogen_index])

    def get_morphogen_array(self, morphogen_index):
        """Get the concentration data for a specific morphogen as a numpy array."""
        return self._M[morphogen_index]

    def get_cell(self, cell_id):
        """Get data for a specific cell."""
        return self._cell_positions.get(cell_id)

    def get_axon(self, cell_id):
        """Get the axon for a specific cell."""
        return self._axons.get(cell_id)

    def is_neuron(self, cell_id):
        """Check if a cell is a neuron."""
        return cell_id in np.unique(self._neurons[self._neurons > 0])

    def is_progenitor(self, cell_id):
        """Check if a cell is a progenitor."""
        return cell_id in np.unique(self._progenitors[self._progenitors > 0])

    def get_neuron_ids(self):
        """Get the IDs of all neurons."""
        return np.unique(self._neurons[self._neurons > 0])

    def get_progenitor_ids(self):
        """Get the IDs of all progenitors."""
        return np.unique(self._progenitors[self._progenitors > 0])

    def get_cell_ids(self):
        """Get the IDs of all cells."""
        neuron_ids = self._neurons[self._neurons > 0]
        progenitor_ids = self._progenitors[self._progenitors > 0]
        return np.unique(np.concatenate([neuron_ids, progenitor_ids]))

    def neuron_count(self):
        """Get the number of neurons."""
        return np.sum(self._neurons > 0)

    def progenitor_count(self):
        """Get the number of progenitors."""
        return np.sum(self._progenitors > 0)

    def cell_count(self):
        """Get the total number of cells."""
        return np.sum(self._neurons > 0) + np.sum(self._progenitors > 0)

    def get_cell_position(self, cell_id):
        """Get the position of a specific cell."""
        neuron_pos = np.where(self._neurons == cell_id)
        if neuron_pos[0].size > 0:
            return (neuron_pos[0][0], neuron_pos[1][0])

        prog_pos = np.where(self._progenitors == cell_id)
        if prog_pos[0].size > 0:
            return (prog_pos[0][0], prog_pos[1][0])

        return None

    def run_simulation(self, verbose=True, display_weights=False):
        """Run simulation for specified number of steps.

        Args:
            verbose (bool): Whether to print progress and create displays
            display_weights (bool): Whether to print connection weights

        Returns:
            Grid: The grid instance after simulation
        """
        self.add_cell((self.size_x // 2, self.size_y // 2), "progenitor")

        if verbose:
            from .morphogen_display import MorphogenDisplay
            from .neuron_graph_display import NeuronGraphDisplay

            # Create displays first
            morphogen_display = MorphogenDisplay(self, update_frequency=10)

        # Run simulation with displays updating each step
        import time

        start_time = time.time()
        for i in range(self.max_growth_steps):
            self.step()  # This will automatically update displays via listeners
            if i % 100 == 0:
                end_time = time.time()
                elapsed_ms = (end_time - start_time) * 1000
                if verbose:
                    print(
                        f"Step {i}; cells: {self.cell_count()}; elapsed: {elapsed_ms:.2f} ms"
                    )
        self.final_step()
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000

        if verbose:
            print(f"Simulation completed in {elapsed_ms:.2f} ms")
            source_indices, target_indices = self.neuron_connections.nonzero()
            weights = (
                self.neuron_connections[source_indices, target_indices]
                .toarray()
                .flatten()
            )
            if display_weights:
                weight_strings = [
                    f"({s},{t})={w:.4f}"
                    for s, t, w in zip(source_indices + 1, target_indices + 1, weights)
                ]
                print("Weights:", " ".join(weight_strings))
            # Get diagonal elements (self-connections) and count non-zero ones
            self_connections = self.neuron_connections.diagonal()
            print(
                f"Number of neurons connected to themselves: {(self_connections > 0).sum()}"
            )
            print(f"Sum of weights: {self.neuron_connections.sum()}")
            print(f"Neuron connections number: {self.neuron_connections.nnz}")
            morphogen_display.show(block=False)
            neuron_display = NeuronGraphDisplay(self)
            neuron_display.show(block=True)

        return self
