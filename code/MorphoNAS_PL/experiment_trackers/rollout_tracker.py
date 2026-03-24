class RolloutTracker:
    """
    Tracks rollout data per generation by wrapping the progress callback.

    This allows visualization of how learning behavior evolves across
    generations, answering questions like:
    - Does evolution improve learning ability?
    - At what generation does learning emerge?
    - How does rollout improvement change over evolution?
    """

    def __init__(self, experiment_runner):
        """
        Initialize rollout tracker.

        Args:
            experiment_runner: The ExperimentRunner instance to track
        """
        self.runner = experiment_runner

    def wrap_callback(self, original_callback):
        """
        Returns enhanced callback that captures rollout data per generation.

        Args:
            original_callback: The original progress callback to wrap

        Returns:
            Enhanced callback function
        """

        def enhanced_callback(optimizer, generation, best_fitness, avg_fitness):
            # Call original callback first
            should_continue = original_callback(optimizer, generation, best_fitness, avg_fitness)

            # Retrieve rollout data captured during multiprocessing evaluation
            rollout_data = self.runner.get_rollout_data(optimizer.best_solution)

            # Add to current generation stats (last item in array)
            if self.runner.generation_stats and rollout_data:
                self.runner.generation_stats[-1]['rollout_stats'] = rollout_data

            return should_continue

        return enhanced_callback

    def __repr__(self):
        return f"RolloutTracker(runner={self.runner.experiment_name})"
