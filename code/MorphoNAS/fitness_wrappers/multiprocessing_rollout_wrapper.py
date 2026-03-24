class MultiprocessingRolloutWrapper:
    def __init__(self, base_fitness_function):
        self.base_fitness_function = base_fitness_function

    def __call__(self, genome):
        result = self.base_fitness_function(genome)

        if isinstance(result, tuple) and len(result) == 2:
            fitness, rollout_data = result
        else:
            fitness = result
            rollout_data = None

        return (fitness, rollout_data)

    def evaluate(self, genome):
        return self.__call__(genome)

    def __getattr__(self, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(
                f"{self.__class__.__name__} object has no attribute '{name}'"
            )
        return getattr(self.base_fitness_function, name)

    def __repr__(self):
        base_name = type(self.base_fitness_function).__name__
        return f"MultiprocessingRolloutWrapper(base={base_name})"
