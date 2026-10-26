"""Several DrivingEnvs on one map, stepped together (the 'vectorized environment' pattern).

The cars are ghosts: they share the world and one route field but never touch each other.
A car whose episode ends resets by itself, so the others keep driving.
"""
from car_sim.driving_env import DrivingEnv
from car_sim.route import RouteField, StraightField


class VecDrivingEnv:
    def __init__(self, world, num_cars, use_route=True):
        self.world = world
        self.route = RouteField() if use_route else StraightField()
        self.envs = [DrivingEnv(world, use_route=use_route, route=self.route)
                     for _ in range(num_cars)]
        self.route_ok = False
        self.states = []  # what each car sees now: the input for the next actions

    @property
    def cars(self):
        return [env.car for env in self.envs]

    def check_route(self):
        """Refresh the shared route once; True if the start has a route to the finish."""
        self.route_ok = self.route.ensure(self.world)
        return self.route_ok

    def reset(self):
        self.states = [env.reset() for env in self.envs]
        self.check_route()
        return list(self.states)

    def step(self, actions):
        """Step every car with its action.

        Returns (next_states, rewards, dones, finished). A finished car's next state is its
        terminal state (that is what the replay buffer must store), while self.states holds
        the state of its fresh episode. finished lists {"car", "outcome", "score"} per ended
        episode, score being the steps it lasted.
        """
        next_states, rewards, dones, finished = [], [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            state, reward, done = env.step(action)
            next_states.append(state)
            rewards.append(reward)
            dones.append(done)
            if done:
                finished.append({"car": i, "outcome": env.outcome, "score": env.steps})
                self.states[i] = env.reset()
            else:
                self.states[i] = state
        return next_states, rewards, dones, finished
