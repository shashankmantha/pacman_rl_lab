"""
The Pac-Man world used in the lab. You do not need to edit this file.

    env = PacmanEnv("tricky")
    state = env.reset()                      # State(pacman, ghosts, food)
    env.legal_actions(state)                 # e.g. ['North', 'East', 'West']; [] once the game is over
    next_state, reward, done = env.step("North")

Geometry helpers that are handy when you write features in Part 5:

    env.next_position(pos, action)           # the square one move away
    env.open_neighbors(pos)                  # the non-wall squares next to pos
    env.maze_distance(a, b)                  # shortest path length through the maze
    env.closest_food_distance(pos, food)     # maze distance to the nearest pellet
"""
import random
from collections import deque, namedtuple, Counter

ACTIONS = ("North", "South", "East", "West")
DIRS = {"North": (0, -1), "South": (0, 1), "East": (1, 0), "West": (-1, 0)}

# Rewards: the same scoring rules as the Berkeley Pac-Man projects
STEP_PENALTY = -1     # every move costs 1 point
FOOD_REWARD = 10      # eat a pellet
WIN_REWARD = 500      # eat the last pellet
LOSE_REWARD = -500    # touch a ghost

# A state is everything that can change during a game.
#   pacman: (x, y)    ghosts: tuple of (x, y), one per moving ghost    food: frozenset of (x, y)
State = namedtuple("State", ["pacman", "ghosts", "food"])

# % wall   . pellet   P Pac-Man   G ghost (moves at random)   X sleeping ghost (never moves)
LAYOUTS = {
    # Part 2: a hallway with one pellet and no ghost
    "hallway": """
%%%%%%%%%
%P     .%
%%%%%%%%%
""",
    # Parts 1 and 3: the board from the "Q-Learning Pacman - Tricky" lecture demo
    "tricky": """
%%%%%%%
%%%.%%%
%.   .%
% %G% %
%  P  %
%%%%%%%
""",
    # Part 4: a row of sleeping ghosts is a cliff
    "ghostCliff": """
%%%%%%%%%%
%        %
%        %
%P      .%
%XXXXXXXX%
%%%%%%%%%%
""",
    # Part 5: two ghosts and 37 pellets, far too many states for a table
    "twoGhosts": """
%%%%%%%%%%%%%
%.....%.....%
%.%%%.%.%%%.%
%.G.......G.%
%.%%%.%.%%%.%
%.....P.....%
%%%%%%%%%%%%%
""",
}


class PacmanEnv:
    """A small Pac-Man world.

    env.reset()                  -> start state
    env.legal_actions(state)     -> list of actions ([] if the game is over in that state)
    env.step(action)             -> (next_state, reward, done)
    env.is_terminal(state)       -> True after a win or a loss

    Moving ghosts take a uniformly random legal step after each Pac-Man move.
    Sleeping ghosts (X) never move. A game also ends after max_steps moves.
    """

    def __init__(self, layout="tricky", max_steps=300, seed=None):
        text = LAYOUTS.get(layout, layout)
        rows = text.strip("\n").split("\n")
        self.name = layout if layout in LAYOUTS else "custom"
        self.height, self.width = len(rows), max(len(r) for r in rows)
        self.walls, food, ghosts, sleepers = set(), set(), [], set()
        self.start_pacman = None
        for y, row in enumerate(rows):
            for x, ch in enumerate(row.ljust(self.width, "%")):
                if ch == "%":
                    self.walls.add((x, y))
                elif ch == ".":
                    food.add((x, y))
                elif ch == "P":
                    self.start_pacman = (x, y)
                elif ch == "G":
                    ghosts.append((x, y))
                elif ch == "X":
                    sleepers.add((x, y))
        if self.start_pacman is None:
            raise ValueError("The layout needs exactly one P (Pac-Man)")
        self.start_state = State(self.start_pacman, tuple(ghosts), frozenset(food))
        self.sleepers = frozenset(sleepers)
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self._dist_cache = {}
        self.state, self.steps, self.score = self.start_state, 0, 0

    # ----- geometry helpers (useful for writing features) -----
    def next_position(self, pos, action):
        dx, dy = DIRS[action]
        return (pos[0] + dx, pos[1] + dy)

    def open_neighbors(self, pos):
        """Non-wall squares one step from pos."""
        return [self.next_position(pos, a) for a in ACTIONS if self.next_position(pos, a) not in self.walls]

    def open_squares(self):
        return [(x, y) for y in range(self.height) for x in range(self.width) if (x, y) not in self.walls]

    def maze_distance(self, a, b):
        """Shortest path length between two squares (breadth-first search, cached)."""
        if a == b:
            return 0
        if a not in self._dist_cache:
            dist, frontier = {a: 0}, deque([a])
            while frontier:
                p = frontier.popleft()
                for q in self.open_neighbors(p):
                    if q not in dist and q not in self.sleepers:
                        dist[q] = dist[p] + 1
                        frontier.append(q)
            self._dist_cache[a] = dist
        return self._dist_cache[a].get(b, self.width * self.height)

    def closest_food_distance(self, pos, food):
        return min((self.maze_distance(pos, f) for f in food), default=0)

    # ----- MDP interface -----
    def is_win(self, state):
        return len(state.food) == 0

    def is_lose(self, state):
        return state.pacman in state.ghosts or state.pacman in self.sleepers

    def is_terminal(self, state):
        return self.is_win(state) or self.is_lose(state)

    def legal_actions(self, state):
        if self.is_terminal(state):
            return []
        return [a for a in ACTIONS if self.next_position(state.pacman, a) not in self.walls]

    def reset(self):
        self.state, self.steps, self.score = self.start_state, 0, 0
        return self.state

    def step(self, action):
        s = self.state
        if action not in self.legal_actions(s):
            hint = " (get_action returned None, but Pac-Man has legal moves)" if action is None else ""
            raise ValueError(f"Illegal action {action!r} at {s.pacman}{hint}. Legal: {self.legal_actions(s)}")
        reward = STEP_PENALTY
        pac = self.next_position(s.pacman, action)
        food = s.food
        if pac in food:
            food = food - {pac}
            reward += FOOD_REWARD
            if not food:
                reward += WIN_REWARD
        ghosts = s.ghosts
        if food and pac not in ghosts and pac not in self.sleepers:      # ghosts move after Pac-Man
            ghosts = tuple(self.rng.choice(self.open_neighbors(g)) for g in ghosts)
        if food and (pac in ghosts or pac in self.sleepers):
            reward += LOSE_REWARD
        self.state = State(pac, ghosts, food)
        self.steps += 1
        self.score += reward
        done = self.is_terminal(self.state) or self.steps >= self.max_steps
        return self.state, reward, done

    def outcome(self):
        if self.is_win(self.state):
            return "win"
        if self.is_lose(self.state):
            return "lose"
        return "timeout"


# ---------------------------------------------------------------------------
# Simple agents used by the demos
# ---------------------------------------------------------------------------

class RandomAgent:
    """Picks a legal action uniformly at random and never learns (Part 1)."""
    epsilon = 1.0

    def __init__(self, env):
        self.env = env

    def get_action(self, state):
        acts = self.env.legal_actions(state)
        return random.choice(acts) if acts else None

    def update(self, state, action, next_state, reward):
        pass


# ---------------------------------------------------------------------------
# Running games without a window (used by the checks and by the demos)
# ---------------------------------------------------------------------------

def run_episode(agent, env, learn=True):
    """Play one game. If learn=True the agent's update() is called after every move.
    Returns (score, outcome) where outcome is 'win', 'lose' or 'timeout'."""
    state = env.reset()
    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        if learn:
            agent.update(state, action, next_state, reward)
        state = next_state
        if done:
            break
    if hasattr(agent, "end_episode"):
        agent.end_episode()
    return env.score, env.outcome()


def train(agent, env, episodes, verbose=0):
    """Train for a number of games. Returns the list of scores."""
    scores, outcomes = [], []
    for i in range(1, episodes + 1):
        sc, out = run_episode(agent, env, learn=True)
        scores.append(sc)
        outcomes.append(out)
        if verbose and i % verbose == 0:
            recent = scores[-verbose:]
            print(f"  game {i:5d} | avg score (last {verbose}) {sum(recent) / len(recent):8.1f} | "
                  f"wins {outcomes[-verbose:].count('win')}/{verbose}")
    return scores


class greedy_mode:
    """with greedy_mode(agent): ...   turns exploration off, and back on afterwards."""
    def __init__(self, agent):
        self.agent = agent

    def __enter__(self):
        self.saved = getattr(self.agent, "epsilon", 0.0)
        self.agent.epsilon = 0.0
        if hasattr(self.agent, "exploring"):
            self.agent.exploring = False
        return self.agent

    def __exit__(self, *exc):
        self.agent.epsilon = self.saved
        if hasattr(self.agent, "exploring"):
            self.agent.exploring = True
        return False


def evaluate(agent, env, games=100):
    """Play greedily (epsilon = 0, no learning). Returns (win_rate, average_score)."""
    wins, total = 0, 0.0
    with greedy_mode(agent):
        for _ in range(games):
            sc, out = run_episode(agent, env, learn=False)
            wins += out == "win"
            total += sc
    return wins / games, total / games


def best_action_fixed(agent, state, actions):
    """argmax of Q without random tie-breaking (first best in order). Used for drawing."""
    return max(actions, key=lambda a: agent.get_q_value(state, a))


def greedy_path(agent, env, max_len=40):
    """Squares visited by the greedy policy from the start (ghosts frozen in place)."""
    s = env.start_state
    path = [s.pacman]
    for _ in range(max_len):
        acts = env.legal_actions(s)
        if not acts:
            break
        a = best_action_fixed(agent, s, acts)
        pac = env.next_position(s.pacman, a)
        s = State(pac, s.ghosts, s.food - {pac})
        path.append(pac)
    return path


def exact_hallway_utilities(env, gamma):
    """True U(s) for the policy 'always East' in the deterministic hallway (no ghosts)."""
    U = {}
    saved = (env.state, env.steps, env.score)
    for x in range(1, env.width - 1):
        s = State((x, env.start_pacman[1]), (), env.start_state.food)
        if s.pacman in env.walls or s.pacman in s.food or env.is_terminal(s):
            continue
        env.state, env.steps, env.score = s, 0, 0
        G, k, done = 0.0, 0, False
        while not done:
            _, r, done = env.step("East")
            G += gamma ** k * r
            k += 1
        U[s] = G
    env.state, env.steps, env.score = saved
    return U


def death_places(agent, env, games=100):
    """Play greedy games. Returns (Counter of squares where Pac-Man died, Counter of how)."""
    places, kinds = Counter(), Counter()
    with greedy_mode(agent):
        for _ in range(games):
            state = env.reset()
            while True:
                nxt, _, done = env.step(agent.get_action(state))
                if done:
                    break
                state = nxt
            if env.is_lose(env.state):
                places[env.state.pacman] += 1
                near = sum(1 for g in state.ghosts if env.maze_distance(state.pacman, g) <= 3)
                kinds["trapped: 2 ghosts within 3 steps" if near >= 2 else "caught by a single ghost"] += 1
    return places, kinds
