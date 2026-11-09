"""
===============================================================================
PART 5   Generalization: approximate Q-learning                   AIMA 4e §22.4
Questions Q9 and Q10 (28 points: 16 code + 12 written)            about 22 minutes

    Write your code:   ApproximateQAgent (Q9) and my_features (Q10),
                       then ANSWER_Q9 and ANSWER_Q10
    Check it:          python autograder.py -p 5
    Watch it:          python part5_approximate.py table      (for Q9)
                       python part5_approximate.py deaths     (for Q10 a)
                       python part5_approximate.py compare    (for Q10 b, c)
    This part builds on your QLearningAgent from Part 3.
===============================================================================

NOTES
-----
WHY THE TABLE BREAKS. A table learns every state separately. The 'twoGhosts'
board has 40 squares for Pac-Man, 40 for each ghost and 37 pellets:

        40 × 40² × 2^37  ≈  9 × 10^15 states

Pac-Man will never see most of them even once, and what it learns about a ghost
on the left says nothing about the same situation on the right.

FEATURES AND A LINEAR Q-FUNCTION (§22.4.1, §22.4.2). Describe each state-action
pair by a few numbers, features f_1(s,a) ... f_n(s,a), and learn one weight each:

        Q̂_w(s,a)  =  w_1·f_1(s,a) + w_2·f_2(s,a) + ... + w_n·f_n(s,a)  =  Σ_i w_i·f_i(s,a)

Four weights instead of 10^15 table entries, and any two situations with the
same features get the same value, whether Pac-Man has seen them or not.

LEARNING THE WEIGHTS. We want Q̂_w(s,a) to match the sample
y = r + γ·max_a' Q̂_w(s',a'). Take the squared error E = ½·(y − Q̂_w(s,a))² and
step every weight downhill, treating y as a fixed target:

        w_i  ←  w_i − α · ∂E/∂w_i  =  w_i + α · [ y − Q̂_w(s,a) ] · ∂Q̂_w(s,a)/∂w_i

Because Q̂ is linear, ∂Q̂/∂w_i = f_i(s,a). That gives the APPROXIMATE
Q-LEARNING UPDATE:

        δ    =  [ r + γ · max_a' Q̂_w(s',a') ]  −  Q̂_w(s,a)
        w_i  ←  w_i + α · δ · f_i(s,a)          for every feature i

If a move went better than predicted (δ > 0), every feature that was active
gets more credit, in proportion to how active it was. Tabular Q-learning is the
special case with one on/off feature per (s, a) pair.

DEEP RL (§22.4.3). Replace the linear function with a neural network that reads
the screen pixels and you have DQN, the method behind the Breakout demo. The
network learns its own features instead of being handed them.

REWARD SHAPING (§22.4.4) is the other common way to give an agent hints: add
F(s,a,s') = γ·Φ(s') − Φ(s) to every reward, for some potential function Φ (for
example Φ(s) = −distance to the nearest pellet). Along any path the extra terms
telescope away, so the optimal policy does not change (Ng, Harada and Russell,
1999), but learning can be much faster.

THE FEATURES PAC-MAN GETS (simple_features, below). They look one move ahead.
All values are divided by 10 so the weight updates stay small.
    bias                  always 1 (lets Q̂ have a baseline)
    ghosts-1-step-away    number of ghosts on or next to the square the move leads to
    eats-food             1 if the move eats a pellet and no ghost is that close
    closest-food          maze distance from that square to the nearest pellet,
                          divided by the board size

WHAT YOU WILL SEE
    table    A table agent and your approximate agent learn 'twoGhosts' side by
             side. With the value overlay on, watch the red "danger zone" move
             with each ghost: the approximate agent knows about squares it has
             never visited.
    deaths   Where and how the trained agent still dies (red circles).
    compare  The baseline features against yours, 3 random seeds each.
"""
from collections import defaultdict

from pacman import PacmanEnv, Lane, run_phase, finish, death_places, outcome_text, setup, run
from part3_qlearning import QLearningAgent


def simple_features(env, state, action):
    nxt = env.next_position(state.pacman, action)                 # where this move leads
    dangers = list(state.ghosts) + list(env.sleepers)
    near = sum(1 for g in dangers if g == nxt or g in env.open_neighbors(nxt))
    f = {"bias": 1.0, "ghosts-1-step-away": float(near)}
    if near == 0 and nxt in state.food:
        f["eats-food"] = 1.0
    f["closest-food"] = env.closest_food_distance(nxt, state.food) / (env.width * env.height)
    return {k: v / 10.0 for k, v in f.items()}


# =============================================================================
# Q9. Approximate Q-learning (10 points)
#
#     get_q_value(s, a) = Σ_i w_i · f_i(s, a)
#     update:  δ = [ r + γ · max_a' Q̂(s', a') ] − Q̂(s, a);   w_i ← w_i + α · δ · f_i(s, a)
#
# Features: self.feature_fn(self.env, state, action) returns a dict
# {feature name: value}. Weights: self.weights, a defaultdict(float) keyed by
# feature name. Use self.get_value(next_state) for the max. Everything else
# (get_value, best_action, get_action) is inherited from your Part 3 agent and
# works unchanged, because it calls get_q_value.
# =============================================================================
class ApproximateQAgent(QLearningAgent):

    def __init__(self, env, feature_fn=None, alpha=0.2, epsilon=0.05, gamma=0.8):
        action_fn = env.legal_actions if env is not None else (lambda s: [])
        super().__init__(action_fn, alpha=alpha, epsilon=epsilon, gamma=gamma)
        self.env = env
        self.feature_fn = feature_fn or simple_features
        self.weights = defaultdict(float)    # feature name -> weight

    def get_q_value(self, state, action):
        # YOUR CODE HERE
        raise NotImplementedError

    def update(self, state, action, next_state, reward):
        # YOUR CODE HERE
        raise NotImplementedError


# =============================================================================
# Q10. Engineer a better Pac-Man (6 points)
#
# Your approximate agent still loses some games on 'twoGhosts'. First run
#     python part5_approximate.py deaths
# to see where and how it dies. Then give it a feature that captures what it is
# missing: start from simple_features and add AT LEAST ONE new feature (keep the
# four originals; divide new values by 10 like they do). Ideas, which you do not
# have to use: the number of ghosts within 2 or 3 maze steps of the square the
# move leads to; the maze distance to the nearest ghost; whether that square is a
# dead end; how many ways out of that square are not next to a ghost.
#
# Helpers: env.next_position(pos, action), env.open_neighbors(pos),
# env.maze_distance(a, b), env.closest_food_distance(pos, food), state.ghosts,
# state.food. Then run: python part5_approximate.py compare
#
# A feature that does not help can still earn full written credit if you explain why.
# =============================================================================
def my_features(env, state, action):
    f = simple_features(env, state, action)
    nxt = env.next_position(state.pacman, action)
    # YOUR CODE HERE
    raise NotImplementedError
    return f


# =============================================================================
# Q9 written (4 points). Run:  python part5_approximate.py table
#
# (a) The table agent learned thousands of Q-table entries and still lost; the
#     approximate agent has 4 weights and wins most games after 100. Explain why,
#     in terms of what each agent can do with a state it has NEVER SEEN.
# (b) Look at your learned weights. Explain the sign and rough size of each one in
#     Pac-Man terms. Why is the closest-food weight negative?
# =============================================================================
ANSWER_Q9 = """
(a)

(b)

"""

# =============================================================================
# Q10 written (8 points). Run:  python part5_approximate.py deaths   and   compare
#
# (a) What did the death report show about how Pac-Man loses? Define your new
#     feature precisely (what it measures, its range) and say which kind of death
#     it is meant to prevent.
# (b) Report the comparison table (baseline vs. yours: win rate per seed and mean
#     score). Did your feature help? Is the difference larger than the
#     seed-to-seed variation?
# (c) Report the learned weight of your new feature. Does its sign make sense?
#     Explain why the feature helped or did not (for example: it duplicates an
#     existing feature, it is badly scaled, or it still cannot tell apart the
#     situations that kill Pac-Man).
# (d) The Breakout DQN learned from raw pixels with no hand-made features. Give one
#     advantage and one disadvantage of hand-designed features compared with
#     features a deep network learns for itself.
# =============================================================================
ANSWER_Q10 = """
(a)

(b)

(c)

(d)

"""


# =============================================================================
# Demo (you do not need to change anything below)
# =============================================================================
def trained_approximate(args):
    env = PacmanEnv("twoGhosts", seed=0)
    lane = Lane("approximate Q-learning, simple_features", ApproximateQAgent(env, simple_features), env, seed=0)
    run_phase(lane, 100, args.display, watch=False,
              banner="Training the approximate agent: 100 games (fast-forward).")
    return lane


def table(args):
    env_t, env_a = PacmanEnv("twoGhosts", seed=0), PacmanEnv("twoGhosts", seed=0)
    tab = Lane("table: tabular Q-learning (Part 3)", QLearningAgent(env_t.legal_actions, 0.5, 0.1, 0.9), env_t, seed=0)
    apx = Lane("features: approximate Q-learning", ApproximateQAgent(env_a, simple_features), env_a, seed=0)
    run_phase([tab, apx], 2, args.display,
              banner="Q9: a Q-table vs. 4 weights on 'twoGhosts'. Watch the first 2 games of each.")
    run_phase([tab, apx], [498, 98], args.display, watch=False,
              banner="Q9: the table trains for 500 games, the approximate agent for only 100 (fast-forward).")
    win, avg = tab.evaluate(50)
    print(f"\nTable agent after 500 games:       wins {win:.0%}, average score {avg:7.1f}, "
          f"{len(tab.agent.q_values):,} Q-table entries")
    win, avg = apx.evaluate(100)
    print(f"Approximate agent after 100 games: wins {win:.0%}, average score {avg:7.1f}")
    print("Learned weights:")
    for k, v in apx.agent.weights.items():
        print(f"  {k:22s} {v:9.1f}")
    tell = lambda l: print(f"  Game {l.games_in_phase}: {outcome_text(l.outcomes[-1])} Score: {l.scores[-1]}")
    run_phase(apx, 2, args.display, learn=False, greedy=True, on_game=tell,
              banner="Q9: the approximate agent plays greedily. The red zones move with the ghosts.")
    return apx


def deaths(args, trained=None):
    trained = trained or trained_approximate(args)
    agent = trained.agent
    env = PacmanEnv("twoGhosts", seed=100)
    lane = Lane("approximate Q-learning, simple_features", agent, env, seed=100, overlay=None)
    with lane.own_rng():
        places, kinds = death_places(agent, env, 100)
    print(f"\nDeath report: lost {sum(places.values())} of 100 greedy games.")
    for k, v in kinds.most_common():
        print(f"  {k:36s} {v}")
    lane.deaths, lane.view_state = places, env.reset()
    lane.scores, lane.outcomes = trained.scores, trained.outcomes      # keep its learning curve
    lane.message = f"lost {sum(places.values())} of 100 greedy games"
    if args.display:
        args.display.begin_phase([lane], "Q10: where Pac-Man died in 100 greedy games (bigger circle = more often).")
    return lane


def compare(args):
    results = {"baseline": [], "mine": []}
    for seed in (0, 1, 2):
        lanes = []
        for name, fn in (("baseline", simple_features), ("mine", my_features)):
            env = PacmanEnv("twoGhosts", seed=seed)
            title = f"simple_features, seed {seed}" if name == "baseline" else f"my_features (yours), seed {seed}"
            lanes.append(Lane(title, ApproximateQAgent(env, fn), env, seed=seed))
        run_phase(lanes, 100, args.display, watch=False,
                  banner=f"Q10: baseline vs. your features, seed {seed} of 0-2: 100 training games each.")
        for name, lane in zip(("baseline", "mine"), lanes):
            results[name].append(lane.evaluate(100))
        last = lanes[1]
    print()
    for name, res in results.items():
        wins = [w for w, _ in res]
        print(f"{name:10s} win rate per seed {', '.join(f'{w:.0%}' for w in wins):16s} "
              f"mean {sum(wins) / len(wins):.0%} | mean score {sum(a for _, a in res) / len(res):7.1f}")
    print("\nWeights learned with your features (seed 2):")
    for k, v in last.agent.weights.items():
        print(f"  {k:22s} {v:9.1f}")
    run_phase(last, 2, args.display, learn=False, greedy=True,
              banner="Q10: your features, playing greedily (seed 2).")
    return last


def main():
    args = setup("Part 5: approximate Q-learning on 'twoGhosts'.",
                 lambda p: p.add_argument("experiment", nargs="?", default="all",
                                          choices=["table", "deaths", "compare", "all"],
                                          help="table (Q9), deaths (Q10a), compare (Q10b,c), or all (default)"))
    shown = None
    if args.experiment in ("table", "all"):
        shown = table(args)
    if args.experiment in ("deaths", "all"):
        shown = deaths(args, shown)
        if args.display and args.experiment == "all":
            args.display.wait(5)      # time to look at the death map
    if args.experiment in ("compare", "all"):
        shown = compare(args)
    print("\nNext: answer ANSWER_Q9 and ANSWER_Q10, then run python autograder.py for your total.")
    finish(args.display, shown, "Part 5 done. Close the window (or press Q) to exit.")


if __name__ == "__main__":
    run(main)
