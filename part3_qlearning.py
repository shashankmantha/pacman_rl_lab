"""
===============================================================================
PART 3   Active reinforcement learning: Q-learning                AIMA 4e §22.3
Questions Q3 to Q6 (36 points: 30 code + 6 written)               about 28 minutes

    Write your code:   the QLearningAgent class below (Q3, Q4, Q5), then ANSWER_Q6
    Check it:          python autograder.py -p 3     (or -q q3, -q q4, -q q5, -q q6)
    Watch it:          python part3_qlearning.py     (works once Q3 to Q5 pass)
===============================================================================

NOTES
-----
Now Pac-Man chooses its own moves. Two new problems appear:

  1. What to learn. U(s) is not enough to act: choosing a move from U needs
     P(s'|s,a) to see where each move leads, and Pac-Man has no model. So we
     learn Q(s,a), the value of each ACTION in each state.
  2. How to act while learning. Always taking the move that looks best
     (exploitation) can lock the agent into a bad habit; it must also try
     things (exploration).

TD Q-LEARNING (§22.3.3). After each move (s, a, r, s'):

        Q(s,a)  ←  Q(s,a) + α · [ r + γ · max_a' Q(s',a')  −  Q(s,a) ]

    which is the same as a running average with the newest sample:

        sample  =  r + γ · max_a' Q(s',a')
        Q(s,a)  ←  (1 − α) · Q(s,a) + α · sample

  * Model-free: no P and no R, just one number per (s, a) pair in a dictionary.
  * Off-policy: the sample uses max_a', the value of acting GREEDILY next,
    whatever the agent actually does next. So it learns the optimal Q* even
    while it behaves partly at random.

ACTING FROM Q. For the legal actions A(s):

        V(s)  =  max_{a in A(s)} Q(s,a)        (0 if A(s) is empty: game over)
        π(s)  =  argmax_{a in A(s)} Q(s,a)

ε-GREEDY EXPLORATION:

        with probability ε      a random legal action (any of them, best included)
        with probability 1 − ε  argmax_a Q(s,a)

Q-learning converges to Q* if every (s, a) keeps being tried and α decays
suitably. A scheme that tries everything infinitely often AND becomes greedy in
the limit is called GLIE (greedy in the limit of infinite exploration). A fixed
ε > 0 explores forever, so it is not GLIE; ε_t = 1/t is.

WHAT YOU WILL SEE (the "Q-Learning Pacman - Tricky - Watch All" lecture demo)
    Phase 1  Watch All: 5 games with learning on. The Q-table starts empty, so
             Pac-Man walks into the ghost (Score: -505 ...).
    Phase 2  Silent training: 2,000 games at full speed. The chart shows the
             average score climbing as Pac-Man learns.
    Phase 3  The trained Pac-Man plays 3 games greedily (ε = 0, no learning).
    Value overlay: every square is coloured by V(s) for "Pac-Man on this
    square, ghost and pellets where they are now" (green good, red bad), with
    an arrow for the best move. Watch the red zone form around the ghost.
"""
import random
from collections import defaultdict

from pacman import PacmanEnv, Lane, run_phase, finish, outcome_text, setup, run


class QLearningAgent:
    """Tabular Q-learning. action_fn(state) gives the legal actions ([] when the game is over)."""

    def __init__(self, action_fn, alpha=0.5, epsilon=0.1, gamma=0.9):
        self.action_fn = action_fn
        self.alpha = alpha                  # learning rate α
        self.epsilon = epsilon              # exploration rate ε
        self.gamma = gamma                  # discount factor γ
        self.q_values = defaultdict(float)  # (state, action) -> Q-value; 0.0 if never updated

    def legal_actions(self, state):
        return self.action_fn(state)

    def get_q_value(self, state, action):
        return self.q_values[(state, action)]

    # =========================================================================
    # Q3. Values and the greedy action (8 points)
    #
    #     get_value(s)   = max over legal a of Q(s, a)     0.0 if no legal actions
    #     best_action(s) = argmax over legal a of Q(s, a)  None if no legal actions
    #
    # * If every legal Q-value is negative, V(s) is negative, not 0.
    # * If several actions tie for best, pick one of them AT RANDOM (random.choice).
    # * Read Q-values with self.get_q_value(state, action), never self.q_values
    #   directly: Part 5 replaces get_q_value and your code must still work there.
    # =========================================================================
    def get_value(self, state):
        # YOUR CODE HERE
        raise NotImplementedError

    def best_action(self, state):
        # YOUR CODE HERE
        raise NotImplementedError

    # =========================================================================
    # Q4. ε-greedy (6 points)
    #
    # With probability self.epsilon return a random legal action, chosen from ALL
    # legal actions; otherwise return self.best_action(state). Return None if there
    # are no legal actions. random.random() gives a float in [0, 1).
    # =========================================================================
    def get_action(self, state):
        # YOUR CODE HERE
        raise NotImplementedError

    # =========================================================================
    # Q5. The Q-learning update (12 points)
    #
    #     sample = r + γ · max_a' Q(s', a')
    #     Q(s,a) ← (1 − α) · Q(s,a) + α · sample
    #
    # max_a' Q(s', a') is exactly self.get_value(next_state), which is already 0
    # when the game is over. Change only Q(state, action).
    # =========================================================================
    def update(self, state, action, next_state, reward):
        # YOUR CODE HERE
        raise NotImplementedError


# =============================================================================
# Q6. Watch Pac-Man learn (10 points: 4 auto-graded + 6 written)
#
# Run python part3_qlearning.py and watch all three phases. The autograder gives
# the 4 points if your agent wins at least 80% of greedy games after 2,000
# training games. Then answer below, using the numbers from YOUR run.
#
# (a) Your first "Watch All" games (and the lecture's) end around -505. Use Q1 to
#     say what sequence of rewards that score is. Then explain why an agent whose
#     Q-table is all zeros walks into the ghost: what does best_action do when
#     every Q-value is 0?
# (b) The average score DURING training (chart, and the progress lines) stays
#     well below the greedy average printed after training, even at the end.
#     What causes the gap? Name the parameter responsible.
# (c) Using the counts printed at the end, compute an upper bound on the number
#     of states on this board (squares for Pac-Man × squares for the ghost ×
#     2^pellets) and multiply by up to 4 actions. Compare it with your number of
#     Q-table entries. Why is the table smaller than the bound, and why would
#     this approach struggle on a full-size Pac-Man board?
# =============================================================================
ANSWER_Q6 = """
(a)

(b)

(c)

"""


# =============================================================================
# Demo (you do not need to change anything below)
# =============================================================================
def main():
    args = setup("Part 3: watch Pac-Man learn with Q-learning on the 'tricky' board.",
                 lambda p: p.add_argument("--train", type=int, default=2000, help="silent training games"))
    env = PacmanEnv("tricky", seed=469)
    agent = QLearningAgent(env.legal_actions, alpha=0.5, epsilon=0.1, gamma=0.9)
    lane = Lane("Q-learning: alpha 0.5, epsilon 0.1, gamma 0.9", agent, env, seed=469)
    tell = lambda l: print(f"  Game {l.games_in_phase}: {outcome_text(l.outcomes[-1])} Score: {l.scores[-1]}")

    print("Phase 1 of 3: WATCH ALL (learning on, Q-table starts empty)")
    run_phase(lane, 5, args.display, on_game=tell,
              banner="Phase 1 of 3: WATCH ALL. Pac-Man learns while you watch; its Q-table starts empty.")
    print(f"Phase 2 of 3: SILENT TRAINING, {args.train} games")
    run_phase(lane, args.train, args.display, watch=False, verbose=250,
              banner=f"Phase 2 of 3: SILENT TRAINING, {args.train} games. Press F to watch individual games.")
    win, avg = lane.evaluate(200)
    print(f"\nGreedy play after training: wins {win:.0%} of 200 games, average score {avg:.1f}")
    print(f"Q-table entries (state, action pairs seen): {len(agent.q_values):,}")
    print(f"For Q6(c): open squares {len(env.open_squares())}, pellets {len(env.start_state.food)}\n")
    print("Phase 3 of 3: the trained Pac-Man plays greedily (epsilon = 0, no learning)")
    run_phase(lane, 3, args.display, learn=False, greedy=True, on_game=tell,
              banner="Phase 3 of 3: the trained Pac-Man plays greedily (epsilon = 0, learning off).")
    print("\nNext: answer ANSWER_Q6, run python autograder.py -p 3, then open part4_exploration_sarsa.py")
    finish(args.display, lane, "Part 3 done. Colours and arrows: what Pac-Man learned, from the start "
                               "position. Close the window to exit.")


if __name__ == "__main__":
    run(main)
