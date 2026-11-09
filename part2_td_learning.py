"""
===============================================================================
PART 2   Passive reinforcement learning: TD learning              AIMA 4e §22.2
Question Q2 (10 points: 7 code + 3 written)                       about 12 minutes

    Write your code:   td_update() below, then answer ANSWER_Q2
    Watch it:          python part2_td_learning.py
    Check it:          python autograder.py -q q2
===============================================================================

NOTES
-----
In PASSIVE reinforcement learning the policy π is fixed. The agent does not
choose its moves; it watches π play and learns how good each state is, U^π(s).
Chapter 22 gives three ways to do it:

  Direct utility estimation (§22.2.1)
      U^π(s) = average of the returns observed after visiting s.
      Ignores that a state's utility depends on its neighbours', so it is slow.

  Adaptive dynamic programming, ADP (§22.2.2)
      Count transitions to estimate a model P̂(s'|s,a) and R̂, then solve
          U^π(s) = Σ_s' P̂(s'|s,π(s)) · [ R̂(s,π(s),s') + γ · U^π(s') ]
      Learns a model.

  Temporal-difference learning, TD (§22.2.3)
      After every move, nudge U^π(s) toward the one-step sample r + γ·U^π(s').
      No model.

THE TD UPDATE. After the agent moves from s to s' and receives reward r:

        U(s)  ←  U(s) + α · [ r + γ · U(s')  −  U(s) ]
                              └── sample ─┘
                            └────── TD error δ ──────┘

  * If the move went better than predicted (δ > 0), U(s) goes up; worse, down.
  * α is the learning rate. TD needs no P(s'|s,a): successor states turn up as
    often as P says they should, so the world does the averaging for you.
  * With a decaying rate such as α_n = 1/n (any schedule with Σ α_n = ∞ and
    Σ α_n² < ∞), U converges to the true utilities U^π.
  * A finished game has no future, so U(terminal) = 0 and the last sample is r.

WHAT YOU WILL SEE
    A hallway with one pellet and no ghost. The fixed policy is "always East".
    Every square shows the utility U(s) your td_update has learned so far
    (green = good). Watch the good news travel back from the pellet, game by
    game. The terminal prints the table after games 1, 2, 3, 6 and 200 and
    compares it with the true utilities.
"""
from collections import defaultdict

from pacman import PacmanEnv, Lane, run_phase, finish, exact_hallway_utilities, setup, run


# =============================================================================
# Q2. One TD(0) step (7 points)
#
#     U(s) ← U(s) + α · [ r + γ · U(s') − U(s) ]
#
# Change U[state] in place (return nothing). U is a defaultdict(float), so a
# state never seen before, including the end of the game, has utility 0.
# Change only U[state].
# =============================================================================
def td_update(U, state, reward, next_state, alpha, gamma):
    # YOUR CODE HERE
    raise NotImplementedError


# =============================================================================
# Q2 written (3 points). Answer in the string below, using the numbers from YOUR
# terminal output.
#
# (a) After game 1 only the square next to the pellet has a large positive
#     utility; after game 2 the next square back has one too. Explain, using the
#     TD update, why good news travels back ONE square per game. How would
#     direct utility estimation have done after game 1?
# (b) What would ADP have to learn that TD does not, and in what kind of world
#     would that extra work pay off?
# =============================================================================
ANSWER_Q2 = """
(a)

(b)

"""


# =============================================================================
# Demo (you do not need to change anything below)
# =============================================================================
class AlwaysEastTD:
    """Follows the fixed policy 'always East' and learns U(s) with your td_update."""
    epsilon = 0.0

    def __init__(self, alpha=0.5, gamma=0.9):
        self.U = defaultdict(float)
        self.alpha, self.gamma = alpha, gamma

    def get_action(self, state):
        return "East"

    def update(self, state, action, next_state, reward):
        td_update(self.U, state, reward, next_state, self.alpha, self.gamma)


def main():
    args = setup("Part 2: TD learning in a hallway with the fixed policy 'always East'.")
    env = PacmanEnv("hallway")
    agent = AlwaysEastTD(alpha=0.5, gamma=0.9)
    lane = Lane("TD learning: policy 'always East', alpha 0.5, gamma 0.9", agent, env,
                overlay="u", table=agent.U)
    exact = exact_hallway_utilities(env, agent.gamma)
    squares = sorted(exact, key=lambda s: s.pacman[0])
    row = lambda vals: "".join(f"{v:9.1f}" for v in vals)

    print(f"{'Pac-Man square x':24s}" + "".join(f"{s.pacman[0]:9d}" for s in squares))

    def snapshot(lane):
        n = len(lane.scores)
        if n in (1, 2, 3, 6, 200):
            print(f"{'U after game ' + str(n):24s}" + row(agent.U[s] for s in squares))

    run_phase(lane, 10, args.display, on_game=snapshot, speed=2,
              banner="Phase 1 of 2: watch 10 games. Each square shows the utility U(s) learned so far. "
                     "See how the good news from the pellet moves back one square per game.")
    run_phase(lane, 190, args.display, watch=False, on_game=snapshot,
              banner="Phase 2 of 2: 190 more games (fast-forward). U(s) settles at the true utilities.")
    print(f"{'true U (from the rules)':24s}" + row(exact[s] for s in squares))
    print("\nNext: answer ANSWER_Q2, run python autograder.py -q q2, then open part3_qlearning.py")
    finish(args.display, lane, "Part 2 done: the squares show the utilities TD learned. "
                               "Close the window and run: python autograder.py -q q2")


if __name__ == "__main__":
    run(main)
