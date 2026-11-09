"""
===============================================================================
PART 4   Smarter exploration, and on-policy learning (SARSA)      AIMA 4e §22.3
Questions Q7 and Q8 (20 points: 10 code + 10 written)             about 20 minutes

    Write your code:   ExplorationQAgent (Q7) and SarsaAgent.sarsa_target (Q8),
                       then ANSWER_Q7 and ANSWER_Q8
    Check it:          python autograder.py -p 4
    Watch it:          python part4_exploration_sarsa.py explore     (for Q7)
                       python part4_exploration_sarsa.py cliff       (for Q8)
    This part builds on your QLearningAgent from Part 3.
===============================================================================

NOTES
-----
EXPLORATION FUNCTIONS (§22.3.1). ε-greedy explores blindly and forever. AIMA's
alternative is optimism under uncertainty: treat an action you have not tried
often enough as if it were great.

        f(u, n)  =  R+   if n < N_e
                    u    otherwise

        a  =  argmax_a'  f( Q(s,a'), N(s,a') )

  * u is the current estimate Q(s,a); n = N(s,a) is how many times a has been
    tried in s.
  * R+ is an optimistic guess, at least as large as the best real value; N_e is
    how many tries an action gets before the agent trusts its estimate.
  * Every action is tried at least N_e times in every state the agent visits,
    then it acts greedily: a GLIE scheme.

ON-POLICY LEARNING: SARSA (§22.3.3). SARSA updates toward the value of the
action the agent WILL ACTUALLY TAKE next, a', chosen by the same ε-greedy rule
(the name lists what it uses: s, a, r, s', a'):

        SARSA       Q(s,a) ← Q(s,a) + α · [ r + γ · Q(s', a')       − Q(s,a) ]
        Q-learning  Q(s,a) ← Q(s,a) + α · [ r + γ · max_a' Q(s',a') − Q(s,a) ]

                    target uses                 learns the value of
    Q-learning      the best next action        the greedy policy  (OFF-policy)
    SARSA           the next action taken       the policy it follows, exploration
                                                included             (ON-policy)

The two agree when ε → 0. They disagree when exploring is dangerous: the SAFE
EXPLORATION problem of §22.3.2 (a real robot cannot afford to fall down the
stairs while it learns).

WHAT YOU WILL SEE
    explore  Three Pac-Men learn the 'tricky' board side by side: ε-greedy,
             pure greedy (ε = 0), and your exploration function. Compare the
             learning curves under the boards.
    cliff    Q-learning and SARSA side by side on 'ghostCliff'. The grey ghosts
             are asleep and never move, but touching one still costs -500. The
             shortest route runs right along them. At the end the dashed line
             shows each agent's greedy path.
"""
import random
from collections import defaultdict

from pacman import PacmanEnv, Lane, run_phase, finish, greedy_path, setup, run
from part3_qlearning import QLearningAgent


# =============================================================================
# Q7. An exploring agent (6 points)
#
#     exploration_value(u, n) = R+ if n < N_e else u        (self.r_plus, self.n_e)
#     get_action(s)           = argmax over legal a of f(Q(s,a), N(s,a)),
#                               ties at random, None if there are no legal actions
#
# The counts N(s,a) are in self.counts[(state, action)]. When self.exploring is
# False (during evaluation), get_action just returns self.best_action(state).
# update() is done for you: it counts the visit, then calls your Part 3 update.
# =============================================================================
class ExplorationQAgent(QLearningAgent):

    def __init__(self, action_fn, alpha=0.5, gamma=0.9, r_plus=600.0, n_e=3):
        super().__init__(action_fn, alpha=alpha, epsilon=0.0, gamma=gamma)
        self.r_plus = r_plus             # optimistic value for actions not tried enough
        self.n_e = n_e                   # tries before an estimate is trusted
        self.counts = defaultdict(int)   # N(s, a)
        self.exploring = True            # switched off while the agent is evaluated

    def exploration_value(self, u, n):
        # YOUR CODE HERE
        raise NotImplementedError

    def get_action(self, state):
        # YOUR CODE HERE
        raise NotImplementedError

    def update(self, state, action, next_state, reward):
        self.counts[(state, action)] += 1
        super().update(state, action, next_state, reward)


# =============================================================================
# Q8. SARSA (4 points)
#
#     sarsa_target(r, s', a') = r + γ · Q(s', a')      or just r if a' is None
#
# The rest of the class is done for you: update() picks a' with ε-greedy BEFORE
# updating, and get_action() makes sure Pac-Man really plays that a' next.
# =============================================================================
class SarsaAgent(QLearningAgent):

    def __init__(self, action_fn, alpha=0.5, epsilon=0.1, gamma=0.9):
        super().__init__(action_fn, alpha=alpha, epsilon=epsilon, gamma=gamma)
        self._planned = None

    def sarsa_target(self, reward, next_state, next_action):
        # YOUR CODE HERE
        raise NotImplementedError

    def get_action(self, state):
        if self._planned is not None and self._planned[0] == state:
            action, self._planned = self._planned[1], None
            return action
        return super().get_action(state)

    def update(self, state, action, next_state, reward):
        next_action = QLearningAgent.get_action(self, next_state)   # ε-greedy choice of a'
        self._planned = (next_state, next_action)                   # ...which will really be played
        q = self.get_q_value(state, action)
        self.q_values[(state, action)] = q + self.alpha * (self.sarsa_target(reward, next_state, next_action) - q)

    def end_episode(self):
        self._planned = None


# =============================================================================
# Q7 written (4 points). Run:  python part4_exploration_sarsa.py explore
#
# (a) Late in training, the exploration-function agent's training score is higher
#     than the ε = 0.1 agent's, yet both greedy policies win nearly every game.
#     Explain why, using the idea of GLIE.
# (b) Pure greedy (ε = 0) also learns here, although greedy agents usually get
#     stuck. Every move costs -1 and every Q-value starts at 0. What happens to
#     Q(s,a) the first time an ordinary move is tried, compared with an untried
#     move, and what does the greedy rule therefore do? How is that related to
#     the exploration function?
# =============================================================================
ANSWER_Q7 = """
(a)

(b)

"""

# =============================================================================
# Q8 written (6 points). Run:  python part4_exploration_sarsa.py cliff
#
# (a) Describe the two greedy paths (dashed lines) and give each agent's number
#     of deaths and average score DURING training, from your terminal output.
# (b) Explain the difference using the two targets. What does each target assume
#     about the move Pac-Man makes after s', and why does that make the squares
#     next to the sleeping ghosts look worse to SARSA than to Q-learning?
# (c) You are training a real cleaning robot in a house with an open staircase.
#     Which algorithm would you use WHILE IT LEARNS, and does your answer change
#     once learning is finished and ε is set to 0? Justify.
# =============================================================================
ANSWER_Q8 = """
(a)

(b)

(c)

"""


# =============================================================================
# Demo (you do not need to change anything below)
# =============================================================================
def explore(args):
    makers = [("epsilon-greedy, epsilon 0.1", lambda e: QLearningAgent(e.legal_actions, 0.5, 0.1, 0.9)),
              ("pure greedy, epsilon 0", lambda e: QLearningAgent(e.legal_actions, 0.5, 0.0, 0.9)),
              ("exploration function, R+ 600, N_e 3", lambda e: ExplorationQAgent(e.legal_actions, 0.5, 0.9, 600.0, 3))]
    lanes = []
    for name, make in makers:
        env = PacmanEnv("tricky", seed=7)
        lanes.append(Lane(name, make(env), env, seed=7))
    run_phase(lanes, 2, args.display, banner="Q7: three ways to explore, the same board. Watch the first 2 games.")
    run_phase(lanes, 998, args.display, watch=False,
              banner="Q7: 1,000 training games each (fast-forward). Compare the learning curves.")
    print(f"\n{'agent':40s} {'training avg (last 200)':>24s} {'greedy win rate':>16s}")
    for lane in lanes:
        win, _ = lane.evaluate(200)
        print(f"{lane.name:40s} {sum(lane.scores[-200:]) / 200:24.1f} {win:16.0%}")
    return lanes


def cliff(args):
    lanes = []
    for name, cls in (("Q-learning (off-policy)", QLearningAgent), ("SARSA (on-policy)", SarsaAgent)):
        env = PacmanEnv("ghostCliff", seed=1)
        lanes.append(Lane(name, cls(env.legal_actions, alpha=0.5, epsilon=0.1, gamma=0.9), env, seed=1))
    run_phase(lanes, 3, args.display, banner="Q8: Q-learning vs SARSA, epsilon 0.1. The grey ghosts never move, "
                                             "but touching one is -500. Watch the first 3 games.")
    run_phase(lanes, 997, args.display, watch=False,
              banner="Q8: 1,000 training games each (fast-forward). Watch the death counts.")
    print()
    for lane in lanes:
        lane.path = greedy_path(lane.agent, lane.env)
        deaths = sum(1 for o in lane.outcomes if o == "lose")
        print(f"{lane.name:25s} training deaths: {deaths:3d} of 1000 | training avg score: "
              f"{sum(lane.scores) / len(lane.scores):6.1f} | greedy path: {len(lane.path) - 1} moves")
        lane.message = f"greedy path: {len(lane.path) - 1} moves, {deaths} deaths while training"
    return lanes


def main():
    args = setup("Part 4: exploration functions and SARSA.",
                 lambda p: p.add_argument("experiment", nargs="?", default="both",
                                          choices=["explore", "cliff", "both"],
                                          help="explore (Q7), cliff (Q8), or both (default)"))
    if args.experiment in ("explore", "both"):
        shown = explore(args)
        message = "Q7 done: the colours show what each agent learned. Close the window to exit."
    if args.experiment in ("cliff", "both"):
        shown = cliff(args)
        message = "Q8: the dashed lines are the greedy paths after training. Close the window to exit."
    print("\nNext: answer ANSWER_Q7 and ANSWER_Q8, run python autograder.py -p 4, then open part5_approximate.py")
    finish(args.display, shown, message)


if __name__ == "__main__":
    run(main)
