"""
===============================================================================
PART 1   Rewards and returns                                      AIMA 4e §22.1
Question Q1 (6 points)                                            about 8 minutes

    Write your code:   discounted_return() below
    Watch it:          python part1_returns.py
    Check it:          python autograder.py -q q1
===============================================================================

NOTES
-----
In Chapter 17 (Markov decision processes) the agent was GIVEN the transition
model P(s' | s, a) and the reward function R(s, a, s'), so it could plan with
value iteration before taking a single step. In reinforcement learning the
agent is dropped into the world WITHOUT the model. All it gets is experience,
one move at a time:

        s0 --a0--> (r1, s1) --a1--> (r2, s2) --a2--> (r3, s3) ...

and from the rewards alone it has to learn to act well.

Return. The agent tries to maximize the discounted sum of future rewards, with
a discount factor 0 <= γ <= 1:

        G  =  r1 + γ·r2 + γ²·r3 + ...  =  Σ (k = 0 .. T-1)  γ^k · r(k+1)

A reward k steps in the future is worth γ^k of its face value. The SCORE under
the board is the same sum with γ = 1.

Utility of a state under a policy π: the expected return if you start in s and
follow π:

        U^π(s)  =  E[ Σ_t  γ^t · R(S_t, π(S_t), S_t+1)  |  S_0 = s ]

Bellman equations (from Chapter 17). The optimal values satisfy

        U(s)    =  max_a  Σ_s'  P(s'|s,a) · [ R(s,a,s') + γ · U(s') ]
        Q(s,a)  =         Σ_s'  P(s'|s,a) · [ R(s,a,s') + γ · max_a' Q(s',a') ]

        U(s) = max_a Q(s,a)              π*(s) = argmax_a Q(s,a)

That last line is the key to this lab: if you know Q, you can act optimally
without knowing P or R. That is why Q-learning (Part 3) learns Q directly.

THE PAC-MAN WORLD AS AN MDP
    state     State(pacman, ghosts, food): Pac-Man's square, each ghost's square,
              and the set of pellets left
    actions   North, South, East, West (the ones that do not walk into a wall)
    moves     Pac-Man moves, then each ghost steps to a random neighbouring
              square. Pac-Man is never told this.
    rewards   -1 per move, +10 per pellet, +500 for the last pellet (win),
              -500 for touching a ghost (lose)

WHAT YOU WILL SEE
    A Pac-Man that moves at random (it cannot learn yet) plays 5 games on the
    board from the lecture demo. The terminal shows each game's rewards, its
    score, and the return your function computes from them.
"""
from pacman import PacmanEnv, RandomAgent, Lane, run_phase, describe_rewards, outcome_text, setup, run


# =============================================================================
# Q1. Discounted return (6 points)
#
#     G = Σ (k = 0 .. T-1) γ^k · r(k+1)        rewards = [r1, r2, ..., rT]
#
# An empty list of rewards gives 0.
# =============================================================================
def discounted_return(rewards, gamma):
    # YOUR CODE HERE
    raise NotImplementedError


# =============================================================================
# Demo (you do not need to change anything below)
# =============================================================================
def main():
    args = setup("Part 1: rewards and returns. A Pac-Man that moves at random plays 5 games.")
    died = [-1, -1, -1, -1, -501]
    print("The lecture's -505 game: four moves, then a fifth move into the ghost.")
    print(f"  rewards                        {died}")
    print(f"  score on screen   (gamma = 1)  {discounted_return(died, 1.0)}")
    print(f"  return the agent sees (0.9)    {discounted_return(died, 0.9):.2f}\n")

    env = PacmanEnv("tricky", seed=1)
    lane = Lane("Random Pac-Man (no learning)", RandomAgent(env), env, seed=1, overlay=None)

    def report(lane):
        r = lane.last_rewards
        print(f"Game {len(lane.scores)}: {outcome_text(lane.outcomes[-1])} Score: {lane.scores[-1]}")
        print(f"    rewards          {describe_rewards(r)}")
        print(f"    return, gamma 0.9 = {discounted_return(r, 0.9):9.2f}      "
              f"gamma 0.5 = {discounted_return(r, 0.5):9.2f}")

    run_phase(lane, 5, args.display, learn=False, on_game=report,
              banner="Part 1: this Pac-Man picks moves at random and never learns. Watch the score: "
                     "-1 per move, +10 per pellet, -500 for touching the ghost.")
    print("\nNext: python autograder.py -q q1     then open part2_td_learning.py")
    if args.display:
        args.display.hold("Part 1 done. Close the window (or press Q) and run: python autograder.py -q q1")


if __name__ == "__main__":
    run(main)
