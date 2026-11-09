"""
The auto-graded tests (69 of the 100 points). Run them with autograder.py.
Your instructor grades with an untouched copy of this file, so editing it does not change your grade.
"""
import importlib
import json
import os
import random as _random
import re
import sys
import traceback
from collections import defaultdict as _dd

from .engine import PacmanEnv, train, evaluate, greedy_path, exact_hallway_utilities
from .features import simple_features

MODULES = {1: "part1_returns", 2: "part2_td_learning", 3: "part3_qlearning",
           4: "part4_exploration_sarsa", 5: "part5_approximate"}
PART_OF = {"q1": 1, "q2": 2, "q3": 3, "q4": 3, "q5": 3, "q6": 3, "q7": 4, "q8": 4, "q9": 5, "q10": 5}
AUTO_POINTS = {"q1": 6, "q2": 7, "q3": 8, "q4": 6, "q5": 12, "q6": 4, "q7": 6, "q8": 4, "q9": 10, "q10": 6}
WRITTEN_POINTS = {"q2": 3, "q6": 6, "q7": 4, "q8": 6, "q9": 4, "q10": 8}
WRITTEN_PARTS = {"q2": "ab", "q6": "abc", "q7": "ab", "q8": "abc", "q9": "ab", "q10": "abcd"}


class _Toy:
    """A tiny made-up MDP: a dict of legal actions per state."""
    def __init__(self, table):
        self.table = table

    def __call__(self, s):
        return list(self.table.get(s, []))


def _close(a, b, tol=1e-6):
    return a is not None and abs(a - b) <= tol


def _module(part):
    return importlib.import_module(MODULES[part])


# ---------------------------------------------------------------------------- Q1
def tests_q1():
    m = _module(1)
    dr = m.discounted_return
    return [
        ("[1, 1, 1] with gamma 0.5 is 1.75; an empty list is 0", 2,
         lambda: _close(dr([1, 1, 1], 0.5), 1.75) and _close(dr([], 0.9), 0.0)),
        ("gamma = 1 adds up the rewards: the -505 game", 2,
         lambda: _close(dr([-1, -1, -1, -1, -501], 1.0), -505.0)),
        ("a late reward is discounted: [0, 0, 10] with gamma 0.9 is 8.1", 2,
         lambda: _close(dr([0, 0, 10], 0.9), 8.1, 1e-9)),
    ]


# ---------------------------------------------------------------------------- Q2
def tests_q2():
    m = _module(2)

    def unit():
        U = _dd(float)
        U["a"], U["b"] = 2.0, 10.0
        m.td_update(U, "a", 1.0, "b", 0.5, 0.9)
        ok = _close(U["a"], 6.0) and _close(U["b"], 10.0)
        U2 = _dd(float)
        m.td_update(U2, "x", -1.0, "end", 0.25, 0.9)      # next state never visited: U = 0
        return ok and _close(U2["x"], -0.25)

    def converges():
        env = PacmanEnv("hallway")
        U = _dd(float)
        for _ in range(200):
            s, done = env.reset(), False
            while not done:
                s2, r, done = env.step("East")
                m.td_update(U, s, r, s2, 0.5, 0.9)
                s = s2
        exact = exact_hallway_utilities(env, 0.9)
        return max(abs(U[s] - v) for s, v in exact.items()) < 0.01

    return [("one TD update moves U(s) toward r + gamma U(s'), and changes only U(s)", 3, unit),
            ("TD with the 'always East' policy converges to the true utilities", 4, converges)]


# ---------------------------------------------------------------------------- Q3 to Q6
def _toy_agent(cls, **kw):
    acts = _Toy({"s": ["a", "b", "c"], "t": ["a", "b"], "u": ["x", "y"], "end": []})
    return cls(acts, **kw)


def tests_q3():
    Q = _module(3).QLearningAgent

    def v_basic():
        ag = _toy_agent(Q)
        if not _close(ag.get_value("s"), 0.0):
            return False
        ag.q_values[("s", "a")], ag.q_values[("s", "b")], ag.q_values[("s", "c")] = -3.0, 7.0, 2.0
        return _close(ag.get_value("s"), 7.0)

    def v_neg():
        ag = _toy_agent(Q)
        ag.q_values[("t", "a")], ag.q_values[("t", "b")] = -4.0, -2.5
        ag.q_values[("t", "c")] = 100.0                    # 'c' is not legal in state 't'
        return _close(ag.get_value("t"), -2.5)

    def terminal():
        ag = _toy_agent(Q)
        return _close(ag.get_value("end"), 0.0) and ag.best_action("end") is None

    def best():
        ag = _toy_agent(Q)
        ag.q_values[("s", "a")], ag.q_values[("s", "b")], ag.q_values[("s", "c")] = 1.0, -1.0, 5.0
        ag.q_values[("t", "a")], ag.q_values[("t", "b")] = -9.0, -8.0
        return all(ag.best_action("s") == "c" for _ in range(20)) and ag.best_action("t") == "b"

    def via_getter():
        class Sub(Q):
            def get_q_value(self, state, action):      # Part 5 replaces get_q_value like this
                return {"a": 1.0, "b": 9.0, "c": 4.0}[action]
        ag = Sub(_Toy({"s": ["a", "b", "c"]}))
        return _close(ag.get_value("s"), 9.0) and ag.best_action("s") == "b"

    def ties():
        _random.seed(3)
        ag = _toy_agent(Q)
        ag.q_values[("s", "a")], ag.q_values[("s", "b")], ag.q_values[("s", "c")] = 4.0, 4.0, 1.0
        return {ag.best_action("s") for _ in range(200)} == {"a", "b"}

    return [("get_value: 0 for an unseen state, max over legal actions", 2, v_basic),
            ("get_value: all-negative values stay negative; illegal actions ignored", 2, v_neg),
            ("no legal actions: value 0.0 and best_action None", 1, terminal),
            ("best_action returns the highest-Q legal action", 1, best),
            ("reads Q-values through self.get_q_value (needed in Part 5)", 1, via_getter),
            ("ties are broken at random", 1, ties)]


def tests_q4():
    Q = _module(3).QLearningAgent

    def greedy():
        ag = _toy_agent(Q, epsilon=0.0)
        ag.q_values[("s", "b")] = 3.0
        return all(ag.get_action("s") == "b" for _ in range(200)) and ag.get_action("end") is None

    def uniform():
        _random.seed(7)
        ag = _toy_agent(Q, epsilon=1.0)
        ag.q_values[("s", "b")] = 3.0
        k, counts = 6000, {a: 0 for a in "abc"}
        for _ in range(k):
            counts[ag.get_action("s")] += 1
        return all(abs(c / k - 1 / 3) < 0.04 for c in counts.values())

    def mixed():
        _random.seed(11)
        ag = _toy_agent(Q, epsilon=0.5)
        ag.q_values[("s", "c")] = 3.0
        k = 6000
        best = sum(ag.get_action("s") == "c" for _ in range(k)) / k
        return abs(best - (0.5 + 0.5 / 3)) < 0.04      # 0.667 if the random pick may include the best

    return [("epsilon = 0: always the best action (None if no legal actions)", 2, greedy),
            ("epsilon = 1: uniform over ALL legal actions", 2, uniform),
            ("epsilon = 0.5: best action chosen about 67% of the time", 2, mixed)]


def _corridor_qstar(gamma=0.9):
    # states 0..4, actions L/R; reaching 4 ends the episode with +10, every other move -1
    nxt = lambda s, a: max(0, s - 1) if a == "L" else s + 1
    rew = lambda s, a: 10.0 if nxt(s, a) == 4 else -1.0
    V = {s: 0.0 for s in range(5)}
    for _ in range(500):
        for s in range(4):
            V[s] = max(rew(s, a) + gamma * V[nxt(s, a)] for a in "LR")
    return {(s, a): rew(s, a) + gamma * V[nxt(s, a)] for s in range(4) for a in "LR"}, nxt, rew


def tests_q5():
    Q = _module(3).QLearningAgent

    def single():
        ag = _toy_agent(Q, alpha=0.5, gamma=0.9)
        ag.q_values[("u", "x")], ag.q_values[("u", "y")] = 4.0, -2.0
        ag.update("s", "a", "u", 1.0)          # sample = 1 + 0.9 * 4 = 4.6
        return _close(ag.get_q_value("s", "a"), 2.3) and _close(ag.get_q_value("u", "x"), 4.0)

    def negative():
        ag = _toy_agent(Q, alpha=0.5, gamma=0.9)
        ag.q_values[("t", "a")], ag.q_values[("t", "b")] = -10.0, -20.0
        ag.q_values[("s", "b")] = 2.0
        ag.update("s", "b", "t", -1.0)         # sample = -1 + 0.9 * -10 = -10
        return _close(ag.get_q_value("s", "b"), -4.0)

    def terminal():
        ag = _toy_agent(Q, alpha=0.5, gamma=0.9)
        ag.update("s", "c", "end", 509.0)       # no actions in 'end': sample = reward
        return _close(ag.get_q_value("s", "c"), 254.5)

    def alphas():
        a1 = _toy_agent(Q, alpha=1.0, gamma=0.9)
        a1.q_values[("s", "a")], a1.q_values[("t", "a")] = 50.0, 10.0
        a1.update("s", "a", "t", 2.0)
        a0 = _toy_agent(Q, alpha=0.0, gamma=0.9)
        a0.q_values[("s", "a")] = 50.0
        a0.update("s", "a", "t", 2.0)
        return _close(a1.get_q_value("s", "a"), 11.0) and _close(a0.get_q_value("s", "a"), 50.0)

    def converge():
        qstar, nxt, rew = _corridor_qstar()
        ag = Q(_Toy({**{s: ["L", "R"] for s in range(4)}, 4: []}), alpha=0.5, epsilon=1.0, gamma=0.9)
        rng = _random.Random(5)
        for _ in range(3000):
            s, a = rng.randrange(4), rng.choice("LR")
            ag.update(s, a, nxt(s, a), rew(s, a))
        return all(abs(ag.get_q_value(s, a) - q) < 0.01 for (s, a), q in qstar.items())

    return [("one update: Q <- (1 - alpha) Q + alpha (r + gamma max Q(s', a'))", 3, single),
            ("uses V(s') even when it is negative", 2, negative),
            ("game over: the sample is just the reward", 2, terminal),
            ("alpha = 1 copies the sample; alpha = 0 changes nothing", 2, alphas),
            ("converges to Q* on a 5-state corridor", 3, converge)]


def tests_q6():
    Q = _module(3).QLearningAgent

    def learns():
        _random.seed(0)
        env = PacmanEnv("tricky", seed=0)
        ag = Q(env.legal_actions, alpha=0.5, epsilon=0.1, gamma=0.9)
        train(ag, env, 2000)
        return evaluate(ag, env, 100)[0] >= 0.80

    return [("after 2,000 training games Pac-Man wins at least 80% of greedy games", 4, learns)]


# ---------------------------------------------------------------------------- Q7, Q8
def tests_q7():
    E = _module(4).ExplorationQAgent

    def fval():
        ag = E(_Toy({}), r_plus=600.0, n_e=3)
        return (ag.exploration_value(5.0, 0) == 600.0 and ag.exploration_value(5.0, 2) == 600.0
                and ag.exploration_value(5.0, 3) == 5.0 and ag.exploration_value(-20.0, 10) == -20.0)

    def picks():
        _random.seed(2)
        ag = E(_Toy({"s": ["a", "b", "c"]}), r_plus=600.0, n_e=3)
        ag.q_values[("s", "a")], ag.counts[("s", "a")] = 100.0, 5
        ag.q_values[("s", "b")], ag.counts[("s", "b")] = -3.0, 5
        first = ag.get_action("s") == "c"                 # an untried action looks worth R+
        ag.counts[("s", "c")], ag.q_values[("s", "c")] = 3, -50.0
        second = all(ag.get_action("s") == "a" for _ in range(20))
        ag.q_values[("s", "b")] = 100.0
        tie = {ag.get_action("s") for _ in range(100)} == {"a", "b"}
        return first and second and tie

    def learns():
        _random.seed(0)
        env = PacmanEnv("tricky", seed=0)
        ag = E(env.legal_actions, alpha=0.5, gamma=0.9, r_plus=600.0, n_e=3)
        train(ag, env, 1000)
        return evaluate(ag, env, 100)[0] >= 0.80

    return [("f(u, n) = R+ while n < N_e, otherwise u", 2, fval),
            ("get_action: argmax of f(Q, N), random among ties", 2, picks),
            ("the exploring agent learns to win on 'tricky'", 2, learns)]


def tests_q8():
    S = _module(4).SarsaAgent

    def target():
        ag = S(_Toy({"t": ["a", "b"]}), alpha=0.5, epsilon=0.1, gamma=0.9)
        ag.q_values[("t", "a")], ag.q_values[("t", "b")] = 1.0, 50.0
        return (_close(ag.sarsa_target(2.0, "t", "a"), 2.9)            # Q(s', a'), not the max
                and _close(ag.sarsa_target(509.0, "end", None), 509.0))

    def safe_path():
        ok = 0
        for seed in (0, 1, 2):
            _random.seed(seed)
            env = PacmanEnv("ghostCliff", seed=seed)
            ag = S(env.legal_actions, alpha=0.5, epsilon=0.1, gamma=0.9)
            train(ag, env, 1000)
            ok += sum(1 for (x, y) in greedy_path(ag, env) if y == 3) <= 2   # only the start and the pellet
        return ok >= 2

    return [("sarsa_target = r + gamma Q(s', a')  (just r if a' is None)", 2, target),
            ("SARSA learns to stay away from the sleeping ghosts", 2, safe_path)]


# ---------------------------------------------------------------------------- Q9, Q10
def tests_q9():
    A = _module(5).ApproximateQAgent
    fake = lambda env, s, a: {"bias": 1.0, "f1": 2.0 if a == "a" else 0.0, "f2": -1.0}

    def make(alpha):
        ag = A(None, feature_fn=fake, alpha=alpha, epsilon=0.0, gamma=0.9)
        ag.action_fn = _Toy({"s": ["a", "b"], "end": []})
        return ag

    def qval():
        ag = make(0.1)
        ag.weights.update({"bias": 0.5, "f1": 3.0, "f2": 4.0})
        return _close(ag.get_q_value("s", "a"), 0.5 + 6.0 - 4.0) and _close(ag.get_q_value("s", "b"), -3.5)

    def upd():
        ag = make(0.1)
        ag.weights.update({"bias": 1.0, "f1": 0.0, "f2": 0.0})
        ag.update("s", "a", "s", 2.0)       # Q(s,a) = 1, V(s) = 1, so delta = 2 + 0.9 - 1 = 1.9
        ok1 = (_close(ag.weights["bias"], 1.19) and _close(ag.weights["f1"], 0.38)
               and _close(ag.weights["f2"], -0.19))
        ag2 = make(0.5)
        ag2.update("s", "b", "end", -500.0)  # game over: delta = -500
        return ok1 and _close(ag2.weights["bias"], -250.0) and _close(ag2.weights["f1"], 0.0) \
            and _close(ag2.weights["f2"], 250.0)

    def learns():
        _random.seed(0)
        env = PacmanEnv("twoGhosts", seed=0)
        ag = A(env, simple_features, alpha=0.2, epsilon=0.05, gamma=0.8)
        train(ag, env, 100)
        return evaluate(ag, env, 100)[0] >= 0.65

    return [("Q(s, a) = sum_i w_i f_i(s, a)", 3, qval),
            ("w_i <- w_i + alpha * delta * f_i(s, a)", 4, upd),
            ("approximate Q-learning wins at least 65% on 'twoGhosts' after 100 games", 3, learns)]


def tests_q10():
    m = _module(5)
    base_keys = {"bias", "ghosts-1-step-away", "eats-food", "closest-food"}

    def valid():
        env = PacmanEnv("twoGhosts", seed=1)
        rng = _random.Random(1)
        s, new = env.reset(), set()
        for _ in range(60):
            acts = env.legal_actions(s)
            if not acts:
                s = env.reset()
                continue
            a = rng.choice(acts)
            f = m.my_features(env, s, a)
            if not isinstance(f, dict) or not all(isinstance(v, (int, float)) for v in f.values()):
                return False
            new |= set(f) - base_keys
            s, _, done = env.step(a)
            if done:
                s = env.reset()
        return len(new) >= 1

    def learns():
        _random.seed(0)
        env = PacmanEnv("twoGhosts", seed=0)
        ag = m.ApproximateQAgent(env, m.my_features, alpha=0.2, epsilon=0.05, gamma=0.8)
        train(ag, env, 100)
        return evaluate(ag, env, 100)[0] >= 0.40

    return [("my_features returns a dict of numbers with at least one new feature", 3, valid),
            ("an agent using my_features trains and wins at least 40% of games", 3, learns)]


TESTS = {q: globals()[f"tests_{q}"] for q in AUTO_POINTS}


# ---------------------------------------------------------------------------- running
def _where(exc):
    """' at part3_qlearning.py line 88' for the deepest line of student code in the traceback."""
    frames = [f for f in traceback.extract_tb(exc.__traceback__)
              if os.path.basename(f.filename).startswith("part")]
    return f" at {os.path.basename(frames[-1].filename)} line {frames[-1].lineno}" if frames else ""


def grade_question(q):
    """Returns (points, possible, list of result lines)."""
    lines, earned = [], 0
    try:
        tests = TESTS[q]()
    except Exception as e:
        return 0, AUTO_POINTS[q], [f"  [ ] could not load {MODULES[PART_OF[q]]}.py ({type(e).__name__}: {e}{_where(e)})"]
    for name, pts, fn in tests:
        try:
            ok, note = bool(fn()), ""
        except NotImplementedError as e:
            ok, note = False, f"not written yet{_where(e)}"
        except Exception as e:
            ok, note = False, f"{type(e).__name__}: {e}{_where(e)}"
        earned += pts if ok else 0
        lines.append(("  [x] " if ok else "  [ ] ") + f"{name}  ({pts if ok else 0}/{pts})"
                     + (f"\n        {note}" if note else ""))
    return earned, AUTO_POINTS[q], lines


def written_status(q):
    """'filled in', 'partly filled in' or 'empty' for a written question."""
    try:
        text = getattr(_module(PART_OF[q]), f"ANSWER_{q.upper()}", "")
    except Exception:
        return "could not read"
    pieces = re.split(r"^\s*\(([a-d])\)", text or "", flags=re.M)
    answers = {pieces[i]: pieces[i + 1] for i in range(1, len(pieces) - 1, 2)}
    labels = WRITTEN_PARTS[q]
    if not answers:      # no (a)/(b) labels: judge the whole text
        n = len(re.sub(r"\s+", "", text or ""))
        return "filled in" if n >= 40 * len(labels) else "partly filled in" if n else "empty"
    done = sum(1 for k in labels if len(re.sub(r"\s+", "", answers.get(k, ""))) >= 25)
    return "filled in" if done == len(labels) else "partly filled in" if done else "empty"


def _dependency_note(part):
    if part in (4, 5):
        core = sum(grade_question(q)[0] for q in ("q3", "q4", "q5"))
        if core < AUTO_POINTS["q3"] + AUTO_POINTS["q4"] + AUTO_POINTS["q5"]:
            return ("  Note: Part %d builds on your Part 3 QLearningAgent, which does not pass all of its "
                    "checks yet.\n  Run python autograder.py -p 3 first." % part)
    return ""


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Auto-grader for the Pac-Man reinforcement learning lab.")
    p.add_argument("-p", "--part", type=int, choices=range(1, 6), help="check one part (1-5)")
    p.add_argument("-q", "--question", choices=sorted(AUTO_POINTS, key=lambda s: int(s[1:])),
                   help="check one question (q1-q10)")
    p.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    sys.path.insert(0, ".")

    if args.json:     # used by the instructor's grading script
        out = {}
        for q in AUTO_POINTS:
            pts, possible, lines = grade_question(q)
            out[q] = {"points": pts, "possible": possible, "details": lines}
        print("@@RESULT@@" + json.dumps(out))
        return

    qs = ([args.question] if args.question else
          [q for q in AUTO_POINTS if PART_OF[q] == args.part] if args.part else list(AUTO_POINTS))
    total = possible = 0
    shown_notes = set()
    for q in qs:
        part = PART_OF[q]
        if part not in shown_notes:
            shown_notes.add(part)
            print(f"\n===== Part {part}: {MODULES[part]}.py =====")
            note = _dependency_note(part)
            if note:
                print(note)
        pts, poss, lines = grade_question(q)
        total, possible = total + pts, possible + poss
        print(f"Question {q}: {pts}/{poss}")
        print("\n".join(lines))
        if q in WRITTEN_POINTS:
            print(f"  Written answer ANSWER_{q.upper()} ({WRITTEN_POINTS[q]} points, graded by your instructor): "
                  f"{written_status(q)}")
    print(f"\nAuto-graded points: {total}/{possible}")
    if not args.question and not args.part:
        filled = sum(written_status(q) == "filled in" for q in WRITTEN_POINTS)
        print(f"Written answers filled in: {filled} of {len(WRITTEN_POINTS)} "
              f"(31 points, graded by your instructor)")
