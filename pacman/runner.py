"""
Runs one or more agents ("lanes") side by side, with or without a window.

Every lane has its own random-number stream, so the numbers printed in the terminal
are the same whether you watch every move, fast-forward, skip, or close the window.
"""
import random
import time
from contextlib import contextmanager

from .engine import evaluate, greedy_mode, State


class Lane:
    """One agent playing on one board."""

    def __init__(self, name, agent, env, seed=0, overlay="q", table=None):
        self.name, self.agent, self.env = name, agent, env
        self.rng_state = random.Random(seed).getstate()
        self.overlay = overlay          # "q": values from the agent, "u": values from `table`, None: off
        self.table = table              # utility table for overlay="u"
        self.scores, self.outcomes = [], []
        self.rewards, self.last_rewards = [], []
        self.state = None               # None = start a new game on the next step
        self.view_state = env.start_state
        self.last_action = None
        self.games_in_phase = 0
        self.target = 0
        self.path = None                # list of squares to draw (greedy path)
        self.deaths = None              # Counter of squares to mark with red circles
        self.message = ""
        self.caption = None             # replaces the SCORE line while fast-forwarding

    @contextmanager
    def own_rng(self):
        outer = random.getstate()
        random.setstate(self.rng_state)
        try:
            yield
        finally:
            self.rng_state = random.getstate()
            random.setstate(outer)

    def finished(self):
        return self.games_in_phase >= self.target

    def step(self, learn):
        """One move. Returns True if that move ended a game. Call inside own_rng()."""
        if self.state is None:
            self.state = self.env.reset()
            self.rewards = []
            self.last_action = None
        action = self.agent.get_action(self.state)
        next_state, reward, done = self.env.step(action)
        if learn:
            self.agent.update(self.state, action, next_state, reward)
        self.rewards.append(reward)
        self.last_action = action
        self.view_state = next_state
        self.state = next_state
        if done:
            if hasattr(self.agent, "end_episode"):
                self.agent.end_episode()
            self.scores.append(self.env.score)
            self.outcomes.append(self.env.outcome())
            self.last_rewards = self.rewards
            self.games_in_phase += 1
            self.state = None
            return True
        return False

    def play_rest_of_game(self, learn):
        while not self.step(learn):
            pass

    def evaluate(self, games=100):
        """Greedy win rate and average score (uses this lane's random stream)."""
        with self.own_rng():
            result = evaluate(self.agent, self.env, games)
        self.env.reset()
        self.view_state = self.env.start_state
        return result


def _report(lane, verbose, prefix):
    n = lane.games_in_phase
    if verbose and n % verbose == 0:
        recent = lane.scores[-verbose:]
        wins = lane.outcomes[-verbose:].count("win")
        print(f"  {prefix}game {n:5d} | avg score (last {verbose}) {sum(recent) / len(recent):8.1f} | "
              f"wins {wins}/{verbose}", flush=True)


def run_phase(lanes, games, display=None, learn=True, greedy=False, watch=True,
              banner="", on_game=None, verbose=0, speed=None):
    """Play `games` games in every lane (an int, or a list with one number per lane).

    learn   call agent.update() after every move
    greedy  turn exploration off (epsilon = 0) during this phase
    watch   start by showing every move (True) or fast-forwarding (False)
    on_game function(lane) called after each finished game
    verbose print a progress line every `verbose` games
    """
    if isinstance(lanes, Lane):
        lanes = [lanes]
    targets = games if isinstance(games, (list, tuple)) else [games] * len(lanes)
    for lane, t in zip(lanes, targets):
        lane.target, lane.games_in_phase = t, 0
        lane.path = None
        lane.deaths = None
    multi = len(lanes) > 1

    def finish_game(lane):
        if on_game:
            on_game(lane)
        _report(lane, verbose, f"[{lane.name}] " if multi else "")

    guards = [greedy_mode(l.agent) for l in lanes] if greedy else []
    for g in guards:
        g.__enter__()
    try:
        if display is not None and not display.closed:
            display.begin_phase(lanes, banner, watch, speed)
        while not all(l.finished() for l in lanes):
            if display is None or display.closed or display.skipping:
                for lane in lanes:
                    with lane.own_rng():
                        while not lane.finished():
                            lane.play_rest_of_game(learn)
                            finish_game(lane)
                break
            if display.paused:
                display.wait(0.05)
                continue
            if display.fast:
                # fast-forward: spread the phase over about display.fast_seconds (0 = flat out)
                t0, frame = time.time(), 0.1
                per_frame = {id(l): max(1, -(-l.target // int(display.fast_seconds / frame)))
                             for l in lanes} if display.fast_seconds > 0 else None
                played = {id(l): 0 for l in lanes}
                while not all(l.finished() for l in lanes):
                    if per_frame is None and time.time() - t0 >= frame:
                        break
                    busy = False
                    for lane in lanes:
                        if lane.finished() or (per_frame and played[id(lane)] >= per_frame[id(lane)]):
                            continue
                        with lane.own_rng():
                            lane.play_rest_of_game(learn)
                        finish_game(lane)
                        played[id(lane)] += 1
                        busy = True
                    if not busy:
                        break
                for lane in lanes:     # show what has been learned, from the start position
                    if lane.scores:
                        lane.caption = f"game {len(lane.scores)}: {OUTCOME_WORD[lane.outcomes[-1]]}, score {lane.scores[-1]}"
                    lane.view_state, lane.last_action = lane.env.start_state, None
                display.draw(lanes)
                display.wait(max(0.0, frame - (time.time() - t0)) if per_frame else 0)
            else:
                any_ended = False
                for lane in lanes:
                    lane.caption = None
                    if not lane.finished():
                        with lane.own_rng():
                            ended = lane.step(learn)
                        if ended:
                            any_ended = True
                            finish_game(lane)
                display.draw(lanes)
                # linger on the last frame of a game so you can see how it ended
                display.wait(max(4 * display.delay, 0.4) if any_ended and display.delay > 0 else display.delay)
        if display is not None and not display.closed:
            display.draw(lanes)
    finally:
        for g in guards:
            g.__exit__(None, None, None)


OUTCOME_WORD = {"win": "won", "lose": "died", "timeout": "out of time"}


def finish(display, lanes, message):
    """Show every lane from its start position (with the values it learned) and keep the window open."""
    if display is None or display.closed:
        return
    lanes = [lanes] if isinstance(lanes, Lane) else lanes
    for lane in lanes:
        lane.view_state, lane.last_action = lane.env.reset(), None
        lane.caption = "after training" if lane.scores and lane.overlay else None
    display.draw(lanes)
    display.hold(message)


def describe_rewards(rewards):
    """[-1, -1, -1, 9, -501] -> '-1 x3, 9, -501'"""
    out, i = [], 0
    while i < len(rewards):
        j = i
        while j < len(rewards) and rewards[j] == rewards[i]:
            j += 1
        out.append(f"{rewards[i]} x{j - i}" if j - i > 1 else f"{rewards[i]}")
        i = j
    return ", ".join(out)


def outcome_text(outcome):
    return {"win": "Pacman emerges victorious!", "lose": "Pacman died!", "timeout": "Out of time."}[outcome]
