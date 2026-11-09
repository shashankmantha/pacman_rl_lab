"""
The game window (Tkinter, part of the Python standard library) and a text-mode fallback.

Window controls
    Space          pause / resume
    Left / Right   slower / faster
    F              fast-forward (show only a few frames, train at full speed) / watch every move
    V              value overlay on / off: each square is coloured by what the agent has
                   learned it is worth if Pac-Man stood there (green good, red bad), with an
                   arrow for the move it would take
    G              guide on / off: the strip under the boards that explains the colours,
                   the settings (alpha, epsilon, gamma, ...) and the update formula
    S              skip to the end of this phase
    Q or Esc       quit
"""
import math
import os
import random
import time

from .engine import State, DIRS

SPEEDS = [1.0, 0.5, 0.3, 0.15, 0.08, 0.04, 0.02, 0.0]
SPEED_NAMES = ["very slow", "slow", "relaxed", "normal", "quick", "fast", "very fast", "no delay"]
FAST_SECONDS = [40, 25, 15, 10, 6, 3, 1.5, 0]     # how long a fast-forwarded phase takes at each speed
DEFAULT_SPEED = 3
HEADING = {"East": 35, "North": 125, "West": 215, "South": 305, None: 35}

UI_BG, UI_FG, UI_DIM = "#202028", "#ececf1", "#9a9aa8"
GUIDE_BG = "#2a2a34"
CHART_H = 100
WALL, FOOD, GHOST, SLEEPER, PACMAN = "#3a3aff", "#ffffff", "#3355ff", "#7a7a99", "#ffff00"


# ---------------------------------------------------------------------------
# What the agent has learned, square by square
# ---------------------------------------------------------------------------

def _q_view(agent, state, actions):
    """Q-values without side effects (reading a defaultdict with [] would add entries)."""
    if isinstance(getattr(agent, "weights", None), dict):
        return {a: agent.get_q_value(state, a) for a in actions}
    table = getattr(agent, "q_values", None)
    if isinstance(table, dict):
        return {a: table.get((state, a), 0.0) for a in actions}
    return {a: agent.get_q_value(state, a) for a in actions}


def overlay_values(lane, state):
    """{square: (value, best_action or None)} for every square Pac-Man could stand on,
    with the ghosts and pellets as they are in `state`."""
    env, out = lane.env, {}
    saved = random.getstate()
    try:
        for pos in env.open_squares():
            if pos in env.sleepers or pos in state.ghosts:
                continue
            s = State(pos, state.ghosts, state.food - {pos})
            if env.is_terminal(s):
                continue
            if lane.overlay == "u":
                if lane.table is not None:
                    out[pos] = (lane.table.get(s, 0.0), None)
                continue
            actions = env.legal_actions(s)
            qs = _q_view(lane.agent, s, actions)
            best = max(actions, key=qs.get)
            out[pos] = (qs[best], best if len(set(qs.values())) > 1 else None)
    except Exception as e:  # unfinished student code: just skip the overlay
        lane.message = f"(value overlay off: {type(e).__name__})"
        return {}
    finally:
        random.setstate(saved)
    return out


def value_color(v, m):
    t = min(1.0, abs(v) / m) ** 0.6 if m > 0 else 0.0
    if v >= 0:
        r, g, b = 12, int(28 + 140 * t), int(12 + 30 * t)
    else:
        r, g, b = int(28 + 160 * t), 12, 12
    return f"#{r:02x}{g:02x}{b:02x}"


def fmt_value(v):
    return f"{v:.0f}" if abs(v) >= 9.95 else f"{v:.1f}"


# ---------------------------------------------------------------------------
# The guide under the boards: board key, settings, update rule
# ---------------------------------------------------------------------------

def agent_kind(lane):
    a = lane.agent
    if lane.overlay == "u":
        return "td"
    if hasattr(a, "sarsa_target"):
        return "sarsa"
    if isinstance(getattr(a, "weights", None), dict):
        return "approx"
    if hasattr(a, "exploration_value"):
        return "explore"
    if isinstance(getattr(a, "q_values", None), dict):
        return "q"
    return "random"


UPDATE_RULES = {
    "random": ("Return: the number Part 1 computes",
               "G = r\u2081 + \u03b3\u00b7r\u2082 + \u03b3\u00b2\u00b7r\u2083 + \u2026",
               "Rewards: \u22121 per move, +10 per pellet, +500 more for the last pellet (win), \u2212500 for "
               "touching a ghost. SCORE adds them up (\u03b3 = 1). The return G multiplies each later reward "
               "by one more \u03b3, so rewards that come sooner matter more."),
    "td": ("TD update, after every move",
           "U(s) \u2190 U(s) + \u03b1\u00b7[ sample \u2212 U(s) ]\n"
           "sample = r + \u03b3\u00b7U(s\u2032)",
           "s = square before the move, s\u2032 = square after, r = reward for the move. The sample is "
           "the reward plus the discounted value of where Pac-Man landed; sample \u2212 U(s) is how wrong "
           "the old guess was, and \u03b1 says how much of that error to fix. Good news from the pellet "
           "moves back one square per game."),
    "q": ("Q-learning update, after every move",
          "Q(s,a) \u2190 Q(s,a) + \u03b1\u00b7[ sample \u2212 Q(s,a) ]\n"
          "sample = r + \u03b3\u00b7max\u2090\u2032 Q(s\u2032,a\u2032)",
          "Q(s,a) = how good move a is in state s (state = where Pac-Man, the ghosts and the pellets are). "
          "The sample uses the best next move a\u2032 (max) even when Pac-Man then explores: off-policy."),
    "sarsa": ("SARSA update, after every move",
              "Q(s,a) \u2190 Q(s,a) + \u03b1\u00b7[ sample \u2212 Q(s,a) ]\n"
              "sample = r + \u03b3\u00b7Q(s\u2032,a\u2032)",
              "a\u2032 = the move Pac-Man actually takes next, random ones included, so SARSA learns how "
              "it really plays: on-policy. That makes it more careful near ghosts."),
    "explore": ("Exploration function",
                "move = argmax\u2090 f(Q(s,a), N(s,a))\n"
                "f(u, n) = R+ if n < N_e, else u",
                "N(s,a) counts how often move a was tried in state s. Untried moves look worth R+, so "
                "Pac-Man tries each one N_e times before trusting Q. Q is updated as in Q-learning."),
    "approx": ("Approximate Q-learning",
               "Q(s,a) = \u03a3\u1d62 w\u1d62\u00b7f\u1d62(s,a)\n"
               "w\u1d62 \u2190 w\u1d62 + \u03b1\u00b7[ sample \u2212 Q(s,a) ]\u00b7f\u1d62(s,a)\n"
               "sample = r + \u03b3\u00b7max\u2090\u2032 Q(s\u2032,a\u2032)",
               "f\u1d62(s,a) = features, numbers that describe the situation (for example 'ghost one step "
               "away'); w\u1d62 = their weights, listed under the board. No table: what Pac-Man learns in "
               "one square carries over to every similar square."),
}

PARAMS = (
    ("alpha", "α (alpha), learning rate",
     "how far each update moves toward the new sample: new = old + α·(sample − old) "
     "(0 = never learns, 1 = forgets the old value). Small α: slow, steady learning. Large α: fast but jumpy."),
    ("epsilon", "ε (epsilon), exploration rate",
     "chance of a random move instead of the best-known one. "
     "Random moves let Pac-Man discover better options."),
    ("gamma", "γ (gamma), discount",
     "a reward t moves away counts γᵗ as much, so sooner rewards matter more."),
    ("r_plus", "R+", "optimistic value given to moves not tried enough yet."),
    ("n_e", "N_e", "tries needed before Pac-Man trusts a move's estimate."),
)


def settings_items(lanes):
    """[(heading, explanation)] for the settings used by the agents on screen."""
    learners = [l for l in lanes if agent_kind(l) != "random"]
    if not learners:
        return [("No settings", "This Pac-Man picks every move at random and never learns. "
                                "epsilon 1.00 under the board means every move is random.")]
    out = []
    for attr, name, meaning in PARAMS:
        users = [l for l in learners if hasattr(l.agent, attr)
                 and not (attr == "epsilon" and agent_kind(l) in ("td", "explore"))]
        if not users:
            continue
        vals = {getattr(l.agent, attr) for l in users}
        if len(vals) == 1:
            v = vals.pop()
            shown = f" = {v:g}"
            extra = ""
            if attr == "gamma" and 0 < v < 1:
                extra = f" Here a reward 10 moves away counts {v:g}\u00b9\u2070 \u2248 {v ** 10:.2f} as much."
            elif attr == "alpha" and 0 < v <= 1:
                extra = (f" Here: old 100, sample 300 \u2192 100 + {v:g}\u00b7(300 \u2212 100) = {100 + 200 * v:g}, "
                         f"{v:.0%} of the way.")
            elif attr == "epsilon" and 0 < v < 1:
                extra = f" Here about 1 move in {round(1 / v)} is random."
        else:
            shown, extra = " (differs per board, see the titles)", ""
        out.append((f"{name}{shown}", f"{meaning[0].upper()}{meaning[1:]}{extra}"))
    if any(agent_kind(l) == "approx" for l in learners):
        out.append(("wᵢ weights (listed under the board)",
                    "How much each feature adds to Q. Learned instead of a table."))
    return out


def board_key_text(lanes, show_values):
    sleepers = any(l.env.sleepers for l in lanes)
    lines = ["Yellow = Pac-Man, blue = ghost, " + ("grey = sleeping ghost (never moves), " if sleepers else "")
             + "white dot = pellet."]
    kinds = {agent_kind(l) for l in lanes if l.overlay}
    if kinds and not show_values:
        lines.append("Values are hidden. Press V to show what Pac-Man has learned about each square.")
    elif kinds == {"td"}:
        lines.append("Number in a square = U(s): the discounted score Pac-Man expects from there "
                     "if it keeps following the policy.")
    elif kinds:
        lines.append("Number in a square = V(s) = maxₐ Q(s,a): what Pac-Man has learned standing there "
                     "is worth, with the ghosts and pellets where they are now.")
        lines.append("Arrow = the move it would choose there (the greedy action).")
    if kinds and show_values:
        lines.append("Black square / no number = nothing learned there yet.")
    if any(l.path for l in lanes):
        lines.append("Dashed yellow line = the path Pac-Man takes when it plays greedily (no random moves).")
    if any(l.deaths for l in lanes):
        lines.append("Red circles = where Pac-Man died; bigger = more deaths there.")
    learning = any(agent_kind(l) != "random" for l in lanes)
    lines.append("Chart = average score over the last 50 games" + (": rising means it is learning." if learning else "."))
    if any(agent_kind(l) in ("q", "sarsa", "explore") for l in lanes):
        lines.append("Q-table entries = (state, move) pairs Pac-Man has a value for.")
    return "\n".join(lines)


def moving_average(xs, w=50):
    out, s = [], 0.0
    for i, x in enumerate(xs):
        s += x
        if i >= w:
            s -= xs[i - w]
        out.append(s / min(i + 1, w))
    return out


def lane_stats(lane):
    a = lane.agent
    n = len(lane.scores)
    lines = [f"games played {n:5d}   this phase {lane.games_in_phase}/{lane.target}"]
    k = min(50, n)
    if k:
        recent = lane.scores[-k:]
        lines.append(f"last {k:2d}: avg score {sum(recent) / k:7.1f}  wins {lane.outcomes[-k:].count('win')}/{k}")
    else:
        lines.append("last games: none yet")
    if getattr(a, "exploring", None) is not None:
        explore = "exploration function" if a.exploring else "greedy (exploration off)"
    else:
        explore = f"epsilon {getattr(a, 'epsilon', 0.0):.2f}"
    w = getattr(a, "weights", None)
    if isinstance(w, dict) and w:
        lines.append(explore)
        items = [f"{k[:18]} {v:+.0f}" for k, v in w.items()]
        for i in range(0, len(items), 2):
            lines.append("  " + "   ".join(items[i:i + 2]))
    elif isinstance(getattr(a, "q_values", None), dict):
        lines.append(f"{explore}   Q-table {len(a.q_values):,} entries")
    elif lane.overlay == "u" and lane.table is not None:
        lines.append(f"utility table: {len(lane.table)} states")
    else:
        lines.append(explore)
    if lane.message:
        lines.append(lane.message)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tkinter window
# ---------------------------------------------------------------------------

class TkDisplay:
    def __init__(self, title="COMP 469 Pac-Man: reinforcement learning", speed=DEFAULT_SPEED):
        import tkinter as tk
        self.tk = tk
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg=UI_BG)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.closed = self.paused = self.fast = self.skipping = False
        self.show_values = True
        self.speed_i = speed
        self.panes, self.lanes_key, self.cell = [], None, 40
        self.mid_run = self.finished = False
        self.lanes = None                 # the lanes drawn last, so V can redraw them at once
        self.guide_auto_hidden = False    # hidden because the window was taller than the screen
        self.guide_user_set = False       # the user pressed G: never hide or show it for them

        self.banner = tk.Label(self.root, text="", font=("Helvetica", 13, "bold"), fg=UI_FG, bg=UI_BG,
                               justify="left", anchor="w", wraplength=1050)
        self.banner.pack(fill="x", padx=12, pady=(10, 4))
        self.pane_frame = tk.Frame(self.root, bg=UI_BG)
        self.pane_frame.pack(padx=8, pady=4)
        self.guide = tk.Frame(self.root, bg=UI_BG)
        self.guide.pack(padx=8, pady=(4, 0))
        self.key_label = self.key_scale = self.settings_box = self.settings_shown = None

        self.show_guide = True
        bar = self.bar = tk.Frame(self.root, bg=UI_BG)
        bar.pack(pady=(6, 2))
        self.b_pause = tk.Button(bar, text="Pause", width=8, command=self.toggle_pause)
        self.b_fast = tk.Button(bar, text="Fast-forward", width=16, command=self.toggle_fast)
        self.b_values = tk.Button(bar, text="Values: on", width=11, command=self.toggle_values)
        self.b_guide = tk.Button(bar, text="Guide: on", width=10, command=self.press_guide)
        for b in (self.b_pause, tk.Button(bar, text="Slower", width=7, command=self.slower),
                  tk.Button(bar, text="Faster", width=7, command=self.faster), self.b_fast, self.b_values,
                  self.b_guide, tk.Button(bar, text="Skip phase >>", width=13, command=self.skip),
                  tk.Button(bar, text="Quit", width=6, command=self.close)):
            b.pack(side="left", padx=3)
        self.status = tk.Label(self.root, text="", font=("Helvetica", 10), fg=UI_DIM, bg=UI_BG)
        self.status.pack()
        tk.Label(self.root, text="Keys: Space pause   Left/Right speed   F fast-forward   V values   G guide   "
                                 "S skip   Q quit",
                 font=("Helvetica", 9), fg=UI_DIM, bg=UI_BG).pack(pady=(0, 8))
        for keys, fn in ((("<space>",), self.toggle_pause), (("<Left>", "<minus>"), self.slower),
                         (("<Right>", "<plus>", "<equal>"), self.faster), (("f", "F"), self.toggle_fast),
                         (("v", "V"), self.toggle_values), (("g", "G"), self.press_guide), (("s", "S"), self.skip),
                         (("q", "Q", "<Escape>"), self.close)):
            for k in keys:
                self.root.bind(k, lambda e, fn=fn: fn())
        self._update_status()
        self._pump()

    # ----- controls -----
    @property
    def delay(self):
        return SPEEDS[self.speed_i]

    @property
    def fast_seconds(self):
        return FAST_SECONDS[self.speed_i]

    def toggle_pause(self):
        self.paused = not self.paused
        self._update_status()

    def slower(self):
        self.speed_i = max(0, self.speed_i - 1)
        self._update_status()

    def faster(self):
        self.speed_i = min(len(SPEEDS) - 1, self.speed_i + 1)
        self._update_status()

    def toggle_fast(self):
        self.fast = not self.fast
        self._update_status()

    def toggle_values(self):
        self.show_values = not self.show_values
        self._update_status()
        if self.lanes:          # redraw now, so it also works while paused and on the final screen
            self._render(self.lanes)

    def press_guide(self):
        self.guide_user_set, self.guide_auto_hidden = True, False
        self.toggle_guide()

    def toggle_guide(self):
        self.show_guide = not self.show_guide
        if self.show_guide:
            self.guide.pack(padx=8, pady=(4, 0), before=self.bar)
        else:
            self.guide.pack_forget()
        self._update_status()

    def _fit_screen(self):
        """Hide the guide if the window would be taller than the screen (small laptop screens)."""
        if self.guide_user_set or not self.show_guide:
            return
        self.root.update_idletasks()
        if self.root.winfo_reqheight() > self.root.winfo_screenheight() - 80:
            self.guide_auto_hidden = True
            self.toggle_guide()

    def skip(self):
        self.skipping = True
        self.paused = False
        self._update_status()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.root.destroy()
        except Exception:
            pass
        if self.mid_run:
            print("Window closed: the program keeps running without graphics (the numbers do not change).")

    def _update_status(self):
        if self.closed:
            return
        mode = ("skipping to the end of this phase" if self.skipping else
                "fast-forward (press F to watch every move)" if self.fast else "watching every move")
        fit = "   |   guide hidden to fit your screen (press G to show it)" if self.guide_auto_hidden else ""
        if self.finished:
            text = "Finished. The window stays open so you can look at it (V and G still work)." + fit
        else:
            text = (f"Speed: {SPEED_NAMES[self.speed_i]}   |   {mode}   |   "
                    f"value overlay {'on' if self.show_values else 'off'}"
                    + ("   |   PAUSED" if self.paused else "") + fit)
        self.status.config(text=text)
        self.b_pause.config(text="Resume" if self.paused else "Pause")
        self.b_fast.config(text="Watch every move" if self.fast else "Fast-forward")
        self.b_values.config(text=f"Values: {'on' if self.show_values else 'off'}")
        self.b_guide.config(text=f"Guide: {'on' if self.show_guide else 'off'}")

    def _pump(self):
        if self.closed:
            return
        try:
            self.root.update()
        except self.tk.TclError:
            self.closed = True

    def wait(self, seconds):
        end = time.time() + seconds
        while True:
            self._pump()
            left = end - time.time()
            if self.closed or left <= 0:
                return
            time.sleep(min(0.02, left))

    # ----- phases -----
    def begin_phase(self, lanes, banner, watch=True, speed=None):
        if self.closed:
            return
        self.mid_run = True
        key = tuple(id(l) for l in lanes)
        if key != self.lanes_key:
            self._build(lanes)
            self.lanes_key = key
        self.banner.config(text=banner)
        self.fast, self.skipping, self.paused = not watch, False, False
        if speed is not None:
            self.speed_i = speed
        self._update_status()
        self.draw(lanes)

    def hold(self, message="Done. Close the window (or press Q) to exit."):
        """Keep the window open until it is closed."""
        if self.closed:
            return
        self.mid_run = False
        self.finished, self.paused, self.fast, self.skipping = True, False, False, False
        self.banner.config(text=message)
        self._update_status()
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            pass
        self.closed = True

    def _build(self, lanes):
        tk = self.tk
        for w in self.pane_frame.winfo_children():
            w.destroy()
        n = len(lanes)
        self.cell = min(max(22, min(52, 1000 // (n * l.env.width), 270 // l.env.height)) for l in lanes)
        self.panes = []
        for i, lane in enumerate(lanes):
            env, c = lane.env, self.cell
            f = tk.Frame(self.pane_frame, bg=UI_BG)
            f.grid(row=0, column=i, padx=10, sticky="n")
            tk.Label(f, text=lane.name, font=("Helvetica", 12, "bold"), fg=UI_FG, bg=UI_BG).pack()
            bw, bh = c * env.width, c * env.height + 34
            board = tk.Canvas(f, width=bw, height=bh, bg="black", highlightthickness=0)
            board.pack()
            cw = max(bw, 270)
            chart = tk.Canvas(f, width=cw, height=CHART_H, bg="#111118", highlightthickness=0)
            chart.pack(pady=(6, 2))
            stats = tk.Label(f, text="", font=("Courier", 10), fg=UI_FG, bg=UI_BG, justify="left", anchor="w")
            stats.pack(fill="x")
            self._draw_walls(board, env, c)
            self.panes.append({"board": board, "chart": chart, "stats": stats, "cw": cw})
        self._build_guide(lanes)
        self._fit_screen()
        self._pump()

    def _build_guide(self, lanes):
        tk = self.tk
        for w in self.guide.winfo_children():
            w.destroy()
        col_w = 340

        def column(i, title):
            f = tk.Frame(self.guide, bg=GUIDE_BG, padx=10, pady=6)
            f.grid(row=0, column=i, sticky="nsew", padx=3)
            tk.Label(f, text=title, font=("Helvetica", 11, "bold"), fg=UI_FG, bg=GUIDE_BG,
                     anchor="w").pack(fill="x")
            return f

        def body(parent, text="", font=("Helvetica", 10), fg=UI_FG, wrap=col_w):
            lab = tk.Label(parent, text=text, font=font, fg=fg, bg=GUIDE_BG, justify="left", anchor="w",
                           wraplength=wrap)
            lab.pack(fill="x", pady=(2, 0))
            return lab

        key = column(0, "What you see")
        self.key_scale = None
        if any(l.overlay for l in lanes):
            self.key_scale = tk.Canvas(key, width=col_w, height=28, bg=GUIDE_BG, highlightthickness=0)
            self.key_scale.pack(anchor="w", pady=(2, 0))
            self._draw_scale(self.key_scale, col_w)
        self.key_label = body(key)

        self.settings_box = tk.Frame(column(1, "Settings in the board titles"), bg=GUIDE_BG)
        self.settings_box.pack(fill="x")
        self.settings_shown = None

        rules = column(2, "The math")
        seen, eps_shown = [], False
        for lane in lanes:
            kind = agent_kind(lane)
            if kind in seen:
                continue
            seen.append(kind)
            title, formula, meaning = UPDATE_RULES[kind]
            body(rules, title, font=("Helvetica", 10, "bold"), fg=UI_DIM)
            body(rules, formula, font=("Courier", 10), fg="#ffee66", wrap=0)
            body(rules, meaning)
            if kind == "td" and hasattr(lane.agent, "alpha") and hasattr(lane.agent, "gamma"):
                a, g = lane.agent.alpha, lane.agent.gamma
                body(rules, "Example: game 1, the square next to the pellet", font=("Helvetica", 10, "bold"),
                     fg=UI_DIM)
                body(rules, f"sample = 509 + {g:g}\u00b70 = 509   (\u22121 move, +10 pellet, +500 win)\n"
                            f"U(s)   \u2190 0 + {a:g}\u00b7[ 509 \u2212 0 ] = {a * 509:g}",
                     font=("Courier", 10), fg="#ffee66", wrap=0)
            if kind in ("q", "sarsa", "approx") and hasattr(lane.agent, "epsilon") and not eps_shown:
                eps_shown = True
                body(rules, "Choosing a move (ε-greedy):", font=("Helvetica", 10, "bold"), fg=UI_DIM)
                body(rules, "move = random,         with probability ε\n"
                            "move = argmaxₐ Q(s,a), otherwise",
                     font=("Courier", 10), fg="#ffee66", wrap=0)

    def _fill_settings(self, items):
        tk = self.tk
        for w in self.settings_box.winfo_children():
            w.destroy()
        for i, (heading, text) in enumerate(items):
            tk.Label(self.settings_box, text=heading, font=("Helvetica", 10, "bold"), fg="#ffee66", bg=GUIDE_BG,
                     anchor="w", justify="left").pack(fill="x", pady=(6 if i else 2, 0))
            tk.Label(self.settings_box, text=text, font=("Helvetica", 10), fg=UI_FG, bg=GUIDE_BG, anchor="w",
                     justify="left", wraplength=340).pack(fill="x")
        self.settings_shown = items

    def _draw_scale(self, cv, w):
        n, L, R = 60, 2, 2
        bw = (w - L - R) / n
        for i in range(n):
            v = -1.0 + 2.0 * (i + 0.5) / n
            cv.create_rectangle(L + i * bw, 2, L + (i + 1) * bw + 1, 13, fill=value_color(v, 1.0), outline="")
        for x, text, anchor in ((L, "bad (negative)", "w"), (w / 2, "0", "center"),
                                (w - R, "good (positive)", "e")):
            cv.create_text(x, 21, text=text, anchor=anchor, fill=UI_DIM, font=("Helvetica", 8))

    # ----- drawing -----
    def _draw_walls(self, cv, env, c):
        for (x, y) in env.walls:
            for (dx, dy) in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                nx, ny = x + dx, y + dy
                if (nx, ny) in env.walls or not (0 <= nx < env.width and 0 <= ny < env.height):
                    continue
                if dx == 0:
                    yy = (y + (1 if dy > 0 else 0)) * c
                    cv.create_line(x * c, yy, (x + 1) * c, yy, fill=WALL, width=2)
                else:
                    xx = (x + (1 if dx > 0 else 0)) * c
                    cv.create_line(xx, y * c, xx, (y + 1) * c, fill=WALL, width=2)

    def _ghost(self, cv, x, y, c, color):
        cx, cy, r = (x + 0.5) * c, (y + 0.5) * c, 0.38 * c
        pts = []
        for i in range(13):
            t = math.pi - i * math.pi / 12
            pts += [cx + r * math.cos(t), cy - 0.05 * c - r * math.sin(t)]
        pts += [cx + r, cy + 0.38 * c, cx + r / 2, cy + 0.26 * c, cx, cy + 0.38 * c,
                cx - r / 2, cy + 0.26 * c, cx - r, cy + 0.38 * c]
        cv.create_polygon(pts, fill=color, outline="", tags="dyn")
        for ex in (-0.14, 0.14):
            ox, oy = cx + ex * c, cy - 0.08 * c
            cv.create_oval(ox - 0.11 * c, oy - 0.11 * c, ox + 0.11 * c, oy + 0.11 * c, fill="white", outline="", tags="dyn")
            cv.create_oval(ox - 0.02 * c, oy - 0.05 * c, ox + 0.08 * c, oy + 0.05 * c, fill="#1a1aff", outline="", tags="dyn")

    def _draw_board(self, pane, lane):
        cv, c, env, st = pane["board"], self.cell, lane.env, lane.view_state
        cv.delete("dyn")
        font = ("Helvetica", max(7, c // 5))
        if self.show_values and lane.overlay:
            vals = overlay_values(lane, st)
            m = max((abs(v) for v, _ in vals.values()), default=0.0)
            for (x, y), (v, best) in vals.items():
                if v == 0 and best is None:      # nothing learned here yet
                    continue
                cv.create_rectangle(x * c + 2, y * c + 2, (x + 1) * c - 2, (y + 1) * c - 2,
                                    fill=value_color(v, m), outline="", tags="dyn")
                ty = (y + 0.82) * c if best == "North" else (y + 0.2) * c
                if c >= 30:
                    ink = "#000000" if (x, y) == st.pacman and lane.deaths is None else "#d8d8d8"
                    cv.create_text((x + 0.5) * c, ty, text=fmt_value(v), fill=ink, font=font,
                                   tags=("dyn", "valtext"))
                if best:
                    dx, dy = DIRS[best]
                    cx, cy = (x + 0.5) * c, (y + 0.5) * c
                    cv.create_line(cx + dx * 0.12 * c, cy + dy * 0.12 * c, cx + dx * 0.36 * c, cy + dy * 0.36 * c,
                                   fill="#bdbdbd", width=2, arrow="last", arrowshape=(6, 7, 3), tags="dyn")
        if lane.path:
            pts = [v for (x, y) in lane.path for v in ((x + 0.5) * c, (y + 0.5) * c)]
            if len(pts) >= 4:
                cv.create_line(*pts, fill="#ffcc00", width=3, dash=(6, 4), tags="dyn")
        if lane.deaths:
            top = max(lane.deaths.values())
            for (x, y), k in lane.deaths.items():
                r = (0.15 + 0.3 * k / top) * c
                cv.create_oval((x + 0.5) * c - r, (y + 0.5) * c - r, (x + 0.5) * c + r, (y + 0.5) * c + r,
                               fill="#ff3b3b", outline="", tags="dyn")
        for (x, y) in st.food:
            r = 0.1 * c
            cv.create_oval((x + 0.5) * c - r, (y + 0.5) * c - r, (x + 0.5) * c + r, (y + 0.5) * c + r,
                           fill=FOOD, outline="", tags="dyn")
        for (x, y) in env.sleepers:
            self._ghost(cv, x, y, c, SLEEPER)
        for (x, y) in st.ghosts:
            self._ghost(cv, x, y, c, GHOST)
        if lane.deaths is None:
            px, py = st.pacman
            r = 0.4 * c
            cv.create_arc((px + 0.5) * c - r, (py + 0.5) * c - r, (px + 0.5) * c + r, (py + 0.5) * c + r,
                          start=HEADING.get(lane.last_action, 35), extent=290, fill=PACMAN, outline="", tags="dyn")
        cv.tag_raise("valtext")
        if lane.caption:
            label = lane.caption
        else:
            label = f"SCORE: {env.score}"
            if env.is_terminal(st):
                label += "   WIN" if env.is_win(st) else "   DIED"
        cv.create_text(env.width * c / 2, env.height * c + 17, text=label, fill="#ffee66",
                       font=("Times", 16 if not lane.caption else 13, "bold"), tags="dyn")

    def _draw_chart(self, pane, lane, lo, hi):
        cv, W, H = pane["chart"], pane["cw"], CHART_H
        cv.delete("all")
        L, R, T, B = 40, 8, 26, 16
        cv.create_text(6, 9, text="score per game (average of last 50)", anchor="w", fill=UI_DIM,
                       font=("Helvetica", 9))
        ymap = lambda v: T + (hi - v) / (hi - lo) * (H - T - B)
        for v in (500, 0, -500):
            if lo <= v <= hi:
                cv.create_line(L, ymap(v), W - R, ymap(v), fill="#33333d" if v else "#55555f")
                cv.create_text(L - 4, ymap(v), text=f"{v:+d}" if v else "0", anchor="e", fill=UI_DIM,
                               font=("Helvetica", 8))
        n = len(lane.scores)
        if n == 0:
            cv.create_text(W / 2, H / 2, text="no games yet", fill=UI_DIM, font=("Helvetica", 9))
            return
        avg = moving_average(lane.scores)
        step = max(1, n // (W - L - R))
        idx = list(range(0, n, step))
        if idx[-1] != n - 1:
            idx.append(n - 1)
        xmap = lambda i: L + (i / max(1, n - 1)) * (W - L - R)
        pts = [v for i in idx for v in (xmap(i), ymap(max(lo, min(hi, avg[i]))))]
        if len(pts) >= 4:
            cv.create_line(*pts, fill="#ffd400", width=2)
        else:
            cv.create_oval(pts[0] - 2, pts[1] - 2, pts[0] + 2, pts[1] + 2, fill="#ffd400", outline="")
        cv.create_text(W - R, H - 7, text=f"{n} game{'s' if n != 1 else ''}   latest avg {avg[-1]:.0f}",
                       anchor="e", fill=UI_DIM,
                       font=("Helvetica", 8))

    def draw(self, lanes):
        if self.closed:
            return
        self._render(lanes)
        self._pump()

    def _render(self, lanes):
        self.lanes = lanes
        hi =max([520.0] + [max(moving_average(l.scores)) + 30 for l in lanes if l.scores])
        for pane, lane in zip(self.panes, lanes):
            self._draw_board(pane, lane)
            self._draw_chart(pane, lane, -540.0, hi)
            pane["stats"].config(text=lane_stats(lane))
        text = board_key_text(lanes, self.show_values)
        if self.key_label.cget("text") != text:
            self.key_label.config(text=text)
        items = settings_items(lanes)
        if items != self.settings_shown:
            self._fill_settings(items)
        if self.key_scale is not None:
            self.key_scale.itemconfigure("all", state="normal" if self.show_values else "hidden")


# ---------------------------------------------------------------------------
# Text mode (no Tkinter, or --text)
# ---------------------------------------------------------------------------

class TextDisplay:
    closed = False
    paused = False

    def __init__(self, speed=DEFAULT_SPEED):
        self.speed_i, self.fast, self.skipping = speed, False, False
        self.show_values = True
        if os.name == "nt":
            os.system("")      # turn on ANSI escape codes in the Windows console

    @property
    def delay(self):
        return max(SPEEDS[self.speed_i], 0.05)

    fast_seconds = 0

    def begin_phase(self, lanes, banner, watch=True, speed=None):
        self.fast, self.skipping = not watch, False
        if speed is not None:
            self.speed_i = speed
        print("\n" + banner + ("\n(press Ctrl+C to fast-forward this phase)" if watch else ""), flush=True)
        self.banner = banner

    def hold(self, message="Done."):
        print(message)

    def wait(self, seconds):
        try:
            time.sleep(seconds)
        except KeyboardInterrupt:
            self.fast = True

    def _render(self, lane):
        env, st = lane.env, lane.view_state
        vals = overlay_values(lane, st) if lane.overlay else {}
        rows, vrows = [], []
        for y in range(env.height):
            row, vrow = "", ""
            for x in range(env.width):
                p = (x, y)
                ch = ("#" if p in env.walls else "G" if p in st.ghosts else "X" if p in env.sleepers else
                      "P" if p == st.pacman else "." if p in st.food else " ")
                row += ch
                if vals and env.width <= 13:
                    vrow += f"{fmt_value(vals[p][0]):>5}" if p in vals else ("  ###" if p in env.walls else "     ")
            rows.append(row)
            vrows.append(vrow)
        out = [lane.name, *rows, f"SCORE: {env.score}"]
        if any(vrows):
            out += ["values Pac-Man has learned for each square:", *vrows]
        return out

    def draw(self, lanes):
        if self.fast or self.skipping:
            return
        blocks = [self._render(l) for l in lanes]
        width = [max(len(s) for s in b) + 4 for b in blocks]
        lines = []
        for i in range(max(len(b) for b in blocks)):
            lines.append("".join((b[i] if i < len(b) else "").ljust(w) for b, w in zip(blocks, width)))
        print("\033[2J\033[H" + self.banner + "\n\n" + "\n".join(lines), flush=True)
