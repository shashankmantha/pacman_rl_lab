# Lab: Pac-Man Learns to Play with Reinforcement Learning

COMP 469 Introduction to Artificial Intelligence, CSU Channel Islands
Textbook: Russell & Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed., **Chapter 22**

**Points:** 100 (10 questions: 69 checked by the autograder, 31 for written answers)  **Time:** about 90 minutes
**Submit:** the five `part*.py` files (zip the whole folder if that is easier)

In lecture you watched the *Q-Learning Pacman, Tricky, Watch All* demo: Pac-Man walked into the ghost game after game (`Pacman died! Score: -505`), and a few hundred games later it was winning. Nobody told it what a ghost is. It learned from the score alone. In this lab you write the code that makes that happen, watch it learn in a game window, and then push it further than the demo did.

## Setup (about 5 minutes, before the lab)

You need Python 3.9 or newer with Tkinter (the standard GUI library). No other packages, so there is nothing to `pip install`.

### Step 1. Unzip and open a terminal in the lab folder

Unzip `pacman_rl_lab.zip`. You get a folder `pacman_rl_lab` with `README.md`, `autograder.py`, five `part*.py` files and a `pacman` folder. Open a terminal **inside that folder**:

| System | How |
|---|---|
| Windows | Open the folder in File Explorer, right-click an empty spot, choose **Open in Terminal**. (Or click the address bar, type `cmd`, press Enter.) |
| macOS | Open **Terminal**, type `cd ` (with a space after it), drag the `pacman_rl_lab` folder into the Terminal window, press Return. |
| Linux | Right-click inside the folder, choose **Open in Terminal**. Or: `cd ~/Downloads/pacman_rl_lab` |

Check that you are in the right place: `dir` (Windows) or `ls` (macOS, Linux) must list `autograder.py` and `part1_returns.py`. **Run every command in this README from this folder.** If you unzipped into a folder that contains another `pacman_rl_lab` folder, `cd pacman_rl_lab` once more.

### Step 2. Find your Python command

| System | Type this | Check the version |
|---|---|---|
| Windows | `py` (or `python`) | `py --version` |
| macOS, Linux | `python3` | `python3 --version` |

The version must be 3.9 or newer. **This README writes `python`; type `py` or `python3` instead if that is your command.** For example, on a Mac `python autograder.py` becomes `python3 autograder.py`.

- Windows: if `py` is not found or `python` opens the Microsoft Store, Python is not installed. Install it from [python.org](https://www.python.org/downloads/) and tick **Add python.exe to PATH** in the installer.
- macOS: install Python from [python.org](https://www.python.org/downloads/). The `python3` that comes with Apple's developer tools has an old Tkinter that can show a blank window.

### Step 3. Check that Tkinter works

Run `python -m tkinter` (`py -m tkinter` on Windows, `python3 -m tkinter` on macOS and Linux). A small window should open; close it. If you get an error instead:

| System | Install Tkinter |
|---|---|
| Windows, or macOS with the python.org installer | Already included. If it is missing, rerun the installer, choose **Modify**, tick **tcl/tk and IDLE** |
| macOS with Homebrew Python | `brew install python-tk` |
| Ubuntu / Debian / Kali | `sudo apt install python3-tk` |

### Step 4. Do not run the lab inside a virtual environment

The lab is meant to be watched in the **game window** (a GUI). Whether you get it depends on how you run Python:

| How you run it | What you get |
|---|---|
| **Not in a virtual environment** (normal terminal) | The game window opens. **This is the way to run the lab.** |
| **Inside a virtual environment** (your prompt starts with `(.venv)` or similar) | Usually **no game window**: the part stops with `Could not open the game window`. A venv often uses a Python without Tkinter. |

If your prompt starts with `(.venv)`, type `deactivate` and run the part again. In VS Code, the **Run** button uses the interpreter shown in the bottom-right corner, which may be a venv; typing the commands in the terminal avoids that.

(If you really want to use a venv: Tkinter must be installed for the venv's Python version. With the venv active, run `python --version`; if it says 3.13, install `sudo apt install python3.13-tk`.)

### Step 5. Check that everything is ready

```
python part1_returns.py
python autograder.py
```

Before you write any code this is what you should see:

- `part1_returns.py`: the game window may flash open, then the terminal says `Not written yet: discounted_return() in part1_returns.py, line 72`. That is expected: it means Python, Tkinter and the lab files all work, and it tells you where to write your first code.
- `autograder.py`: a list of unchecked boxes ending in `Auto-graded points: 0/69`.

If instead a part stops with `Could not open the game window`, Tkinter is missing for the Python you used. The message prints the exact command to install it; do that and run the part again.

## How to work

Do the parts in order. Each part is one file. At the top of each file are the notes and formulas for that part; below them is the code you write (look for `raise NotImplementedError`), then the written questions, then a demo you do not need to touch.

| Part | File | AIMA 4e | Questions | Points | Time |
|---|---|---|---|---|---|
| 1 | `part1_returns.py` | §22.1 rewards, returns, Bellman equations | Q1 | 6 | 8 min |
| 2 | `part2_td_learning.py` | §22.2 passive RL, TD learning | Q2 | 10 | 12 min |
| 3 | `part3_qlearning.py` | §22.3 Q-learning, ε-greedy | Q3 to Q6 | 36 | 28 min |
| 4 | `part4_exploration_sarsa.py` | §22.3 exploration functions, SARSA | Q7, Q8 | 20 | 20 min |
| 5 | `part5_approximate.py` | §22.4 approximate Q-learning, features | Q9, Q10 | 28 | 22 min |

For each part:

1. **Read** the notes at the top of the file.
2. **Write** the code, then **check** it:
   ```
   python autograder.py -p 3        # all questions in Part 3
   python autograder.py -q q5       # one question
   ```
3. **Watch** it learn: `python part3_qlearning.py`
4. **Answer** the written questions in the `ANSWER_Q...` strings in the same file, using the numbers from **your** run.
5. Move on to the next part. Finished parts keep working: each file is checked on its own. Parts 4 and 5 build on your `QLearningAgent` from Part 3, so finish Part 3 before starting them.

`python autograder.py` with no options checks everything, prints your auto-graded total, and says which written answers you have filled in.

## The game window

Each demo opens a window like the lecture demo. Under each board is a chart of the score per game (moving average) and the agent's statistics.

**Value overlay** (on by default): every square is coloured by what the agent has learned that square is worth if Pac-Man stood there, with the ghosts and pellets where they are now. Green is good, red is bad, black means nothing learned yet. The number is that value, and the arrow is the move the agent would make. Watch the colours change as it learns.

**Guide** (on by default): the strip under the boards explains what is on screen, in three columns:

- **What you see:** the colour scale, what the number and arrow in each square mean, the dashed path and red circles, the chart and the statistics.
- **Settings in the board titles:** what α (alpha, learning rate), ε (epsilon, exploration rate), γ (gamma, discount), R+ and N_e mean, with their values in this run.
- **The math:** the formula the agent is using (return, TD update, Q-learning, SARSA, exploration function or approximate Q-learning), with each symbol explained.

Once you know it, press G to hide the guide and make the window smaller. On a small screen (for example a 1366x768 laptop) the guide starts hidden so the window fits; press G to show it.

| Button | Key | What it does |
|---|---|---|
| Pause / Resume | Space | |
| Slower / Faster | Left / Right | Speed of moves, and how long a fast-forwarded phase takes |
| Fast-forward / Watch every move | F | Like the lecture's "Silent train" vs "Watch All" |
| Values: on/off | V | The value overlay |
| Guide: on/off | G | The explanation strip under the boards |
| Skip phase >> | S | Finish the current phase at full speed |
| Quit | Q or Esc | |

The numbers printed in the terminal do not depend on what you do in the window: watching, fast-forwarding, skipping or closing the window gives the same results.

Command-line options for every part:

```
python part3_qlearning.py --speed 5        # start faster (0 = very slow ... 7 = no delay)
python part3_qlearning.py --no-graphics    # no window: just print the results (a few seconds)
python part3_qlearning.py --text           # draw the board in the terminal
```

Parts 4 and 5 run several experiments; you can run one at a time:

```
python part4_exploration_sarsa.py explore   # Q7
python part4_exploration_sarsa.py cliff     # Q8
python part5_approximate.py table           # Q9
python part5_approximate.py deaths          # Q10 (a)
python part5_approximate.py compare         # Q10 (b) and (c)
```

## Files

| File | What it is | Edit? |
|---|---|---|
| `part1_returns.py` ... `part5_approximate.py` | Notes, your code and your written answers | **Yes, these are what you submit** |
| `autograder.py` | Checks your code | No |
| `pacman/` | The game, the window and the tests | No (but `pacman/engine.py` is worth a look in Part 5) |

## Grading

| Question | Part | Auto-graded | Written | Total |
|---|---|---|---|---|
| Q1 Discounted return | 1 | 6 | | 6 |
| Q2 TD learning | 2 | 7 | 3 | 10 |
| Q3 Values and the greedy action | 3 | 8 | | 8 |
| Q4 ε-greedy | 3 | 6 | | 6 |
| Q5 Q-learning update | 3 | 12 | | 12 |
| Q6 Watch Pac-Man learn | 3 | 4 | 6 | 10 |
| Q7 Exploration function | 4 | 6 | 4 | 10 |
| Q8 SARSA on the ghost cliff | 4 | 4 | 6 | 10 |
| Q9 Approximate Q-learning | 5 | 10 | 4 | 14 |
| Q10 Engineer a better Pac-Man | 5 | 6 | 8 | 14 |
| **Total** | | **69** | **31** | **100** |

Your instructor grades with an untouched copy of the autograder, so editing it does not change your grade. Written answers: 2 to 5 sentences per part, and full credit needs the reason **and** a number from your own output.

## Debugging tips

- `Not written yet: get_value() in part3_qlearning.py, line 96` means that function still has `raise NotImplementedError`.
- Run one question at a time: `python autograder.py -q q3`. A failed check shows the file and line where your code raised an error.
- The demos and the autograder seed the random number generators, so the same code always gives the same numbers.
- Pac-Man never learns on 'tricky' in Part 3: check that `get_value` returns negative values when it should, and that `update` uses `next_state`, not `state`.

## Going further (optional, not graded)

- **Decaying exploration.** In Part 3, multiply `agent.epsilon` by 0.999 after every game. Does the training curve end higher? Is that schedule GLIE?
- **Reward shaping.** Add γ·Φ(s') − Φ(s) to every reward, with Φ(s) = −(distance to the nearest pellet). Does the table agent learn 'tricky' faster? Then try a bonus that is *not* of that form (for example +2 whenever Pac-Man gets closer to food) and look for strange behaviour.
- **Approximate SARSA.** Combine Parts 4 and 5. Is it more cautious on 'twoGhosts'?
- **Your own board.** `PacmanEnv(layout)` accepts any layout string: `%` wall, `.` pellet, `P` Pac-Man, `G` ghost, `X` sleeping ghost.
