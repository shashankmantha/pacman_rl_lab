"""Command-line options shared by the five parts, and friendly error messages."""
import argparse
import os
import sys
import traceback

from .display import DEFAULT_SPEED, TextDisplay


def setup(description, add_arguments=None):
    """Parse the command line and open the game window. Returns the parsed arguments;
    args.display is the window (or None with --no-graphics)."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--no-graphics", action="store_true",
                   help="no window: just train and print the results (fastest)")
    p.add_argument("--text", action="store_true",
                   help="draw the board as text in the terminal instead of opening a window")
    p.add_argument("--speed", type=int, default=DEFAULT_SPEED, choices=range(8), metavar="0-7",
                   help=f"starting speed, 0 = very slow ... 7 = no delay (default {DEFAULT_SPEED})")
    if add_arguments:
        add_arguments(p)
    args = p.parse_args()
    args.display = make_display(args)
    return args


def make_display(args):
    if args.no_graphics:
        return None
    if not args.text:
        try:
            from .display import TkDisplay
            return TkDisplay(speed=args.speed)
        except Exception as e:  # no Tkinter installed, or no screen
            _explain_no_window(e)
            sys.exit(1)
    return TextDisplay(speed=args.speed)


def _explain_no_window(e):
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    print(f"\nCould not open the game window ({type(e).__name__}: {e}).\n")
    if isinstance(e, ImportError):
        print(f"Tkinter is not installed for the Python you are using: Python {ver} ({sys.executable}).")
        print("Install it, then run this file again:")
        print(f"  Ubuntu / Debian / Kali:   sudo apt install python{ver}-tk")
        print(f"  macOS with Homebrew:      brew install python-tk@{ver}")
        print("  Windows / python.org:     rerun the Python installer, choose Modify, tick 'tcl/tk and IDLE'")
        print(f"Check it works: python{ver} -m tkinter   (a small window should open)")
        if sys.prefix != sys.base_prefix:
            print("\nYou are in a virtual environment. This lab needs no extra packages, so you can also")
            print("leave it with 'deactivate' and use a system Python that already has Tkinter.")
    else:
        print("Tkinter is installed but there is no screen to draw on (for example over SSH).")
        print("Run this on a computer with a desktop, or enable X forwarding (ssh -X).")
    print("\nTo run without the window anyway: add --text (board in the terminal) or --no-graphics.")


def run(main):
    """Run a part's main() with readable messages for unfinished code."""
    try:
        main()
    except NotImplementedError:
        frames = traceback.extract_tb(sys.exc_info()[2])
        mine = [f for f in frames if os.path.basename(f.filename).startswith("part")]
        f = (mine or frames)[-1]
        print(f"\nNot written yet: {f.name}() in {os.path.basename(f.filename)}, line {f.lineno}.")
        print("Replace the 'raise NotImplementedError' line with your code, then run this file again.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(130)
