"""The official copy of the Part 5 baseline features (the autograder uses this one)."""


def simple_features(env, state, action):
    nxt = env.next_position(state.pacman, action)
    dangers = list(state.ghosts) + list(env.sleepers)
    near = sum(1 for g in dangers if g == nxt or g in env.open_neighbors(nxt))
    f = {"bias": 1.0, "ghosts-1-step-away": float(near)}
    if near == 0 and nxt in state.food:
        f["eats-food"] = 1.0
    f["closest-food"] = env.closest_food_distance(nxt, state.food) / (env.width * env.height)
    return {k: v / 10.0 for k, v in f.items()}
