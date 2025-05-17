"""
Lightweight fallback for the vanished `zsceval.evaluate.zsc_evaluate`.
Bundled inside our repo so imports are rock-solid.

Usage
-----
from third_party.zsc_quick import zsc_evaluate
"""
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent / "extern" / "JaxMARL"))


import jax
import jaxmarl
from types import SimpleNamespace

def zsc_evaluate(
    env_name: str,
    layout: str,
    policy_a: SimpleNamespace,
    policy_b: SimpleNamespace,
    episodes: int = 10,
    key: "jax.Array | None" = None,
) -> float:
    """
    Very small re-implementation:
    * runs `episodes` games
    * returns **mean team reward**
    Assumes two agents called 'agent_0' and 'agent_1'.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    env    = jaxmarl.make(env_name, layout=layout)
    total  = 0.0

    for _ in range(episodes):
        key, sub = jax.random.split(key)
        obs, state = env.reset(sub)
        done_all = False

        while not done_all:
            a0 = policy_a.act(obs["agent_0"])
            a1 = policy_b.act(obs["agent_1"])
            key, sub = jax.random.split(key)
            obs, state, reward, done, _ = env.step(
                sub, state, {"agent_0": a0, "agent_1": a1}
            )
            total     += sum(reward.values())
            done_all   = bool(done["__all__"])

    return total / episodes
