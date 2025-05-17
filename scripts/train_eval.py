from jaxmarl_baselines.mappo import MAPPOConfig, MAPPO
from third_party.zsc_quick import zsc_evaluate
from types import SimpleNamespace
import pickle, functools, jax, jax.numpy as jnp

# --- Train ---
cfg = MAPPOConfig(env_id="overcooked_v2", layout="cramped_room",
                  total_timesteps=5_000, log_interval=1000)
agent = MAPPO(cfg)
agent.train()
agent.save("chkpt.pkl")

# --- Wrap policy ---
params = pickle.load(open("chkpt.pkl", "rb"))
actor  = agent.actor_network          # reuse same net

@functools.partial(jax.jit, static_argnums=0)
def act(obs, p):                       # obs is dict with 'state'
    return int(jnp.argmax(actor.apply({"params": p}, obs["state"])))

policy = SimpleNamespace(act=lambda o: act(o, params))

# --- Evaluate ---
score = zsc_evaluate("overcooked_v2", "cramped_room",
                     policy_a=policy, policy_b=policy, episodes=3)
print("score:", score)
