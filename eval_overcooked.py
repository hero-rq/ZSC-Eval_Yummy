# eval_overcooked.py  (inside your repo)
from third_party.zsc_quick import zsc_evaluate
from types import SimpleNamespace

# example NOP policy
policy = SimpleNamespace(act=lambda obs: 0)

score = zsc_evaluate(
    env_name="overcooked_v2",
    layout="cramped_room",
    policy_a=policy,
    policy_b=policy,
    episodes=5,
)
print("score:", score)
