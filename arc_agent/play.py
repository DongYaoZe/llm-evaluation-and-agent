from arc_agi import Arcade
from arcengine import GameAction

arc = Arcade()
env = arc.make("ls20", render_mode="terminal")

for _ in range(10):
    env.step(GameAction.ACTION1)

print(arc.get_scorecard())
