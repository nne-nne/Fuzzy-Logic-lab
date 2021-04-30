import numpy as np
import gym
import skfuzzy
from skfuzzy import control

env = gym.make('MountainCarContinuous-v0')
print(env.action_space, env.action_space.low, env.action_space.high)
print(env.observation_space, env.observation_space.low, env.observation_space.high)
observation = env.reset()
progress = control.Antecedent(np.arange(env.observation_space.low[0], env.observation_space.high[0], 0.01), 'progress')
velocity = control.Antecedent(np.arange(env.observation_space.low[1], env.observation_space.high[1], 0.01), 'velocity')
acceleration = control.Consequent(np.arange(env.action_space.low[0], env.action_space.high[0], 0.01), 'acceleration')
progress.automf(4, names=['lewo', 'dolek', 'gorka', 'szczyt'])
velocity.automf(5, names=['<<', '<', '-', '>', '>>'])
acceleration.automf(4, names=['cala_wstecz', 'wstecz', 'naprzod', 'cala_naprzod'])
rule1 = control.Rule(progress['lewo'] & velocity['>'], acceleration['cala_naprzod'])
rule2 = control.Rule(progress['lewo'] & velocity['<'], acceleration['cala_wstecz'])
rule3 = control.Rule(progress['gorka'] & velocity['>'], acceleration['cala_naprzod'])
rule4 = control.Rule(progress['gorka'] & velocity['<'], acceleration['cala_wstecz'])

system_ctrl = control.ControlSystem([rule1, rule2, rule3, rule4])
system = control.ControlSystemSimulation(system_ctrl)

for t in range(1000):
    env.render()
    # print(observation)
    system.input['progress'], system.input['velocity'] = observation
    system.compute()
    action = np.array([system.output['acceleration']])
    observation, reward, done, info = env.step(action)
    if done:
        print(f"Episode finished after {t} time steps.")
        break
env.close()
