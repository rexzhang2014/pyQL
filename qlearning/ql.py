import numpy as np
import pandas as pd
import os, sys, json, re

import numpy as np
import time
import sys
import tkinter as tk


UNIT = 40   # pixels
MAZE_H = 5  # grid height
MAZE_W = 5  # grid width


class Maze(tk.Tk, object):

    def __init__(self, n_agents=1, action_space = ['u', 'd', 'r', 'l']):
        super(Maze, self).__init__()
        self.cmap = ['red', 'blue', 'green', 'orange']
        self.action_space = action_space
        self.n_actions = len(self.action_space)
        self.n_agents = n_agents
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self.agents = []
        self.forbid_blocks = []
        self.target_blocks = []
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])


        self.forbid_blocks.extend([
            # self.hell1, 
            # self.hell2,
            # self.hell3,
            # self.hell4,
        ])

        # create oval
        oval1_center = origin + np.array([UNIT*1, UNIT *3])
        self.oval1 = self.canvas.create_oval(
            oval1_center[0] - 15, oval1_center[1] - 15,
            oval1_center[0] + 15, oval1_center[1] + 15,
            fill='yellow')

        # create oval
        oval2_center = origin + np.array([UNIT*3, UNIT * 1])
        self.oval2 = self.canvas.create_oval(
            oval2_center[0] - 15, oval2_center[1] - 15,
            oval2_center[0] + 15, oval2_center[1] + 15,
            fill='yellow')

        # create oval
        oval3_center = origin + np.array([UNIT*2, UNIT * 2])
        self.oval3 = self.canvas.create_oval(
            oval3_center[0] - 15, oval3_center[1] - 15,
            oval3_center[0] + 15, oval3_center[1] + 15,
            fill='yellow')

        # create oval
        oval4_center = origin + np.array([UNIT*3, UNIT * 3])
        self.oval4 = self.canvas.create_oval(
            oval4_center[0] - 15, oval4_center[1] - 15,
            oval4_center[0] + 15, oval4_center[1] + 15,
            fill='yellow')

        self.target_blocks.extend([
            self.oval1, 
            self.oval2,
            self.oval3,
            self.oval4,
        ])

        # create red rect
        self.setup_agents(origin)

        # pack all
        self.canvas.pack()
    def setup_agents(self, origin) :
        for n in range(self.n_agents) :
            
            rx = np.random.choice(range(MAZE_W), 1)[0]
            ry = np.random.choice(range(MAZE_H), 1)[0]
            
            rx = 0
            ry = n
            self.agents.append(self.canvas.create_rectangle(
                origin[0] + rx*UNIT - 15, origin[1] + ry*UNIT - 15,
                origin[0] + rx*UNIT + 15, origin[1] + ry*UNIT + 15,
                fill=self.cmap[n]))

    def empty_agents(self) :
        if self.agents :
            del self.agents
        self.agents = []


    def reset(self):
        self.update()
        time.sleep(0.5)
        for a in self.agents :
            self.canvas.delete(a)
        
        self.empty_agents()
        origin = np.array([20, 20])
        self.setup_agents(origin)
    
        # return observation
        return [ self.canvas.coords(a) for a in self.agents ]


    def step(self, rect, action):

        # for a in self.agents :
        s = self.canvas.coords(rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(rect)  # next state

        # reward function
        # return self.get_reward(s_)
        return s_

    def render(self):
        time.sleep(0.01)
        self.update()

class Reward() :
    
    def __init__(self, forbid=[], target=[], env_weight=10, col_weight=1) :

        self._forbid = forbid
        self._target = target

        self.col_forbid = []
        self.col_target = []
        self.col_weight = col_weight
        self.env_forbid = self._forbid
        self.env_target = self._target
        self.env_weight = env_weight

    def _env_reward(self, state) :
        # reward function      
        state1 = state
        if state in self.env_forbid :
            reward = -1 * self.env_weight
            done = 1
            state1 = 'terminal'
        elif state in self.env_target :
            reward = 1 * self.env_weight
            done = 1
            state1 = 'terminal'
        else:
            reward = 0
            done = 0
        
        return state1, reward, done

    def _col_reward(self, state) :
        state1 = state

        if state in self.col_forbid :
            reward = -1 * self.col_weight
            done = 1
            state1 = 'terminal'
        elif state in self.col_target :
            reward = 1 * self.col_weight
            done = 1
            state1 = 'terminal'
        else:
            reward = 0
            done = 0

        return state1, reward, done


    def get_reward(self, state) :
        # reward function      
        state1, env_reword, env_done = self._env_reward(state)
        state2, col_reword, col_done = self._col_reward(state)
        if state1=='terminal' or state2 =='terminal' :
            state = 'terminal'
        # print(env_reword+col_reword)
        return state, env_reword+col_reword, int(env_done+col_done>0)

    def add_forbid(self, forbid_rect, delta=0.1) :
        self.col_forbid.append(forbid_rect)
        if self.col_weight < 1*self.env_weight :
            self.col_weight += delta
        
        
        # if forbid_rect in self.env_target :
        #     self.env_target = [ r for r in self.env_target if r != forbid_rect ]

    def add_target(self, target_rect) :
        self.col_target.append(target_rect)
        
        # if target_rect in self.env_forbid :
        #     self.env_target = [ r for r in self.env_target if r != target_rect ]


    def reset(self) :
        self.col_forbid = []
        self.col_target = []

class QLearning() :
    def __init__(self, maze, QTables, n_episode=200, n_steps=500, e=1e-5, n_trace=3) -> None:
        self.maze = maze
        self.QTables = QTables
        self.n_episode = n_episode
        self.n_steps = n_steps
        self.e = e
        self.n_trace = n_trace
        self.q_trace = [0] * (n_trace + 1)
        self.history = []
    def update(self) :
        pass
    
    def TotalMaxQValues(self, trace=True) :
        total = 0
        for t in self.QTables :
            total += t.q_table.max(axis=1).sum()

        if trace :
            self.q_trace.pop(0)
            self.q_trace.append(total)

        return total

    def QStop(self) :
        current = self.q_trace[-1]
        lastavg = np.mean(self.q_trace[:-1])
        return abs(current - lastavg) <= self.e

    def TraceQValues(self) :
        total = 0
        for t in self.QTables :
            total += t.q_table.max(axis=1).sum()
        return total


class AgentQLearning(QLearning) :
    def __init__(self, *args, **kwargs) -> None:
        
        QLearning.__init__(self, *args, **kwargs)
        self.ckpt = kwargs.get('ckpt', None)

    def checkpoint(self, path, n_loop) :
        if self.ckpt :
            for t in self.QTables :
                t.q_table.to_csv(os.path.join(path, 'qt{}-{}.csv'.format(t.id, n_loop)))
    
    def save_history(self, path) :
        pd.DataFrame.from_records(self.history).to_csv(os.path.join(path, 'history.csv'))
            
    def update(self) :
        # 跟着行为轨迹
        df = pd.DataFrame(columns=('i','state','action_space','reward','Q','action'))
                # 转换为迷宫坐标（x,y）
        def set_state(observation):
            p = []
            p.append(int((observation[0]-5)/40))
            p.append(int((observation[1]-5)/40))
            return p
        
        for episode in range(self.n_episode):  
            # initial observation
            observations = self.maze.reset()
            ttl_max_qval =  self.TotalMaxQValues(trace=False)
            print("episode:{}, qval:{}.".format(episode, ttl_max_qval))
            self.history.append((episode, ttl_max_qval))

            all_done = np.zeros(len(self.QTables))
            # np.random.permutation(QTables)
            for i, RL, observation in zip(range(len(self.QTables)), self.QTables, observations) :
                # if all_done[i] == 1 :
                #     continue
                # print(RL.rwd.col_forbid)
                # print(RL.rwd.env_target)
                for steps in range(self.n_steps) :
                    
                    # fresh maze
                    self.maze.render()
                    s = set_state(observation)
                    # RL choose action based on observation
                    action = RL.choose_action(str(s))
                    # RL take action and get next observation and reward
                    next_step = self.maze.step(self.maze.agents[i], action)
                    observation_, reward, done = RL.rwd.get_reward(next_step)


                    if observation_ != 'terminal':
                        s_ = set_state(observation_)                                    
                    else :
                        s_ = 'terminal'

                    # RL learn from this transition
                    RL.learn(str(s), action, reward, str(s_), [ qt for qt in self.QTables if qt != RL ], steps)
                    q = RL.q_table.loc[str(s),action]
                    df = df.append(pd.DataFrame({'i':[i],'state':[s],'action_space':[self.maze.action_space[action]],'reward':[reward],'Q':[q],'action':action}), ignore_index=True)
                    
                    # swap observation
                    observation = observation_
                
                    # break while loop when end of this episode
                    if done:
                        
                        all_done[i] = 1

                        for j, t in enumerate(self.QTables) :
                            if j != i :
                                t.rwd.add_forbid(next_step)
                        #         # t.env.re
                        break

                if (all_done==1).all() :
                    all_done = np.zeros(len(self.QTables))
                    # for t in QTables :
                    #     t.rwd.reset()
                    break
            if self.TotalMaxQValues() and self.QStop() :
                print(len(self.q_trace), self.q_trace[-1])
                break
            
            self.checkpoint('./pyQL/qlearning/ckpt', episode)

        # end of game
        print('Learning is over.')
        self.ckpt = True # check the last point. 
        self.checkpoint('./pyQL/qlearning/qtable', episode)
        self.save_history('./pyQL/qlearning/qtable')


        # maze.pause()
        maze.destroy()


    
class QTable:
    def __init__(self, actions, forbid=[], target=[], learning_rate=0.01, reward_decay=0.9, colaborate_decay=0.5, e_greedy=0.9, id=-1, *args, **kwargs):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.beta = colaborate_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.id = np.random.choice(range(100), 1) if id == -1 else id

        self.rwd = Reward(forbid, target, *args, **kwargs)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_, competitors, steps):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]

        c_target = 0

        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal

        else:
            q_target = r  # next state is terminal
            for c in competitors :
                if s in c.q_table.index : 
                    
                    c_target += c.q_table.loc[s, a]
                    c_target = c_target / len(competitors) * self.beta

            # c_target *= r

        # print(c_target)
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict - c_target - 0.0* steps)  # update
        # print(s, a, s_, self.q_table.loc[s,:].tolist(),)
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )



if __name__ == "__main__":
    action_space = ['u', 'd', 'r', 'l']
    
    n_agents = 4
    maze = Maze(n_agents=n_agents, action_space=action_space)

    qtable1 = QTable(
        actions=list(range(len(action_space))),
        forbid=[maze.canvas.coords(rect) for rect in maze.forbid_blocks],
        target=[maze.canvas.coords(rect) for rect in maze.target_blocks],
        id=1,
        learning_rate=0.01,
        reward_decay=0.9,
        colaborate_decay=0.6,
        e_greedy=0.9,
        env_weight=1,
        col_weight=1
    )
    qtable2 = QTable(
        actions=list(range(len(action_space))),
        forbid=[maze.canvas.coords(rect) for rect in maze.forbid_blocks],
        target=[maze.canvas.coords(rect) for rect in maze.target_blocks],
        id=2,
        learning_rate=0.01,
        reward_decay=0.9,
        colaborate_decay=0.8,
        e_greedy=1,
        env_weight=1,
        col_weight=1,
    )
    qtable3 = QTable(
        actions=list(range(len(action_space))),
        forbid=[maze.canvas.coords(rect) for rect in maze.forbid_blocks],
        target=[maze.canvas.coords(rect) for rect in maze.target_blocks],
        id=3,
        learning_rate=0.01,
        reward_decay=0.9,
        colaborate_decay=0.8,
        e_greedy=0.9,
        env_weight=1,
        col_weight=1,
    )
    qtable4 = QTable(
        actions=list(range(len(action_space))),
        # forbid=[maze.canvas.coords(rect) for rect in maze.forbid_blocks],
        target=[maze.canvas.coords(rect) for rect in maze.target_blocks],
        id=4,
        learning_rate=0.01,
        reward_decay=0.9,
        colaborate_decay=0.8,
        e_greedy=0.9,
        env_weight=1,
        col_weight=1,
    )

    qlist = [
        qtable1,
        qtable2,
        qtable3, 
        qtable4,
    ]

    ql = AgentQLearning(maze, qlist, 100, 500, e=1e-7, ckpt=True)
    maze.after(0, lambda : ql.update())
    # maze.after(0, lambda : update1(maze, qlist ))
    maze.mainloop()