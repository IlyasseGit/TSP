#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 17:53:54 2019

@author: ilyasse
"""
import numpy as np
import gym
import random
import time
import gym
import random 
import numpy as np
from random import sample 
dist_hist=[]
full_tour_epis=[]
stefn=[]
env = gym.make('gym_tsp:tsp-v0')
q_table = np.zeros((env.observation_space.n, env.action_space.n))
num_episodes =200000
max_steps_per_episode = env.observation_space.n
distance_min=120*61
learning_rate = 0.1
discount_rate = 0.95

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.0001
exploration_decay_rate = 0.0001
rewards_all_episodes = []
episode_reward = 0
reward_dist=0

 # Tabul des state  (colonne=action, ligne=stat)
T=env.inf_env()
reward_dis=0
c=0
for episode in range(num_episodes):
    # initialize new episode params
    list_tabu=[]

    state = env.reset()
    list_tabu.append(state)
#    done = env.done
    done = False

    rewards_current_episode = 0
    distance=0
    for step in range(max_steps_per_episode): 
        # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state,:])
            
        else:
            
############## Avec tabou_List ######################
#            it_max=0
#            bol= True
#            while bol and it_max<10:
#                
#                action = env.action_space.sample()
#                if T[list_tabu[-1],action] not in list_tabu :
#                    bol= False
#                    
#                it_max+=1
############## Sans tabou _List ######################
            action = env.action_space.sample()
        # Take new action
        newstat, reward, done, _ = env.step(action)
        distance+=1/reward
        list_tabu.append(newstat)
        env.render()
        
        # Update Q-table
        
        rewards_current_episode += reward 

        if done == 1:

            if step<env.observation_space.n-1:
                reward -=5
            else:
#                reward +=10

#                    env.plot_tsp()
#                    print("distance",distance)
#
#                if list_tabu[-1]==list_tabu[0]:
#                    reward +=20
#                    dist_hist.append(distance)
#                    if distance_min>distance:
#                        reward_dist+=10
#                        distance_min=distance
#                        reward +=reward_dis  
#                    if distance_min==distance:
#                        reward +=reward_dis
#                    if distance_min<distance:
#                        
#                        reward -=30

                if list_tabu[-1]==list_tabu[0]:

                    dist_hist.append(distance)
                    full_tour_epis.append(episode)

#                    env.plot_tsp()
#                    print("distance",distance)

                if list_tabu[-1]==list_tabu[0]:
                    reward +=5 

                    if distance_min>distance:
                        reward_dist+=10
                        distance_min=distance
                        reward +=reward_dis  
                    if distance_min==distance:
                        reward +=reward_dis
                    if distance_min<distance:
                        reward =-5
                         
                        
                        
                    

#                else:
#                    reward -=10
            q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[newstat, :]))
            break
        # Set new state
        if done == 0:
            q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[newstat, :]))

            state = newstat
            
            
        # Add new reward
        #rewards_current_episode += reward 
        #if done == 1:
            #break
    # Exploration rate decay 

    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
    # Add current episode reward to total rewards list
    rewards_all_episodes.append(rewards_current_episode)
#    dis,step_test=test_Qtable(max_steps_per_episode)
#    if  episode>=6000 and step_test==5:
#        dist_hist.append(dis)
#        if min(dist_hist)==dis:
#            np.savetxt("q_best_table2.txt", q_table, delimiter =', ', fmt='%1.9f')

        
        
# Calculate and print the average reward per thousand episodes
rewards_per_thosand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
count = 1000
#np.savetxt("rewards_per_thosand_episodes.txt", rewards_per_thosand_episodes, delimiter =', ', fmt='%1.9f')

#
print("********Average reward per thousand episodes********\n")
for r in rewards_per_thosand_episodes:
    print(count, ": ", str(sum(r/1000)))
    count += 1000

# Print updated Q-table
print("\n\n********Q-table********\n")
print(q_table)


#=============== LOAD  Q-LEARNING  TABLE==================
#np.savetxt("q_table_best_new.txt", q_table, delimiter =', ', fmt='%1.9f')
#q_table = np.loadtxt("q_table_best_new.txt", delimiter=',')

#
#

    
state = env.reset()
#    done = env.done
done = False
reward = 0
    #
distance=0
env.render()
for step in range(max_steps_per_episode): 
        # Exploration-exploitation trade-off
        
        action = np.argmax(q_table[state,:])
        
        # Take new action
        newstat, rewards, done, _ = env.step(action)
        env.render()
        
        reward += rewards 
        distance+=1/rewards
        if done == True:
            
            break
        # Set new state
        if done == False:
            state = newstat
print(distance)
#======== Plot TSP  Optimal path================
env.plot_tsp()

#======== Plot evolution of the path distance during learning  ================
import matplotlib.pyplot as plt
plt.plot(full_tour_epis,dist_hist)