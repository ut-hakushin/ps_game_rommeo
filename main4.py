# Configuration section
population_size = 2 # How many AIs in the population
mentor_instances = 1 # How many instances of each defined strategy there are
episode_length = 60 # How many turns to play
dve = 1 # During vs. ending reward

rate = 0.5 #利他性
alpha1 = 5#公平性1
alpha2 = 0.25#公平性1
beta1 = 0.05#公平性1
beta2 = 0.05#公平性1

training_time = 1.0 # How long to train in seconds per agent
testing_episodes = 1000 # How many episodes to play during the testing phase
Nsteps = 30


# Script section
import sys
import random
import numpy as np
from time import time
#from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook

# Prisoner's dillema rewards [Player 1 reward, Player 2 reward]


#reward_matrix = [[[10, 10], # Both players High
                #[10, 48], # Player 1 High, player 2 Low
                #[48, 10], # Player 1 Low, player 2 High
                #[0, 0]]] # Both players Low


# Human agents pick which action to perform
class AgentHuman:
    def pick_action(self, state):
        action = -1

        # Print the given state
        print("State: " + str(state) + " (" + str(len(state)) + "/" + str(episode_length) + ")")

        # Repeat until valid input provided
        while action not in [0, 1]:
            try:
                # Parse human's chosen action
                action = int(input("Choose Cooperate/Defect (0/1): "))
            except ValueError:
                # Prompt human for valid input
                print("Please input a number.")
        
        return action

    def reward_action(self, state, action, reward):
        pass

# Q agents learn the best action to perform for every state encountered
class AgentQ:
    def __init__(self, memory):
        self.wins = 0 # Number of times agent has won an episode
        self.losses = 0 # Number of times agent has lost an episode
        self.Q = {} # Stores the quality of each action in relation to each state
        self.memory = memory # The number of previous states the agent can factor into its decision
        self.epsilon_counter = 1 # Inversely related to learning rate

    def get_q(self, state):
        quality1 = self.Q[str(state[-self.memory:])][0]
        quality2 = self.Q[str(state[-self.memory:])][1]

        return quality1, quality2

    def set_q(self, state, quality1, quality2):
        self.Q[str(state[-self.memory:])][0] = quality1
        self.Q[str(state[-self.memory:])][1] = quality2

    def normalize_q(self, state):
        quality1, quality2 = self.get_q(state)

        normalization = min(quality1, quality2)

        self.set_q(state, (quality1 - normalization) * 0.95, (quality2 - normalization) * 0.95)

    def max_q(self, state):
        quality1, quality2 = self.get_q(state)

        if quality1 == quality2 or random.random() < (1 / self.epsilon_counter):
            return random.randint(0, 1)
        elif quality1 > quality2:
            return 0
        else:
            return 1

    def pick_action(self, state):
        # Decrease learning rate
        self.epsilon_counter += 0.5

        # If the given state was never previously encountered
        if str(state[-self.memory:]) not in self.Q:
            # Initialize it with zeros
            self.Q[str(state[-self.memory:])] = [0, 0]
    
        return self.max_q(state)

    def reward_action(self, state, action, reward):
        # Increase the quality of the given action at the given state
        self.Q[str(state[-self.memory:])][action] += reward

        # Normalize the Q matrix
        self.normalize_q(state)

    def mark_victory(self):
        self.wins += 1

    def mark_defeat(self):
        self.losses += 1

    def analyse(self):
        # What percentage of games resulted in victory/defeat
        percent_won = 0
        if self.wins > 0:
            percent_won = float(self.wins) / (self.wins + self.losses)
        
        '''
        percent_lost = 0
        if self.losses > 0:
            percent_lost = float(self.losses) / (self.wins + self.losses)
        '''

        # How many states will result in cooperation/defection
        times_cooperated = 0
        times_defected = 0

        for state in self.Q:
            action = self.max_q(eval(state))

            if action == 0:
                times_cooperated += 1
            else:
                times_defected += 1

        # What percentage of states will result in cooperation/defection
        percent_cooperated = 0
        if times_cooperated > 0:
            percent_cooperated = float(times_cooperated) / len(self.Q)

        '''
        percent_defected = 0
        if times_defected > 0:
            percent_defected = float(times_defected) / len(self.Q)
        '''

        # Return most relevant analysis
        return self.wins, percent_won, percent_cooperated

    def reset_analysis(self):
        self.wins = 0
        self.losses = 0

# Defined agents know which action to perform
class AgentDefined:
    def __init__(self, strategy):
        self.wins = 0 # Number of times agent has won an episode
        self.losses = 0 # Number of times agent has lost an episode
        self.strategy = strategy

    def pick_action(self, state):
        if self.strategy == 0: # Tit for tat
            if len(state) == 0: # On the first tern
                return 0 # Cooperate
            else: # Otherwise
                return state[-1] # Pick the last action of the opponent
        elif self.strategy == 1: # Holds a grudge
            if 1 in state: # If the enemy has ever defected
                return 1 # Defect
            else: # Otherwise
                return 0 # Cooperate
        elif self.strategy == 2: # Random
            return random.randint(0, 1)

    def reward_action(self, state, action, reward):
        pass # Since these agents are defined, no learning occurs

    def mark_victory(self):
        self.wins += 1

    def mark_defeat(self):
        self.losses += 1

    def analyse(self):
        # What percentage of games resulted in victory/defeat
        percent_won = 0
        if self.wins > 0:
            percent_won = float(self.wins) / (self.wins + self.losses)
        
        percent_lost = 0
        if self.losses > 0:
            percent_lost = float(self.losses) / (self.wins + self.losses)

        # Return most relevant analysis
        return self.wins, percent_won

# Stores all AIs
population = []

# Stores record of analysis of all AIs
population_analysis = []

# Stores all instances of defined strategies
mentors = []

# TODO: Mentor analysis

mem_list = []
mem_up_list = []
mem2_list = []
mem2_up_list = []
mem_down_list = []
total1_mean_list = []
total1_up_list = []
total1_down_list = []
total2_mean_list = []
total2_up_list = []
total2_down_list = []


for mem in range(1, Nsteps):

  # Create a random AI with a random amount of memory
  for i in range(population_size):
    #population.append(AgentQ(random.randint(2, 5)))
    population.append(AgentQ(mem))

  # Create instances of defined strategies
  #for i in range(2): # Number of defined strategies
  for i in range(1): # Number of defined strategies
    for j in range(mentor_instances):
        #mentors.append(AgentDefined(i))
        #mentors.append(AgentQ(random.randint(2, 5)))

        #mentors.append(AgentQ(mem))
        mentors.append(AgentQ(5))

  # Training time initialization
  start_time = time()
  remaining_time = training_time * population_size
  last_remaining_time = int(remaining_time)
  total_training_time = training_time * population_size

  # Training mode with AIs

  alt_list = []
  test_list = []
  test_count = 0

  state_array1 = np.zeros((1000000, episode_length))
  state_array2 = np.zeros((1000000, episode_length))

  state_count = 0
  st = 0

  while remaining_time > 0:
    test_list.append(test_count+1)
    test_count += 1
    # Calculate remaining training time
    remaining_time = start_time + total_training_time - time()

    # Things to be done every second
    if 0 <= remaining_time < last_remaining_time:
        # Alert user to remaining time
        progress = 100 * (total_training_time - remaining_time) / total_training_time
        sys.stdout.write('\rTraining [{0}] {1}%'.format(('#' * int(progress / 5)).ljust(19), int(min(100, progress + 5))))
        sys.stdout.flush()
        last_remaining_time = int(remaining_time * 2) / float(2)

        # Analyse population
        if time() > start_time + 0.5:
            time_step = []
            for agent in population:
                time_step.append(agent.analyse())
                agent.reset_analysis()
            population_analysis.append(time_step)

        # TODO: Analyse mentors

    state1 = [] # State visible to player 1 (actions of player 2)
    state2 = [] # State visible to player 2 (actions of player 1)

    # Pick a random member of the population to serve as player 1
    player1 = random.choice(population)

    # Pick a random member of the population or a defined strategy to serve as player 2
    #player2 = random.choice(population + mentors)
    player2 = random.choice(mentors)

    for i in range(episode_length):
        action = None

        action1 = player1.pick_action(state1) # Select action for player 1
        action2 = player2.pick_action(state2) # Select action for player 2

        state1.append(action2) # Log action of player 2 for player 1
        state2.append(action1) # Log action of player 1 for player 2

    # Stores the total reward over all games in an episode
    total_reward1 = 0
    total_reward2 = 0
    reward_list1 = []
    reward_list2 = []
    episode_list = []
    #state_list1 = []
    #state_list2 = []

    alt = 0
    for i in range(episode_length):
        episode_list.append(i+1)
        pre_action1 = action1
        pre_action2 = action2
        action1 = state2[i]
        action2 = state1[i]


        reward1 = 0 # Total reward due to the actions of player 1 in the entire episode
        reward2 = 0 # Total reward due to the actions of player 2 in the entire episode

        # Calculate rewards for each player
        #rand_list1 = [60, 60, 60, 60, 60, 60, 60, 60, 60, -60]
        rand_list1 = [60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, -60, -60, -60, -60, -60, -60, -60]
        rand_list2 = [60, -60]
        rand_val1 = random.choice(rand_list1)
        rand_val2 = random.choice(rand_list2)
        if action1 == 0 and action2 == 0: # Both players High
            #reward1 = reward_matrix[0][0][0]
            reward1 = 10
            state_array1[st][i] = 0
            #reward2 = reward_matrix[0][0][1]
            reward2 = 10
            state_array2[st][i] = 0
        elif action1 == 0 and action2 == 1: # Only player 2 Low
            #reward1 = reward_matrix[0][1][0]
            reward1 = 10
            state_array1[st][i] = 0
            reward2 = rand_val1
            state_array2[st][i] = 1
        elif action1 == 1 and action2 == 0: # Only player 1 Low
            reward1 = rand_val1
            state_array1[st][i] = 1
            #reward2 = reward_matrix[0][2][1]
            reward2 = 10
            state_array2[st][i] = 0

        elif action1 == 1 and action2 == 1: # Both players Low
            reward1 = rand_val2
            state_array1[st][i] = 1
            reward2 = rand_val2
            state_array2[st][i] = 1

        total_reward1 += reward1
        total_reward2 += reward2
        reward_list1.append(reward1)
        reward_list2.append(reward2)

        player1.reward_action(state1[:i], action1, reward1 * dve + (reward2-reward1) * rate - alpha1 * max([reward2-reward1, 0]) - beta2 * max([reward1-reward2, 0])) # Assign reward to action of player 1
        player2.reward_action(state2[:i], action2, reward2 * dve + (reward1-reward2) * rate - alpha2 * max([reward1-reward2, 0]) - beta2 * max([reward2-reward1, 0]))# Assign reward to action of player 2

        # Assign reward for alternating
        if pre_action1 == 1 and pre_action2 == 0 and action1 == 0 and action2 == 1:
          #reward_chunk = 1000
          reward_chunk = 0
          player1.reward_action(state1[:i], action1, reward_chunk)
          player2.reward_action(state2[:i], action2, reward_chunk)
          alt += 1
        elif pre_action1 == 0 and pre_action2 == 1 and action1 == 1 and action2 == 0:
          reward_chunk = 0
          #reward_chunk = 1000
          player1.reward_action(state1[:i], action1, reward_chunk)
          player2.reward_action(state2[:i], action2, reward_chunk)
          alt += 1
        else:
          reward_chunk = 0
          #reward_chunk = -1000
          player1.reward_action(state1[:i], action1, reward_chunk)
          player2.reward_action(state2[:i], action2, reward_chunk)
        


              

    alt_list.append(alt)
    st += 1
    # Assign reward for winning player
    if total_reward1 > total_reward2:
        reward_chunk = total_reward1 / episode_length * (1 - dve)

        for i in range(episode_length):
            action1 = state2[i]

            player1.reward_action(state1[:i], action1, reward_chunk)

            player1.mark_victory()
            player2.mark_defeat()
    elif total_reward2 > total_reward1:
        reward_chunk = total_reward2 / episode_length * (1 - dve)

        for i in range(episode_length):
            action2 = state1[i]

            player2.reward_action(state2[:i], action2, reward_chunk)

            player1.mark_victory()
            player2.mark_defeat()



            
  # Start new line
  print("")

  # Plot analysis of AIs
  victories_percent_x = []
  victories_percent_y = []
  victories_percent_colors = []

  victories_percent_min_y = 1.0
  victories_percent_max_y = 0.0

  for i in range(len(population_analysis[-1])):
    victories_percent_y.append([])

    wins, percent_won, percent_cooperated = population_analysis[-1][i]

    victories_percent_colors.append(percent_cooperated)

    if percent_cooperated < victories_percent_min_y:
        victories_percent_min_y = percent_cooperated

    if percent_cooperated > victories_percent_max_y:
        victories_percent_max_y = percent_cooperated

  row1_colors = []
  row2_colors = []

  min_color = 0.05
  max_color = 0.95

  for color in victories_percent_colors:
    normalized_color = (color - victories_percent_min_y) * (max_color - min_color) / (victories_percent_max_y - victories_percent_min_y) + min_color

    row1_colors.append(str(color))
    row2_colors.append(str(normalized_color))

  i = 0
  for time_step in population_analysis:
    victories_percent_x.append(i + 1)

    total_wins = 0

    for agent_analysis in time_step:
        wins, percent_won, percent_cooperated = agent_analysis

        total_wins += percent_won

    j = 0
    for agent_analysis in time_step:
        wins, percent_won, percent_cooperated = agent_analysis

        victories_percent = 0
        if wins > 0:
            victories_percent = float(percent_won) / total_wins

        victories_percent_y[j].append(victories_percent)

        j += 1

    i += 1

  #fig = plt.figure(figsize=(12, 10), dpi=80)


  #fig.savefig("figure.png")

  #plt.show()

  # Testing mode
  wins1 = 0
  wins2 = 0
  tie = 0


  total_list1 = []
  total_list2 = []
  alt_test_list = []
  alt_good_list = []
  alt_bad_list = []
  
  for i in range(testing_episodes):
    state1 = [] # State visible to player 1 (actions of player 2)
    state2 = [] # State visible to player 2 (actions of player 1)

    # Use a human to serve as player 1
    player1 = random.choice(population)

    # Use a random AI to serve as player 2
    player2 = random.choice(mentors)
    
    for i in range(episode_length):
        action1 = player1.pick_action(state1) # Allow player 1 to pick action
        action2 = player2.pick_action(state2) # Select action for player 2

        state1.append(action2) # Log action of player 2 for player 1
        state2.append(action1) # Log action of player 1 for player 2

    total_reward1 = 0
    total_reward2 = 0

    alt_test = 0

    for i in range(episode_length):
        pre_action1 = action1
        pre_action2 = action2
        action1 = state2[i]
        action2 = state1[i]

        reward1 = 0 # Total reward due to the actions of player 1 in the entire episode
        reward2 = 0 # Total reward due to the actions of player 2 in the entire episode

        # Calculate rewards for each player
        if action1 == 0 and action2 == 0: # Both players High
            #reward1 = reward_matrix[0][0][0]
            reward1 = 10
            #reward2 = reward_matrix[0][0][1]
            reward2 = 10
        elif action1 == 0 and action2 == 1: # Only player 2 Low
            #reward1 = reward_matrix[0][1][0]
            reward1 = 10
            reward2 = rand_val1
        elif action1 == 1 and action2 == 0: # Only player 1 Low
            reward1 = rand_val1
            #reward2 = reward_matrix[0][2][1]
            reward2 = 10
        elif action1 == 1 and action2 == 1: # Both players Low
            reward1 = rand_val2
            reward2 = rand_val2
        # Assign reward for alternating
        if pre_action1 == 1 and pre_action2 == 0 and action1 == 0 and action2 == 1:
          #reward_chunk = 1000
          alt_test += 1
        elif pre_action1 == 0 and pre_action2 == 1 and action1 == 1 and action2 == 0:
          alt_test += 1

        total_reward1 += reward1
        total_reward2 += reward2

    # Print the winning player and score
    #print("Score: " + str(total_reward1) + " to " + str(total_reward2))
    total_list1.append(total_reward1)
    total_list2.append(total_reward2)
    alt_test_list.append(alt_test)
    if alt_test >= 8:
        alt_good_list.append(1)
    else:
        alt_good_list.append(0)
    if alt_test == 0:
        alt_bad_list.append(1)
    else:
        alt_bad_list.append(0)
        
    if total_reward1 > total_reward2:
        #print("Player 1 wins!")
        wins1 += 1
    elif total_reward2 > total_reward1:
        #print("Player 2 wins!")
        wins2 += 1
    else:
        #print("Tie!")
        tie += 1


#print("Player 1 won " + str(wins1) + " times")
#print("Player 1  " + str(np.mean(total_list1)) + " points")
#print("Player 2 won " + str(wins2) + " times")
#print("Player 2  " + str(np.mean(total_list2)) + " points")
#print("Tie " + str(tie) + " times")
  total1_mean = np.mean(total_list1)
  total1_std = np.std(total_list1, ddof=1)
  total1_error = total1_std / np.sqrt(len(total_list1))
  total2_mean = np.mean(total_list2)
  total2_std = np.std(total_list2, ddof=1)
  total2_error = total2_std / np.sqrt(len(total_list2))
  total1_mean_list.append(total1_mean)
  #total1_up_list.append(total1_mean+total1_std)
  total1_up_list.append(total1_error)
  #total1_down_list.append(total1_mean-total1_std)
  total2_mean_list.append(total2_mean)
  #total2_up_list.append(total2_mean+total2_std)
  total2_up_list.append(total2_error)
  #total2_down_list.append(total2_mean-total2_std)
  print(total1_mean)
  print(total1_std)
  print(len(total_list1))
  print(total2_mean)
  print(total2_std)
  print(len(total_list2))

  print(alt_test_list)
  alt_mean = np.mean(alt_test_list)
  alt_std = np.std(alt_test_list, ddof=1)
  alt_error = alt_std / np.sqrt(len(alt_test_list))
  print(alt_good_list)
  alt_good_mean = np.mean(alt_good_list)
  alt_bad_mean = np.mean(alt_bad_list)
  alt_good_std = np.std(alt_good_list, ddof=1)
  alt_good_error = alt_good_std / np.sqrt(len(alt_good_list))
  alt_bad_std = np.std(alt_bad_list, ddof=1)
  alt_bad_error = alt_bad_std / np.sqrt(len(alt_bad_list))
  #if mem == 0:
  #    alt_mean = 0
  #mem_list.append(alt_mean)
  #mem_up_list.append(alt_mean+alt_std)
  #mem_up_list.append(alt_std)
  #mem_down_list.append(alt_mean-alt_std)
  print(alt_mean)
  #print(alt_std)
  print(alt_error)
  print(len(alt_test_list))

  #mem_list.append(alt_good_mean)
  #mem_list.append(alt_bad_mean)
  
  mem_list.append(alt_mean)
  mem_up_list.append(alt_error)
  
  #mem2_list.append(alt_good_mean)
  #mem_up_list.append(alt_mean+alt_std)
  #mem_up_list.append(alt_good_std)
  #mem_up_list.append(alt_bad_error)
  
  #mem2_up_list.append(alt_good_error)
  #mem_down_list.append(alt_mean-alt_std)
  print(alt_bad_mean)
  print(alt_good_mean)
  #print(alt_bad_std)
  print(alt_bad_error)
  #print(alt_good_std)
  print(alt_good_error)
  print(len(alt_bad_list))
  print(len(alt_good_list))
  print(mem)

#fig = plt.figure(figsize=(12, 10), dpi=80)



t = np.arange(1, Nsteps)
# the 1 sigma upper and lower analytic population bounds
#lower_bound = mem_list - sigma
#upper_bound = mem_list + sigma

fig = plt.figure()

ax = fig.add_subplot(1, 2, 1)
#fig, ax = plt.subplots(1,2,1)

ax.plot(t, mem_list, lw=2, label='alternations', color='blue')

ax.errorbar(t, mem_list, yerr=mem_up_list, marker='o', capthick=1, capsize=10, lw=1)
ax.legend(loc='upper left')

ax.set_xlabel('memory')
ax.set_ylabel('alternations')
#ax.grid()


ax2 = fig.add_subplot(1, 2, 2)

ax2.plot(t, total1_mean_list, lw=2, label='player1 total reward', color='blue')
#ax2.plot(t, mu*t, lw=1, label='population mean', color='black', ls='--')
#ax2.fill_between(t, total1_down_list, total1_up_list, facecolor='yellow', alpha=0.5,label='1 sigma range')

ax2.errorbar(t, total1_mean_list, yerr=total1_up_list, marker='o', capthick=1, capsize=10, lw=1)




ax2.plot(t, total2_mean_list, lw=2, label='player2 total reward', color='red')
#ax2.plot(t, mu*t, lw=1, label='population mean', color='black', ls='--')
#ax2.fill_between(t, total2_down_list, total2_up_list, facecolor='green', alpha=0.5,label='1 sigma range')
ax2.errorbar(t, total2_mean_list, yerr=total2_up_list, marker='o', capthick=1, capsize=10, lw=1)

ax2.legend(loc='upper left')

# here we use the where argument to only fill the region where the
# walker is above the population 1 sigma boundary
#ax2.fill_between(t, mem_up_list, mem_list, where=mem_list > mem_up_list, facecolor='blue',alpha=0.5)
ax2.set_xlabel('memory')
ax2.set_ylabel('total reward')
#ax2.grid()
plt.tight_layout()
plt.show()
