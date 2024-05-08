import math
import copy
import itertools
import numpy as np
import logging
from operator import itemgetter
from functools import cmp_to_key

from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.step_utils.states import TrainState

"""
Here we define the different planning controllers for our agents.
Each controller has an act-function that returns a dictionary containing an action for each agent.
"""


class PlanningController:
    def __init__(self, env):
        self.timestep = -1
        self.max_timesteps = env._max_episode_steps
        self.trajectories = [[[None, None] for y in range(self.max_timesteps)] for x in range(env.get_num_agents())]
        self.trajectories_fixed_until = [0 for x in range(env.get_num_agents())]
        self.conflicts_to_solve = []
        self.followup_conflicts = []
        self.episode_valid = True
        for agent_handle, agent in enumerate(env.agents):
            try:
                shortest_path1 = list(get_k_shortest_paths(env, agent.initial_position, agent.initial_direction, agent.target)[0])
                length1 = len(shortest_path1)
                path1found = True
            except:
                path1found = False
                length1 = np.Inf
            try:
                shortest_path2 = list(
                    get_k_shortest_paths(env, agent.initial_position, (agent.initial_direction+2) % 4, agent.target)[0])
                length2 = len(shortest_path2)
                path2found = True
            except:
                path2found = False
                length2 = np.Inf

            if path1found or path2found:
                if length1 <= length2:
                    shortest_path = shortest_path1
                else:
                    shortest_path = shortest_path2
                agent.initial_direction = shortest_path[0][1]
            else:
                shortest_path = []
                self.episode_valid = False

            self.trajectories[agent_handle] = build_trajectory(shortest_path, agent.earliest_departure, env,
                                                               self.trajectories[agent_handle])
            self.trajectories_fixed_until[agent_handle] = agent.earliest_departure - 1

    def find_conflicts(self, env):
        self.conflicts_to_solve = self.followup_conflicts
        self.followup_conflicts = []
        # for every train first adjust the trajectory in case of malfunction and check if these malfunctions cause any
        # conflicts
        for agent_handle, agent in enumerate(env.agents):
            if self.timestep == 0:
                self.conflicts_to_solve.extend(find_conflicting_trains(self.trajectories, agent, 0))
            if agent.malfunction_handler.malfunction_down_counter > 0:
                if self.timestep + 1 < self.max_timesteps and self.trajectories[agent_handle][self.timestep + 1][
                    0] != agent.position:
                    for i in range(agent.malfunction_handler.malfunction_down_counter + 1):
                        self.trajectories[agent_handle].insert(self.timestep, Waypoint(agent.position, agent.direction))
                        del self.trajectories[agent_handle][self.max_timesteps + 1:]
                    if self.timestep < self.trajectories_fixed_until[agent_handle]:
                        self.trajectories[agent_handle] = build_trajectory(agent.get_shortest_path(env.distance_map),
                                                                           min(agent.malfunction_handler.malfunction_down_counter +
                                                                           self.timestep, self.max_timesteps), env,
                                                                           self.trajectories[agent_handle])
                    self.trajectories_fixed_until[agent_handle] = min(agent.malfunction_handler.malfunction_down_counter + \
                                                                      self.timestep, self.max_timesteps)
                    self.conflicts_to_solve.extend(find_conflicting_trains(self.trajectories, agent,
                                                                           max(self.timestep, agent.earliest_departure - 2)))
        self.conflicts_to_solve.sort()
        self.conflicts_to_solve = list(k for k, _ in itertools.groupby(self.conflicts_to_solve))
        return self.conflicts_to_solve

    def find_solutions(self, conflict_to_solve, conflict_number, env):
        # compute conflict solutions for a given conflict
        conflict = conflict_to_solve
        possible_solutions = []
        agent1 = env.agents[conflict[1]]
        agent2 = env.agents[conflict[2]]
        possible_solutions.extend(solution_by_waiting(self.trajectories, self.timestep, agent1, agent2,
                                                      self.max_timesteps, max(conflict[0] - 2 * conflict_number,
                                                                              self.timestep),
                                                      self.trajectories_fixed_until, conflict_number, env))
        possible_solutions.extend(solution_by_waiting(self.trajectories, self.timestep, agent1, agent2,
                                                      self.max_timesteps, self.timestep,
                                                      self.trajectories_fixed_until, conflict_number, env))
        possible_solutions.extend(solutions_by_rerouting(self.trajectories, env, agent1, agent2, self.timestep, 6,
                                                         self.trajectories_fixed_until, self.max_timesteps))
        return possible_solutions

    def choose_solution_manually(self, conflict_to_solve, possible_solutions):
        # choose the appropriate conflict resolution according to heuristic values attributed to each option
        possible_solutions_sorted = sorted(possible_solutions, key=cmp_to_key(compare))
        if possible_solutions_sorted[0] is not None:
            self.trajectories[conflict_to_solve[1]] = possible_solutions_sorted[0][2]
            self.trajectories[conflict_to_solve[2]] = possible_solutions_sorted[0][3]
            self.trajectories_fixed_until = possible_solutions_sorted[0][5]
            self.followup_conflicts.extend(possible_solutions_sorted[0][4])
            return possible_solutions.index(possible_solutions_sorted[0])

    def always_choose_wait(self, conflict_to_solve, possible_solutions):
        # choose to make the train with the higher handle wait, if that conflict resolution is valid
        if possible_solutions[2] is not None:
            self.trajectories[conflict_to_solve[1]] = possible_solutions[2][2]
            self.trajectories[conflict_to_solve[2]] = possible_solutions[2][3]
            self.trajectories_fixed_until = possible_solutions[2][5]
            self.followup_conflicts.extend(possible_solutions[2][4])

    def choose_solution(self, conflict_to_solve, chosen_solution):
        # apply changes of the chosen conflict resolution to the trajectories
        self.trajectories[conflict_to_solve[1]] = chosen_solution[2]
        self.trajectories[conflict_to_solve[2]] = chosen_solution[3]
        self.trajectories_fixed_until = chosen_solution[5]
        self.followup_conflicts.extend(chosen_solution[4])

    def act(self, observations, env):
        # choose action to follow the planned trajectory
        actions = dict()
        for agent_handle, agent in enumerate(env.agents):
            action = action_on_trajectory(self.trajectories, agent_handle, self.timestep)
            actions.update({agent_handle: action})
        self.timestep += 1
        return actions


class BaselineController:
    def __init__(self, env):
        self.malfunction_happening = False
        self.shortest_paths = []
        for agent_handle, agent in enumerate(env.agents):
            shortest_path = agent.get_shortest_path(env.distance_map)
            self.shortest_paths.append(shortest_path)
            # for random environments we change the trains' orientation to be in the direction of the shortest path
            agent.initial_direction = self.shortest_paths[agent_handle][0][1]

    def act(self, observations, env):
        actions = dict()
        self.malfunction_happening = False
        for agent_handle, agent in enumerate(env.agents):
            if agent.state == TrainState.MALFUNCTION:
                self.malfunction_happening = True
        for agent_handle, agent in enumerate(env.agents):
            if self.malfunction_happening:
                action = 4
            elif agent.state == TrainState.DONE or agent.state == TrainState.MALFUNCTION_OFF_MAP \
                or agent.state == TrainState.WAITING:
                action = 0
            else:
                if agent.position != self.shortest_paths[agent_handle][0][0] and agent.state != TrainState.READY_TO_DEPART:
                    self.shortest_paths[agent_handle].pop(0)
                if self.shortest_paths[agent_handle][1][1] == (agent.direction - 1) % 4:
                    action = 1
                elif self.shortest_paths[agent_handle][1][1] == (agent.direction + 1) % 4:
                    action = 3
                else:
                    action = 2
            actions.update({agent_handle: action})
        return actions


def compare(x, y):
    if x is None:
        return 1
    elif y is None:
        return -1
    elif x[0] < y[0]:
        return -1
    elif y[0] < x[0]:
        return 1
    else:
        return 0
    return compare


def build_trajectory(path, timestep, env, current_trajectory):
    result = []
    for i in range(timestep):
        result.append(current_trajectory[i])
    for i in range(len(path)):
        result.append(path[i])
    remaining_steps = env._max_episode_steps + 1 - len(result)
    for i in range(remaining_steps):
        result.append([None, None])
    if remaining_steps < 0:
        del result[env._max_episode_steps + 1:]
    return result


def find_conflicting_trains(trajectories, agent, timestep):
    # Returns a list of conflicts of one specific agent containing the four attributes: earliest time of conflict
    # between the two trains, their handles ordered by priority and the absolute value of the difference of their
    # directions at the time of conflict to identify the type of conflict.
    # A conflict occurs when two trains want to use the same cell at the same time or the train with the lower handle
    # tries to access in the next two steps a cell where a train with a lower handle is currently located
    result = []
    # check for conflicts only with trains of lower handle, because their trajectories already included potential
    # malfunctions / with all other trains in case that only trains in malfunction are checked
    for i in range(len(trajectories)):
        if i == agent.handle:
            continue
        for j in range(timestep, len(trajectories[0])):
            if j < len(trajectories[0]) - 2 and trajectories[agent.handle][j][0] is None and \
                trajectories[agent.handle][j+1][0] is None and trajectories[agent.handle][j+2][0] is None:
                continue
            elif trajectories[agent.handle][j][0] == trajectories[i][j][0] and trajectories[i][j][0] is not None:
                if i < agent.handle:
                    result.append([j, i, agent.handle, abs(trajectories[i][j][1]-trajectories[agent.handle][j][1])])
                else:
                    result.append([j, agent.handle, i, abs(trajectories[i][j][1]-trajectories[agent.handle][j][1])])
                break
            elif (j < len(trajectories[0]) - 1 and trajectories[agent.handle][j][0] == trajectories[i][j+1][0] and \
                    trajectories[i][j+1][0] is not None):
                if i < agent.handle:
                    result.append([j, i, agent.handle, abs(trajectories[i][j+1][1]-trajectories[agent.handle][j][1])])
                else:
                    result.append(
                        [j, agent.handle, i, abs(trajectories[i][j + 1][1] - trajectories[agent.handle][j][1])])
                break
            elif (j < len(trajectories[0]) - 1 and trajectories[agent.handle][j+1][0] == trajectories[i][j][0] and \
                    trajectories[i][j][0] is not None):
                if i < agent.handle:
                    result.append([j, i, agent.handle, abs(trajectories[i][j][1]-trajectories[agent.handle][j+1][1])])
                else:
                    result.append(
                        [j, agent.handle, i, abs(trajectories[i][j][1] - trajectories[agent.handle][j+1][1])])
                break
            elif (j < len(trajectories[0]) - 2 and trajectories[agent.handle][j][0] == trajectories[i][j+2][0] and \
                    trajectories[i][j+2][0] is not None):
                if i < agent.handle:
                    result.append([j, i, agent.handle, abs(trajectories[i][j+2][1]-trajectories[agent.handle][j][1])])
                else:
                    result.append(
                        [j, agent.handle, i, abs(trajectories[i][j+2][1] - trajectories[agent.handle][j][1])])
                break
            elif (j < len(trajectories[0]) - 2 and trajectories[agent.handle][j+2][0] == trajectories[i][j][0] and \
                    trajectories[i][j][0] is not None):
                if agent.handle < i:
                    result.append([j, agent.handle, i, abs(trajectories[i][j][1]-trajectories[agent.handle][j+2][1])])
                else:
                    result.append(
                        [j, i, agent.handle, abs(trajectories[i][j][1] - trajectories[agent.handle][j + 2][1])])
                break
            if len(trajectories) > 2 and trajectories[agent.handle][j][0] == agent.target:
                break
    return result


def action_on_trajectory(trajectories, agent_handle, timestep):
    current_position = trajectories[agent_handle][timestep][0]
    current_direction = trajectories[agent_handle][timestep][1]
    next_position = trajectories[agent_handle][timestep + 1][0]
    next_direction = trajectories[agent_handle][timestep + 1][1]
    if next_position is None:
        action = 0
    elif current_position is None:
        action = 2
    elif next_direction == (current_direction - 1) % 4:
        action = 1
    elif next_direction == (current_direction + 1) % 4:
        action = 3
    elif current_position == next_position:
        action = 4
    else:
        action = 2
    return action


def is_conflict_free(trajectories, timestep, agent1):
    # Determines whether two trajectories are conflict free. It is therefore important that the agent that is given as
    # input is the one with handle 1, even if the trajectories are of two different trains. The method
    # find_conflicting_trains will then compare the two trajectories and see if there is a conflict.
    result = True
    if find_conflicting_trains(trajectories, agent1, timestep):
        result = False
    return result


def solution_by_waiting(trajectories, timestep, agent1, agent2, max_timesteps, time_of_waiting,
                        trajectories_fixed_until, conflict_number, env, is_followup_conflict=False):
    result = [None, None]
    followup_penalty = np.Inf
    current_trajectory1 = trajectories[agent1.handle]
    current_trajectory2 = trajectories[agent2.handle]
    temp_trajectories2 = copy.deepcopy(trajectories)
    temp_trajectories1 = copy.deepcopy(trajectories)
    temp_waiting_times1 = copy.deepcopy(trajectories_fixed_until)
    temp_waiting_times2 = copy.deepcopy(trajectories_fixed_until)
    if timestep == time_of_waiting:
        time_of_waiting1 = max(trajectories_fixed_until[agent1.handle], timestep, agent1.earliest_departure-1)
        time_of_waiting2 = max(trajectories_fixed_until[agent2.handle], timestep, agent2.earliest_departure-1)
    else:
        time_of_waiting1 = max(time_of_waiting, agent1.earliest_departure-1)
        time_of_waiting2 = max(time_of_waiting, agent2.earliest_departure-1)
    k = 1
    while k <= 50 + 2 * conflict_number:
        temp_trajectories2[agent2.handle].insert(time_of_waiting2,
                                                 Waypoint(trajectories[agent2.handle][time_of_waiting2][0],
                                                          trajectories[agent2.handle][time_of_waiting2][1]))
        del temp_trajectories2[agent2.handle][max_timesteps + 1:]
        if is_conflict_free([current_trajectory1, temp_trajectories2[agent2.handle]], timestep, env.agents[1]):
            # insert additional waiting steps for chains of waiting trains
            for i in range(3 * conflict_number):
                temp_trajectories2[agent2.handle].insert(time_of_waiting2,
                                                         Waypoint(trajectories[agent2.handle][time_of_waiting2][0],
                                                                  trajectories[agent2.handle][time_of_waiting2][1]))
                del temp_trajectories2[agent2.handle][max_timesteps + 1:]
                k += 1
            induced_conflicts = find_conflicting_trains(temp_trajectories2, agent2, timestep)
            new_conflicts = [0, 0, 0, 0]
            for j in range(len(induced_conflicts)):
                new_conflicts[induced_conflicts[j][3]] += 1
            temp_waiting_times2[agent2.handle] = min(max(time_of_waiting2, trajectories_fixed_until[agent2.handle]) + k,
                                                     max_timesteps)
            if not is_followup_conflict:
                penalty = k + assess_followup_conflicts(temp_trajectories2, timestep, max_timesteps, temp_waiting_times2,
                                                        induced_conflicts, env)
            else:
                followup_penalty = k
                break

            result[0] = [penalty, new_conflicts, current_trajectory1, temp_trajectories2[agent2.handle], induced_conflicts,
                    temp_waiting_times2]
            break
        k += 1

    while k <= 50 + 2 * conflict_number:
        temp_trajectories1[agent1.handle].insert(time_of_waiting1,
                                                 Waypoint(trajectories[agent1.handle][time_of_waiting1][0],
                                                          trajectories[agent1.handle][time_of_waiting1][1]))
        del temp_trajectories1[agent1.handle][max_timesteps + 1:]
        if is_conflict_free([current_trajectory2, temp_trajectories1[agent1.handle]], timestep, env.agents[1]):
            for i in range(3 * conflict_number):
                temp_trajectories1[agent1.handle].insert(time_of_waiting1,
                                                         Waypoint(trajectories[agent1.handle][time_of_waiting1][0],
                                                                  trajectories[agent1.handle][time_of_waiting1][1]))
                del temp_trajectories1[agent1.handle][max_timesteps + 1:]
                k += 1
            induced_conflicts = find_conflicting_trains(temp_trajectories1, agent1, timestep)
            new_conflicts = [0, 0, 0, 0]
            for j in range(len(induced_conflicts)):
                new_conflicts[induced_conflicts[j][3]] += 1
            temp_waiting_times1[agent1.handle] = min(max(time_of_waiting1, trajectories_fixed_until[agent1.handle]) + k,
                                                     max_timesteps)
            if not is_followup_conflict:
                penalty = k + assess_followup_conflicts(temp_trajectories1, timestep, max_timesteps, temp_waiting_times1,
                                                        induced_conflicts, env)
            else:
                followup_penalty = min(followup_penalty, k)
                break
            result[1] = [penalty, new_conflicts, temp_trajectories1[agent1.handle], current_trajectory2, induced_conflicts,
                    temp_waiting_times1]
            break
        k += 1
    if not is_followup_conflict:
        return result
    else:
        return followup_penalty


def assess_followup_conflicts(trajectories, timestep, max_timesteps, trajectories_fixed_until, conflicts, env):
    result = 0
    for i in range(len(conflicts)):
        if solution_by_waiting(trajectories, timestep, env.agents[conflicts[i][1]], env.agents[conflicts[i][2]],
                               max_timesteps, timestep, trajectories_fixed_until, 0, env, True) == [None, None]:
            return np.Inf
        else:
            result += solution_by_waiting(trajectories, timestep, env.agents[conflicts[i][1]],
                                          env.agents[conflicts[i][2]], max_timesteps, timestep,
                                          trajectories_fixed_until, 0, env, True)
    return result


def solutions_by_rerouting(trajectories, env, agent1, agent2, timestep, max_routes, trajectories_fixed_until,
                           max_timesteps):
    result = [None for x in range(max_routes)]
    current_trajectory1 = trajectories[agent1.handle]
    current_trajectory2 = trajectories[agent2.handle]
    rerouting_time1 = max(trajectories_fixed_until[agent1.handle], timestep)
    rerouting_time2 = max(trajectories_fixed_until[agent2.handle], timestep)
    planned_waypoint1 = current_trajectory1[rerouting_time1]
    planned_waypoint2 = current_trajectory2[rerouting_time2]

    for i in range(math.floor(max_routes/2)):
        k = i + 1
        if planned_waypoint2[0] is not None:
            result[2 * i] = helper_rerouting(env, trajectories, timestep, rerouting_time2, k, agent1, agent2, 2,
                                             planned_waypoint2, trajectories_fixed_until, max_timesteps)
        else:
            result[2 * i] = helper_rerouting(env, trajectories, timestep, max(rerouting_time2, agent2.earliest_departure),
                                             k, agent1, agent2, 2, Waypoint(agent2.initial_position, agent2.initial_direction),
                                             trajectories_fixed_until, max_timesteps)

        if planned_waypoint1[0] is not None:
            result[2 * i + 1] = helper_rerouting(env, trajectories, timestep, rerouting_time1, k, agent1, agent2, 1,
                                                 planned_waypoint1, trajectories_fixed_until, max_timesteps)
        else:
            result[2 * i + 1] = helper_rerouting(env, trajectories, timestep, max(rerouting_time1, agent1.earliest_departure),
                                                 k, agent1, agent2, 1, Waypoint(agent1.initial_position, agent1.initial_direction),
                                                 trajectories_fixed_until, max_timesteps)

    return result


def helper_rerouting(env, trajectories, timestep, rerouting_time, k, agent1, agent2, reroute_handle, planned_waypoint,
                     trajectories_fixed_until, max_timesteps):
    trajectory1 = trajectories[agent1.handle]
    trajectory2 = trajectories[agent2.handle]
    temp_trajectories = copy.deepcopy(trajectories)

    if reroute_handle == 1:
        possible_paths = get_k_shortest_paths(env, planned_waypoint[0], planned_waypoint[1], agent1.target, k)
        try:
            temp_trajectories[agent1.handle] = build_trajectory(possible_paths[-1], rerouting_time, env, trajectory1)
        except:
            # logging.debug('Error during rerouting of agent1')
            return None
        if is_conflict_free([trajectory2, temp_trajectories[agent1.handle]], timestep, env.agents[1]):
            induced_conflicts = find_conflicting_trains(temp_trajectories, agent1, timestep)
            new_conflicts = [0, 0, 0, 0]
            for j in range(len(induced_conflicts)):
                new_conflicts[induced_conflicts[j][3]] += 1
            delay = len(possible_paths[-1]) - len(possible_paths[0]) + assess_followup_conflicts(temp_trajectories,
                                                                                                 timestep,
                                                                                                 max_timesteps,
                                                                                                 trajectories_fixed_until,
                                                                                                 induced_conflicts, env)
            return [delay, new_conflicts, temp_trajectories[agent1.handle], trajectory2, induced_conflicts,
                    trajectories_fixed_until]

    else:
        possible_paths = get_k_shortest_paths(env, planned_waypoint[0], planned_waypoint[1], agent2.target, k)
        try:
            temp_trajectories[agent2.handle] = build_trajectory(possible_paths[-1], rerouting_time, env, trajectory2)
        except:
            # logging.debug('Error during rerouting of agent2')
            return None
        if is_conflict_free([trajectory1, temp_trajectories[agent2.handle]], timestep, env.agents[1]):
            induced_conflicts = find_conflicting_trains(temp_trajectories, agent2, timestep)
            new_conflicts = [0, 0, 0, 0]
            for j in range(len(induced_conflicts)):
                new_conflicts[induced_conflicts[j][3]] += 1
            delay = len(possible_paths[-1]) - len(possible_paths[0]) + assess_followup_conflicts(temp_trajectories,
                                                                                                 timestep,
                                                                                                 max_timesteps,
                                                                                                 trajectories_fixed_until,
                                                                                                 induced_conflicts, env)
            return [delay, new_conflicts, trajectory1, temp_trajectories[agent2.handle], induced_conflicts,
                    trajectories_fixed_until]
