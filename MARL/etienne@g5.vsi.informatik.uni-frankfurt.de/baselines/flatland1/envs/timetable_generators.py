import os
import json
import itertools
import warnings
from typing import Tuple, List, Callable, Mapping, Optional, Any
from flatland.envs.timetable_utils import Timetable
from flatland.envs.rail_trainrun_data_structures import Waypoint

import numpy as np
from numpy.random.mtrand import RandomState
import math

from flatland.envs.agent_utils import EnvAgent
from flatland.envs.distance_map import DistanceMap
from flatland.envs.rail_env_shortest_paths import get_shortest_paths
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths

def len_handle_none(v):
    if v is not None:
        return len(v)
    else:
        return 0

def timetable_generator(agents: List[EnvAgent], distance_map: DistanceMap,
                            agents_hints: dict, np_random: RandomState = None) -> Timetable:
    """
    Calculates earliest departure and latest arrival times for the agents
    This is the new addition in Flatland 3
    Also calculates the max episodes steps based on the density of the timetable

    inputs:
        agents - List of all the agents rail_env.agents
        distance_map - Distance map of positions to tagets of each agent in each direction
        agent_hints - Uses the number of cities
        np_random - RNG state for seeding
    returns:
        Timetable with the latest_arrivals, earliest_departures and max_episdode_steps
    """
    # max_episode_steps calculation
    if agents_hints:
        city_positions = agents_hints['city_positions']
        num_cities = len(city_positions)
    else:
        num_cities = 2

    timedelay_factor = 4
    alpha = 2
    max_episode_steps = int(timedelay_factor * alpha * \
        (distance_map.rail.width + distance_map.rail.height + (len(agents) / num_cities)))

    # Multipliers
    old_max_episode_steps_multiplier = 3.0
    new_max_episode_steps_multiplier = 1.5
    travel_buffer_multiplier = 1.3 # must be strictly lesser than new_max_episode_steps_multiplier
    assert new_max_episode_steps_multiplier > travel_buffer_multiplier
    end_buffer_multiplier = 0.05
    mean_shortest_path_multiplier = 0.2

    shortest_paths = get_shortest_paths(distance_map)
    shortest_paths_lengths = [len_handle_none(v) for k,v in shortest_paths.items()]

    # Find mean_shortest_path_time
    agent_speeds = [agent.speed_counter.speed for agent in agents]
    agent_shortest_path_times = np.array(shortest_paths_lengths)/ np.array(agent_speeds)
    mean_shortest_path_time = np.mean(agent_shortest_path_times)

    # Deciding on a suitable max_episode_steps
    longest_speed_normalized_time = np.max(agent_shortest_path_times)
    mean_path_delay = mean_shortest_path_time * mean_shortest_path_multiplier
    max_episode_steps_new = int(np.ceil(longest_speed_normalized_time * new_max_episode_steps_multiplier) + mean_path_delay)

    max_episode_steps_old = int(max_episode_steps * old_max_episode_steps_multiplier)

    max_episode_steps = min(max_episode_steps_new, max_episode_steps_old)

    end_buffer = int(max_episode_steps * end_buffer_multiplier)
    latest_arrival_max = max_episode_steps-end_buffer

    # Useless unless needed by returning
    earliest_departures = []
    latest_arrivals = []

    for agent in agents:
        agent_shortest_path_time = agent_shortest_path_times[agent.handle]
        agent_travel_time_max = int(np.ceil((agent_shortest_path_time * travel_buffer_multiplier) + mean_path_delay))

        departure_window_max = max(latest_arrival_max - agent_travel_time_max, 1)

        earliest_departure = np_random.randint(0, departure_window_max)
        latest_arrival = earliest_departure + agent_travel_time_max

        earliest_departures.append(earliest_departure)
        latest_arrivals.append(latest_arrival)

        agent.earliest_departure = earliest_departure
        agent.latest_arrival = latest_arrival

    return Timetable(earliest_departures=earliest_departures, latest_arrivals=latest_arrivals,
                    max_episode_steps=max_episode_steps)


def custom_timetable_generator(agents: List[EnvAgent], distance_map: DistanceMap,
                        agents_hints: dict, np_random: RandomState = None) -> Timetable:
    """
    Calculates earliest departure and latest arrival times for the agents
    This is the new addition in Flatland 3

    inputs:
        agents - List of all the agents rail_env.agents
        distance_map - Distance map of positions to targets of each agent in each direction
        agent_hints - Uses the number of cities
        np_random - RNG state for seeding
    returns:
        Timetable with the latest_arrivals, earliest_departures and max_episode_steps
    """

    shortest_paths = get_shortest_paths(distance_map)
    shortest_paths_lengths = [len_handle_none(v) for k, v in shortest_paths.items()]

    # Find mean_shortest_path_time
    agent_speeds = [agent.speed_counter.speed for agent in agents]
    agent_shortest_path_times = np.array(shortest_paths_lengths) / np.array(agent_speeds)

    # Useless unless needed by returning
    earliest_departures = []
    latest_arrivals = []

    for agent in agents:
        agent_shortest_path_time = agent_shortest_path_times[agent.handle]

        # manual schedule
        if agent.handle % 50 in range(2):
            earliest_departure = 111
        elif agent.handle % 50 == 2 or agent.handle % 50 == 4 or agent.handle % 50 in range(7, 9) or \
            agent.handle % 50 in range(10, 14) or agent.handle % 50 == 15:
            earliest_departure = 101 + 150 * math.floor(agent.handle / 2)
        elif agent.handle % 50 == 3 or agent.handle % 50 == 5 or agent.handle % 50 == 9:
            earliest_departure = 266 + 150 * (math.floor(agent.handle / 2) - 1)
        elif agent.handle % 50 == 6 or agent.handle % 50 == 14:
            earliest_departure = 550 + 150 * (math.floor(agent.handle / 2) - 3)
        elif agent.handle % 50 in range(16, 18):
            earliest_departure = 221 + 120 * (agent.handle - 16)
        elif agent.handle % 50 == 18 or agent.handle % 50 in range(20, 23):
            earliest_departure = 201 + 150 * (agent.handle - 16)
        elif agent.handle % 50 == 19 or agent.handle % 50 == 23:
            earliest_departure = 681 + 300 * (math.floor(agent.handle / 2) - 9)
        elif agent.handle % 50 in range(24, 28):
            earliest_departure = 471 + 160 * (agent.handle % 2) + 600 * (math.floor(agent.handle / 2) - 12)
        elif agent.handle % 50 == 28:
            earliest_departure = 321
        elif agent.handle % 50 in range(29, 36):
            earliest_departure = 311 - 140 * (agent.handle % 2) + 300 * (math.floor(agent.handle / 2) - 14)
        elif agent.handle % 50 == 37 or agent.handle % 50 == 39:
            earliest_departure = 196 + 600 * (math.floor(agent.handle / 2) - 18)
        elif agent.handle % 50 == 36 or agent.handle % 50 == 38:
            earliest_departure = 151 + 600 * (math.floor(agent.handle / 2) - 18)
        elif agent.handle % 50 in range(40, 42):
            earliest_departure = 21 - 20 * (agent.handle % 2)
        elif agent.handle % 50 in range(42, 44):
            earliest_departure = 621 - 20 * (agent.handle % 2)
        elif agent.handle % 50 in range(44, 46):
            earliest_departure = 31 - 30 * (agent.handle % 2) + 600 * (math.floor(agent.handle / 2) - 20)
        elif agent.handle % 50 in range(46, 48):
            earliest_departure = 171 + 175 * (agent.handle % 2)
        elif agent.handle % 50 in range(48, 50):
            earliest_departure = 321 - 130 * (agent.handle % 2)

        latest_arrival = earliest_departure + agent_shortest_path_time

        earliest_departures.append(earliest_departure)
        latest_arrivals.append(latest_arrival)

        agent.earliest_departure = earliest_departure
        agent.latest_arrival = latest_arrival

        # manual limit so that all trains can reach their target (without malfunctions)
        max_episode_steps = 1500

    return Timetable(earliest_departures=earliest_departures, latest_arrivals=latest_arrivals,
                     max_episode_steps=max_episode_steps)


def conflict_free_timetable_generator(env, agents: List[EnvAgent], distance_map: DistanceMap,
                        agents_hints: dict, np_random: RandomState = None) -> Timetable:
    """
    Calculates earliest departure and latest arrival times for the agents
    This is the new addition in Flatland 3

    inputs:
        agents - List of all the agents rail_env.agents
        distance_map - Distance map of positions to targets of each agent in each direction
        agent_hints - Uses the number of cities
        np_random - RNG state for seeding
    returns:
        Timetable with the latest_arrivals, earliest_departures and max_episode_steps
    """

    shortest_paths = get_shortest_paths(distance_map)
    shortest_paths_lengths = [len_handle_none(v) for k, v in shortest_paths.items()]

    # Find mean_shortest_path_time
    agent_speeds = [agent.speed_counter.speed for agent in agents]
    agent_shortest_path_times = np.array(shortest_paths_lengths) / np.array(agent_speeds)

    # Useless unless needed by returning
    earliest_departures = []
    latest_arrivals = []
    agent_paths = [[] for x in range(len(agents))]

    for agent in agents:
        if list(get_k_shortest_paths(env, agent.initial_position, agent.initial_direction, agent.target)):
            shortest_path1 = list(get_k_shortest_paths(env, agent.initial_position, agent.initial_direction, agent.target)[0])
            length1 = len(shortest_path1)
        else:
            length1 = np.Inf
        if list(get_k_shortest_paths(env, agent.initial_position, (agent.initial_direction + 2) % 4, agent.target)):
            shortest_path2 = list(get_k_shortest_paths(env, agent.initial_position, (agent.initial_direction + 2) % 4, agent.target)[0])
            length2 = len(shortest_path2)
        else:
            length2 = np.Inf
        if length1 < length2:
            agent_paths[agent.handle] = shortest_path1
        else:
            agent_paths[agent.handle] = shortest_path2
        # agent_paths[agent.handle].insert(0, Waypoint(agent.initial_position, agent.initial_direction))
        agent_paths[agent.handle].insert(len(agent_paths[agent.handle]), Waypoint(agent.target, 0))
        agent_paths[agent.handle].insert(len(agent_paths[agent.handle]), Waypoint(agent.target, 0))

    for agent in agents:
        agent_shortest_path_time = agent_shortest_path_times[agent.handle]
        if agent.handle == 0:
            departure_time = 1
        else:
            # choose this departure time for random environments
            departure_time = latest_departure_time + np.clip(round(np_random.normal(5, 5/3)), 0, 10)
            # choose this departure time for the double track, where the trains should not be too close to each other
            # departure_time = latest_departure_time + np.clip(round(np_random.normal(10, 10/3)), 0, 20)
            i = 0
            while i < agent.handle:
                found_conflict_free_path = True
                upper_bound = min(len(agent_paths[agent.handle]), len(agent_paths[i]) - departure_time + earliest_departures[i])
                for j in range(upper_bound):
                    if agent_paths[agent.handle][j][0] == agent_paths[i][j+departure_time-earliest_departures[i]][0]  or \
                        (j < upper_bound - 1 and \
                         (agent_paths[agent.handle][j+1][0] == agent_paths[i][j+departure_time-earliest_departures[i]][0] or \
                            agent_paths[agent.handle][j][0] == agent_paths[i][j+1+departure_time-earliest_departures[i]][0])) or \
                        (j+departure_time-earliest_departures[i] > 0 and \
                            agent_paths[agent.handle][j][0] == agent_paths[i][j-1+departure_time-earliest_departures[i]][0]) or \
                        (j < upper_bound - 2 and \
                         (agent_paths[agent.handle][j+2][0] == agent_paths[i][j+departure_time-earliest_departures[i]][0] or \
                             agent_paths[agent.handle][j][0] == agent_paths[i][j+2+departure_time-earliest_departures[i]][0])) or \
                        (j+departure_time-earliest_departures[i] > 1 and \
                             agent_paths[agent.handle][j][0] == agent_paths[i][j-2+departure_time-earliest_departures[i]][0]):
                        found_conflict_free_path = False
                        departure_time += 1 + np.clip(round(np_random.normal(3, 3/3)), 0, 6)
                        i = 0
                        break
                if found_conflict_free_path:
                    i += 1
        latest_departure_time = departure_time
        earliest_departure = departure_time
        latest_arrival = earliest_departure + len(agent_paths[agent.handle])

        earliest_departures.append(earliest_departure)
        latest_arrivals.append(latest_arrival)

        agent.earliest_departure = earliest_departure
        agent.latest_arrival = latest_arrival

    # manual limit so that all trains can reach their target (without malfunctions)
    max_episode_steps = latest_arrival + 5 * len(agents)
    return Timetable(earliest_departures=earliest_departures, latest_arrivals=latest_arrivals,
                     max_episode_steps=max_episode_steps)


def test_timetable_generator(agents: List[EnvAgent], distance_map: DistanceMap,
                        agents_hints: dict, np_random: RandomState = None) -> Timetable:
    """
    Calculates earliest departure and latest arrival times for the agents
    This is the new addition in Flatland 3

    inputs:
        agents - List of all the agents rail_env.agents
        distance_map - Distance map of positions to targets of each agent in each direction
        agent_hints - Uses the number of cities
        np_random - RNG state for seeding
    returns:
        Timetable with the latest_arrivals, earliest_departures and max_episode_steps
    """

    shortest_paths = get_shortest_paths(distance_map)
    shortest_paths_lengths = [len_handle_none(v) for k, v in shortest_paths.items()]

    # Find mean_shortest_path_time
    agent_speeds = [agent.speed_counter.speed for agent in agents]
    agent_shortest_path_times = np.array(shortest_paths_lengths) / np.array(agent_speeds)

    # Useless unless needed by returning
    earliest_departures = []
    latest_arrivals = []

    for agent in agents:
        agent_shortest_path_time = agent_shortest_path_times[agent.handle]

        # manual schedule
        earliest_departure = 1 + 12 * agent.handle

        latest_arrival = earliest_departure + agent_shortest_path_time

        earliest_departures.append(earliest_departure)
        latest_arrivals.append(latest_arrival)

        agent.earliest_departure = earliest_departure
        agent.latest_arrival = latest_arrival

        # manual limit so that all trains can reach their target (without malfunctions)
        max_episode_steps = 50 + 25 * int(len(agents) / 2)

    return Timetable(earliest_departures=earliest_departures, latest_arrivals=latest_arrivals,
                     max_episode_steps=max_episode_steps)
