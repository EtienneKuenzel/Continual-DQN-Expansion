from typing import Tuple

import numpy as np


from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap
from typing import Tuple
import random
import time

def make_custom_rail() -> Tuple[GridTransitionMap, np.array]:
    """
    In this method we create a custom railway grid according to the specifications of our model.
    Returns a GridTransitionMap "rail" and an array "rail_map", which can be used by a rail_generator
    """
    transitions = RailEnvTransitions()
    cells = transitions.transition_list

    empty = cells[0]
    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    dead_end_from_north = transitions.rotate_transition(dead_end_from_south, 180)
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
    right_turn_from_south = cells[8]
    right_turn_from_west = transitions.rotate_transition(right_turn_from_south, 90)
    right_turn_from_north = transitions.rotate_transition(right_turn_from_south, 180)
    right_turn_from_east = transitions.rotate_transition(right_turn_from_south, 270)
    simple_switch_north_left = cells[2]
    simple_switch_north_right = cells[10]
    simple_switch_east_left = transitions.rotate_transition(simple_switch_north_left, 90)
    simple_switch_east_right = transitions.rotate_transition(simple_switch_north_right, 90)
    simple_switch_west_left = transitions.rotate_transition(simple_switch_north_left, 270)
    simple_switch_west_right = transitions.rotate_transition(simple_switch_north_right, 270)
    simple_switch_south_left = transitions.rotate_transition(simple_switch_north_left, 180)
    simple_switch_south_right = transitions.rotate_transition(simple_switch_north_right, 180)
    double_switch_north_west = cells[5]
    double_switch_north_east = transitions.rotate_transition(double_switch_north_west, 90)

    rail_map = np.array(
        [[empty] * 118] * 5 +
        [[empty] * 37 + [dead_end_from_south] + [empty] * 80] +
        [[empty] * 37 + [vertical_straight] + [empty] * 80] +
        [[empty] * 37 + [vertical_straight] + [empty] * 19 + [dead_end_from_east] + [horizontal_straight] + [
            simple_switch_west_left] + [horizontal_straight] * 2 + [dead_end_from_west] + [empty] * 42 + [
             dead_end_from_east] + [horizontal_straight] + [simple_switch_west_left] + [horizontal_straight] * 3 + [
             simple_switch_east_right] + [horizontal_straight] + [dead_end_from_west] + [empty] * 4] +
        [[empty] + [dead_end_from_east] + [horizontal_straight] * 4 + [simple_switch_east_right] + [
            horizontal_straight] * 4 + [
             simple_switch_west_left] + [horizontal_straight] * 8 + [simple_switch_west_left] + [
             horizontal_straight] * 16 + [
             simple_switch_west_right] + [simple_switch_east_right] + [horizontal_straight] * 18 + [
             simple_switch_east_right] + [
             simple_switch_west_left] + [simple_switch_east_left] + [horizontal_straight] * 15 + [
             simple_switch_west_left] + [
             horizontal_straight] * 30 + [simple_switch_west_left] + [simple_switch_east_left] + [
             horizontal_straight] * 3 + [
             simple_switch_west_right] + [simple_switch_east_right] + [horizontal_straight] * 2 + [
             dead_end_from_west] + [
             empty] * 2] +
        [[empty] + [dead_end_from_east] + [horizontal_straight] * 4 + [double_switch_north_east] + [
            simple_switch_west_left] + [
             horizontal_straight] * 3 + [simple_switch_east_left] + [horizontal_straight] * 7 + [
             simple_switch_west_left] + [
             simple_switch_east_left] + [horizontal_straight] * 17 + [simple_switch_west_right] + [
             horizontal_straight] * 18 + [
             simple_switch_west_right] + [simple_switch_east_left] + [simple_switch_east_right] + [
             horizontal_straight] * 14 + [
             simple_switch_west_left] + [simple_switch_east_left] + [horizontal_straight] * 30 + [
             simple_switch_east_left] + [
             horizontal_straight] * 5 + [simple_switch_west_right] + [horizontal_straight] * 2 + [
             dead_end_from_west] + [
             empty] * 2] +
        [[empty] + [dead_end_from_east] + [horizontal_straight] * 4 + [simple_switch_west_right] + [
            simple_switch_east_left] + [
             horizontal_straight] * 3 + [right_turn_from_west] + [empty] * 7 + [vertical_straight] + [empty] * 36 + [
             dead_end_from_east] + [horizontal_straight] + [simple_switch_east_right] + [simple_switch_north_left] + [
             empty] * 5 + [right_turn_from_south] + [horizontal_straight] * 6 + [right_turn_from_west] + [empty] + [
             simple_switch_north_right] + [simple_switch_west_left] + [horizontal_straight] + [dead_end_from_west] + [
             empty] * 40] +
        [[empty] + [dead_end_from_east] + [horizontal_straight] * 5 + [simple_switch_east_left] + [
            horizontal_straight] * 2 + [
             simple_switch_east_right] + [double_switch_north_east] + [horizontal_straight] * 4 + [
             simple_switch_west_left] + [
             horizontal_straight] * 2 + [simple_switch_east_left] + [horizontal_straight] + [dead_end_from_west] + [
             empty] * 36 + [vertical_straight] * 2 + [empty] * 2 + [dead_end_from_east] + [horizontal_straight] + [
             simple_switch_west_left] + [simple_switch_east_left] + [horizontal_straight] * 6 + [
             simple_switch_west_right] + [
             simple_switch_west_left] + [right_turn_from_north] + [simple_switch_south_left] + [horizontal_straight] + [
             dead_end_from_west] + [empty] * 40] +
        [[empty] * 7 + [dead_end_from_east] + [horizontal_straight] + [right_turn_from_west] + [
            vertical_straight] * 2 + [
             empty] * 4 + [vertical_straight] + [empty] * 41 + [vertical_straight] + [simple_switch_south_left] + [
             horizontal_straight] * 4 + [simple_switch_east_left] + [horizontal_straight] * 7 + [
             simple_switch_west_left] + [
             right_turn_from_north] + [empty] + [vertical_straight] + [empty] * 42] +
        [[empty] + [dead_end_from_east] + [horizontal_straight] * 7 + [double_switch_north_east] + [
            simple_switch_west_right] + [double_switch_north_east] + [horizontal_straight] * 3 + [
             simple_switch_west_left] + [
             right_turn_from_north] + [empty] * 41 + [vertical_straight] + [right_turn_from_east] + [
             simple_switch_east_right] + [horizontal_straight] * 10 + [simple_switch_west_left] + [
             right_turn_from_north] + [
             empty] * 2 + [vertical_straight] + [empty] * 42] +
        [[empty] * 7 + [dead_end_from_east] + [horizontal_straight] + [double_switch_north_east] + [
            horizontal_straight] + [
             simple_switch_west_right] + [horizontal_straight] * 2 + [simple_switch_west_left] + [
             right_turn_from_north] + [
             empty] * 40 + [dead_end_from_east] + [horizontal_straight] + [simple_switch_west_right] + [
             horizontal_straight] + [
             simple_switch_west_right] + [simple_switch_east_right] + [horizontal_straight] * 9 + [
             double_switch_north_west] + [
             horizontal_straight] * 3 + [simple_switch_south_right] + [empty] * 42] +
        [[empty] * 7 + [dead_end_from_east] + [horizontal_straight] + [simple_switch_west_right] + [
            horizontal_straight] * 4 + [
             simple_switch_east_left] + [horizontal_straight] + [dead_end_from_west] + [empty] * 44 + [
             right_turn_from_east] + [
             simple_switch_east_right] + [horizontal_straight] * 7 + [simple_switch_west_left] + [
             right_turn_from_north] + [
             empty] * 3 + [vertical_straight] + [empty] * 42] +
        [[empty] * 62 + [right_turn_from_east] + [simple_switch_east_right] + [horizontal_straight] * 5 + [
            simple_switch_west_left] + [right_turn_from_north] + [empty] * 4 + [dead_end_from_north] + [empty] * 42] +
        [[empty] * 63 + [right_turn_from_east] + [simple_switch_east_right] + [horizontal_straight] * 3 + [
            simple_switch_west_left] + [right_turn_from_north] + [empty] * 48] +
        [[empty] * 64 + [right_turn_from_east] + [horizontal_straight] * 2 + [simple_switch_west_left] + [
            right_turn_from_north] + [empty] * 49] +
        [[empty] * 65 + [dead_end_from_east] + [horizontal_straight] + [right_turn_from_north] + [empty] * 50] +
        [[empty] * 118] * 7, dtype=np.uint16)

    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map
    print(rail_map[10][7])
    city_positions = [(12, 11), (13, 66), (8, 109)]
    train_stations = [[((8, 15), 0), ((9, 15), 1), ((11, 14), 2), ((13, 13), 3), ((15, 12), 4)], [
        ((7, 61), 0), ((8, 67), 1), ((9, 67), 2), ((12, 61), 3), ((13, 66), 4)], [
                          ((7, 109), 0), ((8, 109), 1), ((9, 109), 2)]]
    city_orientations = [1, 1, 3]
    agents_hints = {'city_positions': city_positions,
                    'train_stations': train_stations,
                    'city_orientations': city_orientations
                    }
    optionals = {'agents_hints': agents_hints}
    return rail, rail_map, optionals

def make_deadlock_training(x_length, y_length, switch_prop) -> Tuple[GridTransitionMap, np.array]:
    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    empty = cells[0]
    # Deadends
    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    # straights
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
    # switches
    simple_switch_north_left = cells[2]
    simple_switch_north_right = cells[10]
    out_of_lane_down = transitions.rotate_transition(simple_switch_north_right, 90)
    out_of_lane_up = transitions.rotate_transition(simple_switch_north_left, 90)
    in_to_lane_up = transitions.rotate_transition(simple_switch_north_right, 270)
    in_to_lane_down = transitions.rotate_transition(simple_switch_north_left, 270)
    random.seed(time.clock())
    # Insertanfang
    inswitch = []
    outswitch = []
    rails = [[empty] * x_length]
    for y in range(y_length):
        inswitch_copy, outswitch_copy = inswitch, outswitch
        inswitch, outswitch = [], []
        insert = []
        for x in range(x_length):
            if x == 0:
                insert = [dead_end_from_east]
            elif x == x_length - 1:
                insert = insert + [dead_end_from_west]
            elif x in inswitch_copy:
                insert = insert + [in_to_lane_up]
            elif x in outswitch_copy:
                insert = insert + [out_of_lane_up]
            elif random.randint(0, 99) < switch_prop and y != y_length-1:
                if random.randint(1, x_length) > x:
                    insert = insert + [out_of_lane_down]
                    inswitch.append(x)
                else:
                    insert = insert + [in_to_lane_down]
                    outswitch.append(x)
            else:
                insert = insert + [horizontal_straight]
        rails = rails + [insert]

    rail_map = np.array(rails, dtype=np.uint16)
    rail = GridTransitionMap(width=rail_map.shape[1], height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map

    city_positions = [(1, 0), (1, x_length-1),(2, 0), (2, x_length-1)]

    train_stations = [
        [((1, 0), 0), ((1, x_length-1), 1)],
        [((1, x_length-1), 0), ((1, 0), 1)],
        [((2, 0), 0), ((2, x_length - 1), 1)],
        [((2, x_length - 1), 0), ((2, 0), 1)]
    ]
    city_orientations = [1, 3, 1, 3]
    agents_hints = {'city_positions': city_positions,
                    'train_stations': train_stations,
                    'city_orientations': city_orientations
                    }
    optionals = {'agents_hints': agents_hints}
    return rail, rail_map, optionals

def make_deadlock_training1(x_length, y_length, switch_prop) -> Tuple[GridTransitionMap, np.array]:
    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    empty = cells[0]
    # Deadends
    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    # straights
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
    # switches
    simple_switch_north_left = cells[2]
    simple_switch_north_right = cells[10]
    out_of_lane_down = transitions.rotate_transition(simple_switch_north_right, 90)
    out_of_lane_up = transitions.rotate_transition(simple_switch_north_left, 90)
    in_to_lane_up = transitions.rotate_transition(simple_switch_north_right, 270)
    in_to_lane_down = transitions.rotate_transition(simple_switch_north_left, 270)
    random.seed(time.clock())
    # Insertanfang
    inswitch = []
    outswitch = []
    rails = [[empty] * x_length]
    for y in range(y_length):
        inswitch_copy, outswitch_copy = inswitch, outswitch
        inswitch, outswitch = [], []
        insert = []
        for x in range(x_length):
            if x == 0:
                insert = [dead_end_from_east]
            elif x == x_length - 1:
                insert = insert + [dead_end_from_west]
            elif x in inswitch_copy:
                insert = insert + [in_to_lane_up]
            elif x in outswitch_copy:
                insert = insert + [out_of_lane_up]
            elif random.randint(0, 99) < switch_prop and y != y_length-1:
                if random.randint(1, x_length) > x:
                    insert = insert + [out_of_lane_down]
                    inswitch.append(x)
                else:
                    insert = insert + [in_to_lane_down]
                    outswitch.append(x)
            else:
                insert = insert + [horizontal_straight]
        rails = rails + [insert]

    rail_map = np.array(rails, dtype=np.uint16)
    rail = GridTransitionMap(width=rail_map.shape[1], height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map

    city_positions = [(1, 0), (1, x_length-1),(2, 0), (2, x_length-1),(3, 0), (3, x_length-1),(4, 0), (4, x_length-1),]

    train_stations = [
        [((1, 0), 0), ((1, 0), 1)],
        [((1, x_length-1), 0), ((1, x_length-1), 1)],
        [((2, 0), 0), ((2, 0), 1)],
        [((2, x_length - 1), 0), ((2, x_length - 1), 1)],
        [((3, 0), 0), ((3, 0), 1)],
        [((3, x_length - 1), 0), ((3, x_length - 1), 1)],
        [((4, 0), 0), ((4, 0), 1)],
        [((4, x_length - 1), 0), ((4, x_length - 1), 1)],

    ]
    city_orientations = [1, 3, 1 , 3, 1, 3, 1, 3]
    agents_hints = {'city_positions': city_positions,
                    'train_stations': train_stations,
                    'city_orientations': city_orientations
                    }
    optionals = {'agents_hints': agents_hints}
    return rail, rail_map, optionals
def make_double_track1() -> Tuple[GridTransitionMap, np.array]:
    # We instantiate a very simple rail network on a 10x18 grid:
    # Note that that cells have invalid RailEnvTransitions!
    #
    #
    #    _ _ _ _ _ _ _ _ _ _
    #       / \     / \
    #    _ _ _ _ _ _ _ _ _ _
    #
    #
    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    empty = cells[0]
    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
    simple_switch_north_left = cells[2]
    simple_switch_north_right = cells[10]
    simple_switch_west_right = transitions.rotate_transition(simple_switch_north_right, 270)
    simple_switch_west_left = transitions.rotate_transition(simple_switch_north_left, 270)
    simple_switch_east_right = transitions.rotate_transition(simple_switch_north_right, 90)
    simple_switch_east_left = transitions.rotate_transition(simple_switch_north_left, 90)
    rail_map = np.array(
        [[empty] * 18] * 4 +
        [[dead_end_from_east] + [horizontal_straight] * 4 + [simple_switch_west_left] + [simple_switch_east_right] +
         [horizontal_straight] * 4 + [simple_switch_west_left] + [simple_switch_east_right] + [horizontal_straight] * 4 +
         [dead_end_from_west]] +
        [[dead_end_from_east] + [horizontal_straight] * 4 + [simple_switch_east_left] + [simple_switch_west_right] +
         [horizontal_straight] * 4 + [simple_switch_east_left] + [simple_switch_west_right] + [horizontal_straight] * 4 +
         [dead_end_from_west]] +
        [[empty] * 18] * 4, dtype=np.uint16)
    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map
    city_positions = [(5, 0), (5, 16)]
    train_stations = [
                      [((4, 1), 0), ((5, 1), 1)],
                      [((4, 16), 0), ((5, 17), 1)],
                     ]
    city_orientations = [1, 3]
    agents_hints = {'city_positions': city_positions,
                    'train_stations': train_stations,
                    'city_orientations': city_orientations
                   }
    optionals = {'agents_hints': agents_hints}
    return rail, rail_map, optionals
def make_pathfinding_track(size) -> Tuple[GridTransitionMap, np.array]:
    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    empty = cells[0]
    right_turn_from_south = cells[8]
    test = cells[5]
    test2 = transitions.rotate_transition(test, 90)
    test3 = [test] + [test2]
    test4 = cells[6]
    test5 = transitions.rotate_transition(test4, 180)
    test6 = transitions.rotate_transition(test4, 90)
    test7 = transitions.rotate_transition(test4, 270)

    right_turn_from_north = transitions.rotate_transition(right_turn_from_south, 180)
    right_turn_from_west = transitions.rotate_transition(right_turn_from_south, 90)
    right_turn_from_east = transitions.rotate_transition(right_turn_from_south, 270)

    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    dead_end_from_north = transitions.rotate_transition(dead_end_from_south, 180)
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
    simple_switch_north_left = cells[2]
    simple_switch_north_right = cells[10]
    simple_switch_west_right = transitions.rotate_transition(simple_switch_north_right, 270)
    simple_switch_west_left = transitions.rotate_transition(simple_switch_north_left, 270)
    simple_switch_east_right = transitions.rotate_transition(simple_switch_north_right, 90)
    simple_switch_east_left = transitions.rotate_transition(simple_switch_north_left, 90)
    simple_switch_south_left = transitions.rotate_transition(simple_switch_north_left, 180)
    simple_switch_south_right = transitions.rotate_transition(simple_switch_north_right, 180)

    rails = []
    for y in range(size):
        insert = []
        for x in range(size):
            if x == 0 and y == 0:
                insert = [right_turn_from_south]
            elif x == size - 1 and y == 0:
                insert = insert + [right_turn_from_west]
            elif x == size-1 and y == size - 1:
                insert = insert + [right_turn_from_north]
            elif x == 0 and y == size-1:
                insert = insert + [right_turn_from_east]
            elif y == 0:
                insert = insert + [test4]
            elif y == size-1:
                insert = insert + [test5]
            elif x == 0:
                insert = insert + [test7]
            elif x == size - 1:
                insert = insert + [test6]
            elif (x % 2 == 0 and y%2 == 1) or (x % 2 == 1 and y%2 == 0):
                insert = insert + [test]
            else:
                insert = insert + [test2]

        rails = rails + [insert]


    rail_map = np.array(rails, dtype=np.uint16)

    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map
    random.seed(time.clock())
    city_positions = [(random.randint(0, size-1), random.randint(0, size-1)), (random.randint(0, size-1),random.randint(0, size-1)), (random.randint(0, size-1),random.randint(0, size-1)), (random.randint(0, size-1),random.randint(0, size-1))]
    train_stations = [#Ziele (y,x) entscheided wo die ziele sind
                      [((random.randint(0, size-1), random.randint(0, size-1)), 0)],
                      [((random.randint(0, size-1), random.randint(0, size-1)), 0)],
                      [((random.randint(0, size - 1), random.randint(0, size - 1)), 0)],
                      [((random.randint(0, size - 1), random.randint(0, size - 1)), 0)],
                      [((random.randint(0, size - 1), random.randint(0, size - 1)), 0)],
                     ]
    city_orientations = [1,3, 1,3]
    agents_hints = {'city_positions': city_positions,
                    'train_stations': train_stations,
                    'city_orientations': city_orientations
                   }
    optionals = {'agents_hints': agents_hints}
    return rail, rail_map, optionals

def make_malfunction_training(height, length) -> Tuple[GridTransitionMap, np.array]:
    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    empty = cells[0]
    right_turn_from_south = cells[8]
    test = cells[3]
    right_turn_from_north = transitions.rotate_transition(right_turn_from_south, 180)
    right_turn_from_west = transitions.rotate_transition(right_turn_from_south, 90)
    right_turn_from_east = transitions.rotate_transition(right_turn_from_south, 270)

    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    dead_end_from_north = transitions.rotate_transition(dead_end_from_south, 180)
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
    simple_switch_north_left = cells[2]
    simple_switch_north_right = cells[10]
    simple_switch_west_right = transitions.rotate_transition(simple_switch_north_right, 270)
    simple_switch_west_left = transitions.rotate_transition(simple_switch_north_left, 270)
    simple_switch_east_right = transitions.rotate_transition(simple_switch_north_right, 90)
    simple_switch_east_left = transitions.rotate_transition(simple_switch_north_left, 90)
    simple_switch_south_left = transitions.rotate_transition(simple_switch_north_left, 180)
    simple_switch_south_right = transitions.rotate_transition(simple_switch_north_right, 180)

    rail_map = np.array(
        [[dead_end_from_east] + [simple_switch_east_right] + [horizontal_straight] * length + [simple_switch_west_left] + [dead_end_from_west]] +
        [[empty] + [simple_switch_south_left] + [horizontal_straight] * length + [simple_switch_south_right] + [empty]] * height +
        [[empty] + [right_turn_from_east] + [horizontal_straight] * length + [right_turn_from_north] + [empty]]
    , dtype=np.uint16)
    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map
    city_positions = [(0, 0), (0,0)]
    random.seed(time.clock())
    if random.randint(0,1)== 1:
        trainstations = [[((0,0),0), ((0, length + 3), 1)], [((0, 0), 0), ((0, length + 3), 1)]]
    else:
        trainstations = [[((0, length + 3), 1), ((0, 0), 0)], [((0, length + 3), 1), ((0, 0), 0)]]

    train_stations = trainstations
    city_orientations = [1, 1]
    agents_hints = {'city_positions': city_positions,
                    'train_stations': train_stations,
                    'city_orientations': city_orientations
                   }
    optionals = {'agents_hints': agents_hints}
    return rail, rail_map, optionals



























def make_simple_rail2() -> Tuple[GridTransitionMap, np.array]:
    # We instantiate a very simple rail network on a 7x10 grid:
    #        |
    #        |
    #        |
    # _ _ _ _\ _ _  _  _ _ _
    #               \
    #                |
    #                |
    #                |
    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    empty = cells[0]
    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_north = transitions.rotate_transition(dead_end_from_south, 180)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
    simple_switch_north_right = cells[10]
    simple_switch_east_west_north = transitions.rotate_transition(simple_switch_north_right, 270)
    simple_switch_west_east_south = transitions.rotate_transition(simple_switch_north_right, 90)
    rail_map = np.array(
        [[empty] * 3 + [dead_end_from_south] + [empty] * 6] +
        [[empty] * 3 + [vertical_straight] + [empty] * 6] * 2 +
        [[dead_end_from_east] + [horizontal_straight] * 2 +
         [simple_switch_east_west_north] +
         [horizontal_straight] * 2 + [simple_switch_west_east_south] +
         [horizontal_straight] * 2 + [dead_end_from_west]] +
        [[empty] * 6 + [vertical_straight] + [empty] * 3] * 2 +
        [[empty] * 6 + [dead_end_from_north] + [empty] * 3], dtype=np.uint16)
    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map
    city_positions = [(0,3), (6, 6)]
    train_stations = [
                      [( (0, 3), 0 ) ],
                      [( (6, 6), 0 ) ],
                     ]
    city_orientations = [0, 2]
    agents_hints = {'city_positions': city_positions,
                    'train_stations': train_stations,
                    'city_orientations': city_orientations
                   }
    optionals = {'agents_hints': agents_hints}
    return rail, rail_map, optionals


def make_simple_rail_with_alternatives() -> Tuple[GridTransitionMap, np.array]:
    # We instantiate a very simple rail network on a 7x10 grid:
    #  0 1 2 3 4 5 6 7 8 9  10
    # 0        /-------------\
    # 1        |             |
    # 2        |             |
    # 3 _ _ _ /_  _ _        |
    # 4              \   ___ /
    # 5               |/
    # 6               |
    # 7               |
    transitions = RailEnvTransitions()
    cells = transitions.transition_list

    empty = cells[0]
    dead_end_from_south = cells[7]
    right_turn_from_south = cells[8]
    right_turn_from_west = transitions.rotate_transition(right_turn_from_south, 90)
    right_turn_from_north = transitions.rotate_transition(right_turn_from_south, 180)
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_north = transitions.rotate_transition(dead_end_from_south, 180)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    vertical_straight = cells[1]
    simple_switch_north_left = cells[2]
    simple_switch_north_right = cells[10]
    simple_switch_left_east = transitions.rotate_transition(simple_switch_north_left, 90)
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
    double_switch_south_horizontal_straight = horizontal_straight + cells[6]
    double_switch_north_horizontal_straight = transitions.rotate_transition(
        double_switch_south_horizontal_straight, 180)
    rail_map = np.array(
        [[empty] * 3 + [right_turn_from_south] + [horizontal_straight] * 5 + [right_turn_from_west]] +
        [[empty] * 3 + [vertical_straight] + [empty] * 5 + [vertical_straight]] * 2 +
        [[dead_end_from_east] + [horizontal_straight] * 2 + [simple_switch_left_east] + [horizontal_straight] * 2 + [
            right_turn_from_west] + [empty] * 2 + [vertical_straight]] +
        [[empty] * 6 + [simple_switch_north_right] + [horizontal_straight] * 2 + [right_turn_from_north]] +
        [[empty] * 6 + [vertical_straight] + [empty] * 3] +
        [[empty] * 6 + [dead_end_from_north] + [empty] * 3], dtype=np.uint16)
    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map
    city_positions = [(0,0), (0, 0)]
    train_stations = [
                      [( (0, 9), 0 ) ],
                      [( (6, 6), 0 ) ],
                     ]
    city_orientations = [0, 2]
    agents_hints = {'city_positions': city_positions,
                    'train_stations': train_stations,
                    'city_orientations': city_orientations
                   }
    optionals = {'agents_hints': agents_hints}
    return rail, rail_map, optionals




def sarsa_qlearning_comaparison() -> Tuple[GridTransitionMap, np.array]:
    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    empty = cells[0]

    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
    simple_switch_north_left = cells[2]
    simple_switch_north_right = cells[10]
    simple_switch_west_right = transitions.rotate_transition(simple_switch_north_right, 270)
    simple_switch_west_left = transitions.rotate_transition(simple_switch_north_left, 270)
    simple_switch_east_right = transitions.rotate_transition(simple_switch_north_right, 90)
    simple_switch_east_left = transitions.rotate_transition(simple_switch_north_left, 90)

    rail_map = np.array(
        [[empty] * 18] * 3+
        [[empty] * 7 + [dead_end_from_south] * 4 + [empty] * 7] +
        [[dead_end_from_east] + [horizontal_straight] * 5 + [simple_switch_east_right] +
         [simple_switch_east_left] * 4 + [simple_switch_west_left] + [horizontal_straight] * 5 +
         [dead_end_from_west]] +

        [[empty] * 6 + [simple_switch_west_right] +
         [horizontal_straight] * 4 + [simple_switch_east_left] + [empty] * 6] +
        [[empty] * 18] * 4, dtype=np.uint16)
    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map
    city_positions = [(4, 0), (4, 16)]
    train_stations = [
                      [((4, 1), 0), ((4, 1), 1)],
                      [((4, 16), 0), ((4, 17), 1)],
                     ]
    city_orientations = [1, 3]
    agents_hints = {'city_positions': city_positions,
                    'train_stations': train_stations,
                    'city_orientations': city_orientations
                   }
    optionals = {'agents_hints': agents_hints}
    return rail, rail_map, optionals
def deadlock_example_1() -> Tuple[GridTransitionMap, np.array]:
    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    empty = cells[0]
    right_turn_from_south = cells[8]
    test = cells[3]
    right_turn_from_north = transitions.rotate_transition(right_turn_from_south, 180)
    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    dead_end_from_north = transitions.rotate_transition(dead_end_from_south, 180)
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
    simple_switch_north_left = cells[2]
    simple_switch_north_right = cells[10]
    simple_switch_west_right = transitions.rotate_transition(simple_switch_north_right, 270)
    simple_switch_west_left = transitions.rotate_transition(simple_switch_north_left, 270)
    simple_switch_east_right = transitions.rotate_transition(simple_switch_north_right, 90)
    simple_switch_east_left = transitions.rotate_transition(simple_switch_north_left, 90)

    rail_map = np.array(
        [[empty] * 4 + [dead_end_from_south] + [empty] * 2] +
        [[empty] * 4 + [vertical_straight] + [empty] * 2] +
        [[empty] * 2 + [right_turn_from_south] + [horizontal_straight] + [test] + [horizontal_straight] + [dead_end_from_west]] +
        [[empty] * 2 + [vertical_straight] + [empty] + [vertical_straight] + [empty] * 2] +
        [[dead_end_from_east] + [horizontal_straight] + [test] + [horizontal_straight] + [right_turn_from_north] + [empty] * 2] +
        [[empty] * 2 + [vertical_straight] + [empty] * 4] +
        [[empty] * 2 + [dead_end_from_north] + [empty] * 4]
, dtype=np.uint16)
    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map
    city_positions = [(4, 0), (2, 6), (0, 4), (6, 2)]
    train_stations = [#Ziele (y,x) entscheided wo die ziele sind
                      [((4, 0), 0), ((0, 4), 1)],
                      [((2, 6), 0), ((6, 2), 1)],
                      [((2, 6), 0), ((6, 2), 1)],
                      [((2, 6), 0), ((6, 2), 1)],
                     ]
    city_orientations = [2, 3, 1, 1]
    agents_hints = {'city_positions': city_positions,
                    'train_stations': train_stations,
                    'city_orientations': city_orientations
                   }
    optionals = {'agents_hints': agents_hints}
    return rail, rail_map, optionals
def make_deadlock_example_2() -> Tuple[GridTransitionMap, np.array]:
    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    empty = cells[0]
    right_turn_from_south = cells[8]
    test = cells[3]
    right_turn_from_north = transitions.rotate_transition(right_turn_from_south, 180)
    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    dead_end_from_north = transitions.rotate_transition(dead_end_from_south, 180)
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
    simple_switch_north_left = cells[2]
    simple_switch_north_right = cells[10]
    simple_switch_west_right = transitions.rotate_transition(simple_switch_north_right, 270)
    simple_switch_west_left = transitions.rotate_transition(simple_switch_north_left, 270)
    simple_switch_east_right = transitions.rotate_transition(simple_switch_north_right, 90)
    simple_switch_east_left = transitions.rotate_transition(simple_switch_north_left, 90)

    rail_map = np.array(
        [[empty] * 7] +
        [[empty] * 7] +
        [[empty] * 7] +
        [[dead_end_from_east] + [horizontal_straight]*5 + [dead_end_from_west]] +
        [[empty] * 7] +
        [[empty] * 7] +
        [[empty] * 7]
, dtype=np.uint16)
    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map
    city_positions = [(3, 0), (3, 6)]
    train_stations = [#Ziele (y,x) entscheided wo die ziele sind
                      [((3, 0), 0),((3, 0), 1)],
                      [((3, 6), 0),((3,6), 1)],
                     ]
    city_orientations = [2, 3]
    agents_hints = {'city_positions': city_positions,
                    'train_stations': train_stations,
                    'city_orientations': city_orientations
                   }
    optionals = {'agents_hints': agents_hints}
    return rail, rail_map, optionals
def make_deadklock_example_3() -> Tuple[GridTransitionMap, np.array]:
    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    empty = cells[0]
    right_turn_from_south = cells[8]
    test = cells[3]
    right_turn_from_north = transitions.rotate_transition(right_turn_from_south, 180)
    right_turn_from_west = transitions.rotate_transition(right_turn_from_south, 90)
    right_turn_from_east = transitions.rotate_transition(right_turn_from_south, 270)

    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    dead_end_from_north = transitions.rotate_transition(dead_end_from_south, 180)
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
    simple_switch_north_left = cells[2]
    simple_switch_north_right = cells[10]
    simple_switch_west_right = transitions.rotate_transition(simple_switch_north_right, 270)
    simple_switch_west_left = transitions.rotate_transition(simple_switch_north_left, 270)
    simple_switch_east_right = transitions.rotate_transition(simple_switch_north_right, 90)
    simple_switch_east_left = transitions.rotate_transition(simple_switch_north_left, 90)

    rail_map = np.array(
        [[right_turn_from_south] + [horizontal_straight]*4 + [right_turn_from_west] + [empty]] +
        [[vertical_straight] + [empty] * 4 + [vertical_straight] + [empty]] +
        [[right_turn_from_east] + [horizontal_straight] + [simple_switch_west_left] + [horizontal_straight]*2 + [simple_switch_north_left] + [empty]] +
        [[empty] * 2 + [vertical_straight] + [empty]*2 + [vertical_straight] + [empty]] +
        [[empty] * 2 + [vertical_straight] + [empty]*2 + [vertical_straight] + [empty]] +
        [[empty] * 2 + [right_turn_from_east] + [horizontal_straight]*2 + [right_turn_from_north] + [empty]] +
        [[empty] * 7]
, dtype=np.uint16)
    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map
    city_positions = [(2, 0), (0, 5)]
    train_stations = [#Ziele (y,x) entscheided wo die ziele sind
                      [((2, 0), 0)],
                      [((0, 5), 0)],
                     ]
    city_orientations = [2,3]
    agents_hints = {'city_positions': city_positions,
                    'train_stations': train_stations,
                    'city_orientations': city_orientations
                   }
    optionals = {'agents_hints': agents_hints}
    return rail, rail_map, optionals
def make_chainreaction() -> Tuple[GridTransitionMap, np.array]:
    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    empty = cells[0]
    right_turn_from_south = cells[8]
    test = cells[3]
    right_turn_from_north = transitions.rotate_transition(right_turn_from_south, 180)
    right_turn_from_west = transitions.rotate_transition(right_turn_from_south, 90)
    right_turn_from_east = transitions.rotate_transition(right_turn_from_south, 270)

    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    dead_end_from_north = transitions.rotate_transition(dead_end_from_south, 180)
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
    simple_switch_north_left = cells[2]
    simple_switch_north_right = cells[10]
    simple_switch_west_right = transitions.rotate_transition(simple_switch_north_right, 270)
    simple_switch_west_left = transitions.rotate_transition(simple_switch_north_left, 270)
    simple_switch_east_right = transitions.rotate_transition(simple_switch_north_right, 90)
    simple_switch_east_left = transitions.rotate_transition(simple_switch_north_left, 90)

    rail_map = np.array(
        [[dead_end_from_east] + [horizontal_straight] *5 + [right_turn_from_west]] +
        [[empty] * 5 + [right_turn_from_south] + [simple_switch_north_left]] +
        [[empty] * 5 + [vertical_straight] + [vertical_straight]] +
        [[empty] * 5 + [vertical_straight] + [vertical_straight]] +
        [[dead_end_from_east] + [right_turn_from_west] + [empty] * 3 + [vertical_straight]*2] +
        [[dead_end_from_east] + [simple_switch_west_right] + [horizontal_straight]*2 + [simple_switch_east_right] + [simple_switch_west_right] + [right_turn_from_north]] +
        [[empty] * 4 + [right_turn_from_east] + [horizontal_straight] + [dead_end_from_west]]
, dtype=np.uint16)
    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map
    city_positions = [(5, 0), (4, 0), (0,0)]
    train_stations = [#Ziele (y,x) entscheided wo die ziele sind
                      [((4, 0), 0)],
                      [((5, 0), 0)],
                      [((6, 6), 0)],
                     ]
    city_orientations = [2,3,4]
    agents_hints = {'city_positions': city_positions,
                    'train_stations': train_stations,
                    'city_orientations': city_orientations
                   }
    optionals = {'agents_hints': agents_hints}
    return rail, rail_map, optionals
def malfunction_test() -> Tuple[GridTransitionMap, np.array]:
    transitions = RailEnvTransitions()
    cells = transitions.transition_list
    empty = cells[0]
    right_turn_from_south = cells[8]
    test = cells[3]
    right_turn_from_north = transitions.rotate_transition(right_turn_from_south, 180)
    right_turn_from_west = transitions.rotate_transition(right_turn_from_south, 90)
    right_turn_from_east = transitions.rotate_transition(right_turn_from_south, 270)

    dead_end_from_south = cells[7]
    dead_end_from_west = transitions.rotate_transition(dead_end_from_south, 90)
    dead_end_from_east = transitions.rotate_transition(dead_end_from_south, 270)
    dead_end_from_north = transitions.rotate_transition(dead_end_from_south, 180)
    vertical_straight = cells[1]
    horizontal_straight = transitions.rotate_transition(vertical_straight, 90)
    simple_switch_north_left = cells[2]
    simple_switch_north_right = cells[10]
    simple_switch_west_right = transitions.rotate_transition(simple_switch_north_right, 270)
    simple_switch_west_left = transitions.rotate_transition(simple_switch_north_left, 270)
    simple_switch_east_right = transitions.rotate_transition(simple_switch_north_right, 90)
    simple_switch_east_left = transitions.rotate_transition(simple_switch_north_left, 90)

    rail_map = np.array(
        [[dead_end_from_east] + [simple_switch_east_right] + [horizontal_straight] * 3 + [simple_switch_west_left] + [dead_end_from_west]] +
        [[empty] + [right_turn_from_east] + [horizontal_straight]* 3 + [right_turn_from_north] + [empty]]
, dtype=np.uint16)
    rail = GridTransitionMap(width=rail_map.shape[1],
                             height=rail_map.shape[0], transitions=transitions)
    rail.grid = rail_map
    city_positions = [(0, 0), (0,6)]
    train_stations = [#Ziele (y,x) entscheided wo die ziele sind
                      [((0, 0), 0)],
                      [((0, 6), 0)],
                     ]
    city_orientations = [2,3]
    agents_hints = {'city_positions': city_positions,
                    'train_stations': train_stations,
                    'city_orientations': city_orientations
                   }
    optionals = {'agents_hints': agents_hints}
    return rail, rail_map, optionals
