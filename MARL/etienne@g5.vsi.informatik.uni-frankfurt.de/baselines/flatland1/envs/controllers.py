import numpy as np
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths
from flatland.envs.step_utils.states import TrainState
"""
Here we define the different controllers for our agents.
Each controller has an act-function that returns a dictionary containing an action for each agent.
"""


class ConstantGoForwardController:
    def __init__(self):
        pass

    def act(self, observations, env):
        actions = dict()
        for agent_handle, agent in enumerate(env.agents):
            # always go forward
            action = 2
            actions.update({agent_handle: action})
        return actions


class RandomController:
    def __init__(self, action_size):
        self.action_size = action_size

    def act(self, observations, env):
        actions = dict()
        for agent_handle, agent in enumerate(env.agents):
            # always choose random action
            action = np.random.randint(self.action_size)
            actions.update({agent_handle: action})
        return actions


class ShortestPathController:
    def __init__(self, env):
        self.shortest_paths = []
        for agent_handle, agent in enumerate(env.agents):
            shortest_path = agent.get_shortest_path(env.distance_map)
            self.shortest_paths.append(shortest_path)
            # for random environments we change the trains' orientation to be in the direction of the shortest path
            agent.initial_direction = self.shortest_paths[agent_handle][0][1]

    def act(self, observations, env):
        actions = dict()
        for agent_handle, agent in enumerate(env.agents):
            if agent.state == TrainState.DONE or agent.state == TrainState.MALFUNCTION \
                or agent.state == TrainState.MALFUNCTION_OFF_MAP or agent.state == TrainState.WAITING:
                action = 0
            else:
                if agent.position != self.shortest_paths[agent_handle][0][0] and agent.state != TrainState.READY_TO_DEPART:
                    self.shortest_paths[agent_handle].pop(0)
                # select action to follow shortest path
                action = action_on_selected_path(self.shortest_paths, agent_handle, agent.direction)
            actions.update({agent_handle: action})
        return actions


class SequentialController:
    def __init__(self, env):
        self.done_counter = 0
        self.shortest_paths = []
        for agent_handle, agent in enumerate(env.agents):
            shortest_path = agent.get_shortest_path(env.distance_map)
            self.shortest_paths.append(shortest_path)
            # for random environments we change the trains' orientation to be in the direction of the shortest path
            agent.initial_direction = self.shortest_paths[agent_handle][0][1]

    def act(self, observations, env):
        actions = dict()
        for agent_handle, agent in enumerate(env.agents):
            if agent.state == TrainState.DONE:
                self.done_counter = agent_handle + 1
            if agent_handle > self.done_counter:
                action = 0
            else:
                action = action_on_selected_path(self.shortest_paths, agent_handle, agent.direction)
            actions.update({agent_handle: action})
        return actions


class BlockPathController:
    """
    The controller has two attributes, so that they don't have to be computed again, but can be manipulated in place.
    The dictionary occupied has two lists for each cell, one containing the agent_handles of the agent, for which the
    cell is blocked, the other containing the directions of the agents when traversing the cell.
    The list shortest_paths contains the shortest paths for each agent.
    """

    def __init__(self, env):
        self.occupied = {(x, y): [[], []] for x in range(env.height) for y in range(env.width)}
        self.shortest_paths = []
        for agent_handle, agent in enumerate(env.agents):
            shortest_path = agent.get_shortest_path(env.distance_map)
            self.shortest_paths.append(shortest_path)

    def act(self, observations, env):
        actions = dict()
        for agent_handle, agent in enumerate(env.agents):
            if agent.state == TrainState.WAITING or agent.state == TrainState.MALFUNCTION \
                or agent.state == TrainState.MALFUNCTION_OFF_MAP:
                action = 0
            elif agent.state == TrainState.DONE:
                if self.shortest_paths[agent_handle]:
                    self.occupied[(self.shortest_paths[agent_handle][0][0][0], \
                                   self.shortest_paths[agent_handle][0][0][1])][0].remove(agent_handle)
                    self.occupied[(self.shortest_paths[agent_handle][0][0][0], \
                                   self.shortest_paths[agent_handle][0][0][1])][1].remove(
                        self.shortest_paths[agent_handle][0][1])
                    self.shortest_paths[agent_handle].pop(0)

                x = agent.target[0]
                y = agent.target[1]
                if self.occupied[(x, y)][0] and agent_handle in self.occupied[(x, y)][0]:
                    self.occupied[(x, y)][0].remove(agent_handle)
                    self.occupied[(x, y)][1].remove(agent.direction)
                if self.shortest_paths[agent_handle]:
                    self.shortest_paths[agent_handle].pop(0)
                action = 0
            elif agent.state == TrainState.READY_TO_DEPART:
                if is_bsfw_clear_in_direction(self.occupied, self.shortest_paths, agent_handle):
                    action = action_on_selected_path(self.shortest_paths, agent_handle, agent.direction)
                    if agent_handle not in self.occupied[agent.initial_position][0]:
                        self.occupied = block_bsfw(self.occupied, self.shortest_paths, agent_handle)
                else:
                    action = 0
            else:
                if agent.position != self.shortest_paths[agent_handle][0][0]:
                    self.occupied[(self.shortest_paths[agent_handle][0][0][0], \
                                   self.shortest_paths[agent_handle][0][0][1])][0].remove(agent_handle)
                    self.occupied[(self.shortest_paths[agent_handle][0][0][0], \
                                   self.shortest_paths[agent_handle][0][0][1])][1].remove(self.shortest_paths[agent_handle][0][1])
                    self.shortest_paths[agent_handle].pop(0)
                action = action_on_selected_path(self.shortest_paths, agent_handle, agent.direction)
            actions.update({agent_handle: action})
        return actions


class RHSController:
    """
    The controller has two dictionaries and a listas fixed attributes. The first dictionary specifies, in which direction
    the rails between cities may be driven on. The second dictionary gives the driveway for a pair of origin/destination
    locations following tracks on the right side. The list contains the remaining path for every agent and may be manipulated
    in place.
    """

    def __init__(self, env):
        self.occupied = {}
        self.driveways = {}
        self.shortest_paths = []

        for y in list(range(12, 19)) + list(range(21, 27)) + list(range(28, 37)) + list(range(40, 54)) + list(
            range(56, 68)):
            self.occupied[(8, y)] = 3
            self.occupied[(9, y)] = 1

        for agent_handle, agent in enumerate(env.agents):
            origin = agent.initial_position
            destination = agent.target
            direction = agent.initial_direction
            if (origin, destination) not in self.driveways:
                k = 1
                driveway_not_found = True
                while driveway_not_found:
                    possible_paths = get_k_shortest_paths(env, origin, direction, destination, k)
                    if path_on_regelgleis(possible_paths[-1], self.occupied):
                        driveway_not_found = False
                        self.driveways[(origin, destination)] = list(possible_paths[-1])
                    else:
                        k += 1

            self.shortest_paths.append(self.driveways.get((origin, destination))[:])

    def act(self, observations, env):
        actions = dict()
        for agent_handle, agent in enumerate(env.agents):
            if agent.state == TrainState.WAITING or agent.state == TrainState.DONE or agent.state == \
                TrainState.MALFUNCTION or agent.state == TrainState.MALFUNCTION_OFF_MAP:
                action = 0
            else:
                if agent.position != self.shortest_paths[agent_handle][0][0] and agent.state != TrainState.READY_TO_DEPART:
                    self.shortest_paths[agent_handle].pop(0)
                action = action_on_selected_path(self.shortest_paths, agent_handle, agent.direction)
            actions.update({agent_handle: action})
        return actions


class RHSandClusterController:
    def __init__(self, env):
        # define which cells belong to each switching area
        self.Clusters = [[(9, 19), (9, 20)] + [(x, y) for x in range(10, 15) for y in range(12, 23)], \
                         [(7, 39), (7, 40), (7, 41)] + [(x, y) for x in range(8, 12) for y in range(37, 40)] + \
                         [(12, y) for y in range(37, 42)] + [(x, y) for x in range(13, 20) for y in range(37, 47)], \
                         [(9, 54), (9, 55)] + [(x, y) for x in range(10, 13) for y in range(41, 60)] + \
                         [(x, y) for x in range(13, 20) for y in range(46, 60)], \
                         [(7, 68), (7, 69), (7, 70), (7, 71), (8, 68), (8, 69), (9, 68)]]
        # define which cells have a fixed driving direction
        self.occupied = {}
        # dictionary with (origin, destination) as key and valid driveway from origin to destination as value
        self.driveways = {}
        # list of chosen paths for every agent
        self.shortest_paths = []
        # list to keep track of trains in switching areas
        self.occupied_clusters = [[], [], [], []]
        # list of entry and exit points of the clusters for every agent
        self.cluster_connections = [[] for x in range(env.get_num_agents())]
        # matrices to keep track of valid connections in each switching area
        self.switching_matrices = np.array([np.zeros((6, 6), dtype=int), np.zeros((12, 12), dtype=int),
                                            np.zeros((10, 10), dtype=int), np.zeros((5, 5), dtype=int)])

        for y in list(range(12, 19)) + list(range(21, 27)) + list(range(28, 37)) + list(range(40, 54)) + list(
            range(56, 68)):
            self.occupied[(8, y)] = 3
            self.occupied[(9, y)] = 1

        for agent_handle, agent in enumerate(env.agents):
            origin = agent.initial_position
            destination = agent.target
            direction = agent.initial_direction
            if (origin, destination) not in self.driveways:
                k = 1
                driveway_not_found = True
                while driveway_not_found:
                    possible_paths = get_k_shortest_paths(env, origin, direction, destination, k)
                    if path_on_regelgleis(possible_paths[-1], self.occupied):
                        driveway_not_found = False
                        self.driveways[(origin, destination)] = list(possible_paths[-1])
                    else:
                        k += 1

            self.shortest_paths.append(self.driveways.get((origin, destination))[:])
            self.cluster_connections[agent_handle] = (get_cluster_connections(self.driveways.get((origin, destination))[:]))

    def act(self, observations, env):
        actions = dict()
        self.switching_matrices = update_connections(self.switching_matrices, self.occupied_clusters, self.shortest_paths)
        for agent_handle, agent in enumerate(env.agents):
            if agent.state == TrainState.WAITING or agent.state == TrainState.MALFUNCTION_OFF_MAP:
                action = 0
            elif agent.state == TrainState.MALFUNCTION:
                print('Agent {} has a malfunction for {} more steps'.format(agent_handle, agent.malfunction_handler.malfunction_down_counter))
                action = 0
            elif agent.state == TrainState.DONE:
                x = agent.target[0]
                y = agent.target[1]
                clusters = in_which_clusters(self, (x, y))
                for i in range(len(clusters)):
                    if agent_handle in self.occupied_clusters[clusters[i]]:
                        self.occupied_clusters[clusters[i]].remove(agent_handle)
                action = 0
            elif agent.state == TrainState.READY_TO_DEPART:
                future_clusters = in_which_clusters(self, self.shortest_paths[agent_handle][1][0])
                if future_clusters:
                    origin = self.cluster_connections[agent_handle][future_clusters[0]][0]
                    destination = self.cluster_connections[agent_handle][future_clusters[0]][1]
                    if self.switching_matrices[future_clusters[0]][origin][destination] == 0:
                        action = 4
                    else:
                        self.occupied_clusters[future_clusters[0]].append(agent_handle)
                        self.switching_matrices[future_clusters[0]] = block_connections(future_clusters[0],
                                                                                            self.occupied_clusters[future_clusters[0]],
                                                                                            self.cluster_connections)
                        action = action_on_selected_path(self.shortest_paths, agent_handle, agent.direction)
                else:
                    action = action_on_selected_path(self.shortest_paths, agent_handle, agent.direction)
            else:
                if agent.position != self.shortest_paths[agent_handle][0][0]:
                    self.shortest_paths[agent_handle].pop(0)
                clusters_old = in_which_clusters(self, self.shortest_paths[agent_handle][0][0])
                clusters_new = in_which_clusters(self, self.shortest_paths[agent_handle][1][0])
                if clusters_old and clusters_new:
                    action = action_on_selected_path(self.shortest_paths, agent_handle, agent.direction)
                elif clusters_new:
                    origin = self.cluster_connections[agent_handle][clusters_new[0]][0]
                    destination = self.cluster_connections[agent_handle][clusters_new[0]][1]
                    if self.switching_matrices[clusters_new[0]][origin][destination] == 0:
                        action = 4
                    else:
                        self.occupied_clusters[clusters_new[0]].append(agent_handle)
                        action = action_on_selected_path(self.shortest_paths, agent_handle, agent.direction)
                        self.switching_matrices[clusters_new[0]] = block_connections(clusters_new[0],
                                                                                         self.occupied_clusters[
                                                                                             clusters_new[0]],
                                                                                         self.cluster_connections)
                elif clusters_old:
                    action = action_on_selected_path(self.shortest_paths, agent_handle, agent.direction)
                    if agent_handle in self.occupied_clusters[clusters_old[0]]:
                        self.occupied_clusters[clusters_old[0]].remove(agent_handle)
                else:
                    action = action_on_selected_path(self.shortest_paths, agent_handle, agent.direction)
            actions.update({agent_handle: action})
        return actions


def in_which_clusters(self, pos):
    clusters = []
    for i in range(4):
        if (pos in self.Clusters[i]):
            clusters.append(i)
    return clusters


def action_on_selected_path(shortest_paths, agent_handle, direction):
    if shortest_paths[agent_handle][1][1] == (direction - 1) % 4:
        action = 1
    elif shortest_paths[agent_handle][1][1] == (direction + 1) % 4:
        action = 3
    else:
        action = 2
    return action


def path_on_regelgleis(possible_path, occupied):
    result = True
    for i in range(len(possible_path)):
        if possible_path[i][0] in occupied and possible_path[i][1] != occupied.get(possible_path[i][0]):
            return False
    return result


def is_bsfw_clear_in_direction(occupied, shortest_paths, agent_handle):
    result = 1
    for i in range(len(shortest_paths[agent_handle])):
        x = shortest_paths[agent_handle][i][0][0]
        y = shortest_paths[agent_handle][i][0][1]
        direction = shortest_paths[agent_handle][i][1]
        other_directions = [0, 1, 2, 3]
        other_directions.remove(direction)
        if any(item in occupied[(x, y)][1] for item in other_directions):
            print('Cell ({},{}) cannot be accessed by agent {}'.format(x, y, agent_handle))
            return 0
    return result


def block_bsfw(occupied, shortest_paths, agent_handle):
    for i in range(len(shortest_paths[agent_handle])):
        x = shortest_paths[agent_handle][i][0][0]
        y = shortest_paths[agent_handle][i][0][1]
        direction = shortest_paths[agent_handle][i][1]
        occupied[(x, y)][0].append(agent_handle)
        occupied[(x, y)][1].append(direction)
    return occupied


def get_cluster_connections(path):
    list_of_cells = []
    result = [[], [], [], []]
    for i in range(len(path)):
        list_of_cells.append(path[i][0])
    cluster0left = [(9, 18), (11, 14), (13, 13), (14, 14)]
    cluster0right = [(8, 20), (9, 21)]
    cluster1left = [(8, 36), (9, 36)]
    cluster1right = [(7, 41), (8, 47), (9, 47), (12, 41), (13, 46), (14, 46), (15, 46), (16, 46), (17, 46), (18, 46)]
    cluster2left = [(9, 53), (12, 41), (13, 46), (14, 46), (15, 46), (16, 46), (17, 46), (18, 46)]
    cluster2right = [(8, 55), (9, 56)]
    cluster3left = [(8, 67), (9, 67)]
    cluster3right = [(7, 71), (8, 71), (9, 71)]
    node0left = get_cluster_node_in_path(cluster0left, list_of_cells)
    node0right = get_cluster_node_in_path(cluster0right, list_of_cells)
    if node0right:
        node0right[0] += 4
    node1left = get_cluster_node_in_path(cluster1left, list_of_cells)
    node1right = get_cluster_node_in_path(cluster1right, list_of_cells)
    if node1right:
        node1right[0] += 2
    node2left = get_cluster_node_in_path(cluster2left, list_of_cells)
    node2right = get_cluster_node_in_path(cluster2right, list_of_cells)
    if node2right:
        node2right[0] += 8
    node3left = get_cluster_node_in_path(cluster3left, list_of_cells)
    node3right = get_cluster_node_in_path(cluster3right, list_of_cells)
    if node3right:
        node3right[0] += 2
    if node0left and node0right:
        if node0left[1] < node0right[1]:
            result[0] = [node0left[0], node0right[0]]
        else:
            result[0] = [node0right[0], node0left[0]]
    if node1left and node1right:
        if node1left[1] < node1right[1]:
            result[1] = [node1left[0], node1right[0]]
        else:
            result[1] = [node1right[0], node1left[0]]
    if node2left and node2right:
        if node2left[1] < node2right[1]:
            result[2] = [node2left[0], node2right[0]]
        else:
            result[2] = [node2right[0], node2left[0]]
    if node3left and node3right:
        if node3left[1] < node3right[1]:
            result[3] = [node3left[0], node3right[0]]
        else:
            result[3] = [node3right[0], node3left[0]]
    return result


def get_cluster_node_in_path(cluster, path):
    result = []
    for cell in cluster:
        if cell in path:
            result = [cluster.index(cell), path.index(cell)]
    return result


def block_connections(index, list_of_agents, cluster_connections):
    if index == 0:
        matrix = np.zeros((6, 6), dtype=int)
    elif index == 1:
        matrix = np.zeros((12, 12), dtype=int)
    elif index == 2:
        matrix = np.zeros((10, 10), dtype=int)
    else:
        matrix = np.zeros((5, 5), dtype=int)
    for i in range(len(list_of_agents)):
        origin = cluster_connections[list_of_agents[i]][index][0]
        destination = cluster_connections[list_of_agents[i]][index][1]
        matrix[origin][destination] = 1
    return matrix


def update_connections(matrices, occupied_clusters, shortest_paths):
    result = np.array([np.zeros((6, 6)), np.zeros((12, 12)), np.zeros((10, 10)), np.zeros((5, 5))])
    default_matrices = np.array([[[0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 1],
                                [0, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0]],
                                 [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                 [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                                  [0, 1, 1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                                 [[0, 0, 0, 0, 0], [0, 0, 1, 0, 1], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [0, 0, 0, 0, 0]]])
    for i in range(4):
        if occupied_clusters[i]:
            result[i] = matrices[i]
            if i == 0:
                cells_needed = {(x, y): 1 for x in range(9, 15) for y in range(12, 23)}
                for j in occupied_clusters[i]:
                    for k in range(len(shortest_paths[j])):
                        cells_needed[shortest_paths[j][k][0]] = 0
                if cells_needed[(9, 19)] == 1 and cells_needed[(9, 20)] == 1:
                    result[0][0][5] = 1
                if cells_needed[(11, 14)] == 1 and cells_needed[(11, 16)] == 1 and cells_needed[(9, 19)] == 1 and \
                    cells_needed[(9, 20)] == 1:
                    result[0][1][5] = result[0][4][1] = 1
                if cells_needed[(13, 13)] == 1 and cells_needed[(13, 15)] == 1 and cells_needed[(11, 16)] == 1 and \
                    cells_needed[(9, 19)] == 1 and cells_needed[(9, 20)] == 1:
                    result[0][2][5] = result[0][4][2] = 1
                if cells_needed[(14, 14)] == 1 and cells_needed[(13, 14)] == 1 and cells_needed[(11, 16)] == 1 and \
                    cells_needed[(9, 19)] == 1 and cells_needed[(9, 20)] == 1:
                    result[0][3][5] = result[0][4][3] = 1
            elif i == 1:
                cells_needed = {(x, y): 1 for x in range(7, 20) for y in range(37, 47)}
                for j in occupied_clusters[i]:
                    for k in range(len(shortest_paths[j])):
                        cells_needed[shortest_paths[j][k][0]] = 0
                if cells_needed[(9, 37)] == 1 and cells_needed[(8, 38)] == 1 and cells_needed[(7, 41)] == 1:
                    result[1][1][2] = 1
                if cells_needed[(9, 37)] == 1 and cells_needed[(9, 39)] == 1:
                    result[1][1][4] = 1
                if cells_needed[(9, 37)] == 1 and cells_needed[(9, 39)] == 1 and cells_needed[(12, 39)] == 1 and \
                    cells_needed[(12, 41)] == 1:
                    result[1][1][5] = 1
                    if cells_needed[(8, 37)] == 1:
                        result[1][5][0]
                if cells_needed[(9, 37)] == 1 and cells_needed[(9, 39)] == 1 and cells_needed[(12, 39)] == 1 and \
                    cells_needed[(13, 40)] == 1 and cells_needed[(13, 46)] == 1:
                    result[1][1][6]
                    if cells_needed[(8, 37)] == 1:
                        result[1][6][0]
                if cells_needed[(9, 37)] == 1 and cells_needed[(9, 39)] == 1 and cells_needed[(12, 39)] == 1 and \
                    cells_needed[(13, 40)] == 1 and cells_needed[(14, 41)] == 1 and cells_needed[(14, 46)] == 1:
                    result[1][1][7] = 1
                    if cells_needed[(8, 37)] == 1:
                        result[1][7][0]
                if cells_needed[(9, 37)] == 1 and cells_needed[(9, 39)] == 1 and cells_needed[(12, 39)] == 1 and \
                    cells_needed[(13, 40)] == 1 and cells_needed[(14, 41)] == 1 and cells_needed[(15, 42)] == 1 and \
                    cells_needed[(15, 46)] == 1:
                    result[1][1][8] = 1
                    if cells_needed[(8, 37)] == 1:
                        result[1][8][0]
                if cells_needed[(9, 37)] == 1 and cells_needed[(9, 39)] == 1 and cells_needed[(12, 39)] == 1 and \
                    cells_needed[(13, 40)] == 1 and cells_needed[(14, 41)] == 1 and cells_needed[(15, 42)] == 1 and \
                    cells_needed[(16, 43)] == 1 and cells_needed[(16, 46)] == 1:
                    result[1][1][9] = 1
                    if cells_needed[(8, 37)] == 1:
                        result[1][9][0]
                if cells_needed[(9, 37)] == 1 and cells_needed[(9, 39)] == 1 and cells_needed[(12, 39)] == 1 and \
                    cells_needed[(13, 40)] == 1 and cells_needed[(14, 41)] == 1 and cells_needed[(15, 42)] == 1 and \
                    cells_needed[(16, 43)] == 1 and cells_needed[(17, 44)] == 1 and cells_needed[(17, 46)] ==1:
                    result[1][1][10] = 1
                    if cells_needed[(8, 37)] == 1:
                        result[1][10][0]
                if cells_needed[(9, 37)] == 1 and cells_needed[(9, 39)] == 1 and cells_needed[(12, 39)] == 1 and \
                    cells_needed[(13, 40)] == 1 and cells_needed[(14, 41)] == 1 and cells_needed[(15, 42)] == 1 and \
                    cells_needed[(16, 43)] == 1 and cells_needed[(17, 44)] == 1 and cells_needed[(18, 46)] == 1 and \
                    cells_needed[(18, 45)] == 1:
                    result[1][1][11] = 1
                    if cells_needed[(8, 37)] == 1:
                        result[1][11][0]
                if cells_needed[(8, 37)] == 1 and cells_needed[(8, 38)] == 1 and cells_needed[(7, 41)] == 1:
                    result[1][2][0] = 1
                if cells_needed[(8, 39)] == 1 and cells_needed[(8, 37)] == 1:
                    result[1][3][0] = 1
            elif i == 2:
                cells_needed = {(x, y): 1 for x in range(9, 20) for y in range(41, 60)}
                for j in occupied_clusters[i]:
                    for k in range(len(shortest_paths[j])):
                        cells_needed[shortest_paths[j][k][0]] = 0
                if cells_needed[(9, 54)] == 1 and cells_needed[(9, 55)] == 1:
                    result[2][0][9] = 1
                if cells_needed[(12, 41)] == 1 and cells_needed[(12, 52)] == 1 and cells_needed[(9, 55)] == 1:
                    result[2][1][9] = result[2][8][1] = 1
                if cells_needed[(13, 46)] == 1 and cells_needed[(13, 51)] == 1 and \
                    cells_needed[(12, 52)] == 1 and cells_needed[(9, 55)] == 1:
                    result[2][2][9] = result[2][8][2] = 1
                if cells_needed[(14, 46)] == 1 and cells_needed[(14, 51)] == 1 and cells_needed[(13, 51)] == 1 and \
                    cells_needed[(12, 52)] == 1 and cells_needed[(9, 55)] == 1:
                    result[2][3][9] = result[2][8][3] = 1
                if cells_needed[(15, 46)] == 1 and cells_needed[(15, 50)] == 1 and cells_needed[(14, 51)] == 1 and \
                    cells_needed[(13, 51)] == 1 and cells_needed[(12, 52)] == 1 and cells_needed[(9, 55)] == 1:
                    result[2][4][9] = result[2][8][4] = 1
                if cells_needed[(16, 46)] == 1 and cells_needed[(16, 49)] == 1 and cells_needed[(15, 50)] == 1 and \
                    cells_needed[(14, 51)] == 1 and cells_needed[(13, 51)] == 1 and cells_needed[(12, 52)] == 1 and \
                    cells_needed[(9, 55)] == 1:
                    result[2][5][9] = result[2][8][5] = 1
                if cells_needed[(17, 46)] == 1 and cells_needed[(17, 48)] == 1 and cells_needed[(16, 49)] == 1 and \
                    cells_needed[(15, 50)] == 1 and cells_needed[(14, 51)] == 1 and cells_needed[(13, 51)] == 1 and \
                    cells_needed[(12, 52)] == 1 and cells_needed[(9, 55)] == 1:
                    result[2][6][9] = result[2][8][6] = 1
                if cells_needed[(18, 46)] == 1 and cells_needed[(17, 48)] == 1 and \
                    cells_needed[(16, 49)] == 1 and cells_needed[(15, 50)] == 1 and cells_needed[(14, 51)] == 1 and \
                    cells_needed[(13, 51)] == 1 and cells_needed[(12, 52)] == 1 and cells_needed[(9, 55)] == 1:
                    result[2][7][9] = result[2][8][7] = 1
            else:
                cells_needed = {(x, y): 1 for x in range(7, 10) for y in range(68, 72)}
                for j in occupied_clusters[3]:
                    for k in range(len(shortest_paths[j])):
                        cells_needed[shortest_paths[j][k][0]] = 0
                if cells_needed[(9, 68)] == 1 and cells_needed[(8, 68)] == 1 and cells_needed[(7, 71)] == 1:
                    result[3][1][2] = 1
                if cells_needed[(9, 68)] == 1:
                    result[3][1][4] = 1
                if cells_needed[(7, 71)] == 1 and cells_needed[(8, 68)] == 1:
                    result[3][2][0]
                if cells_needed[(8, 69)] == 1 and cells_needed[(8, 68)] == 1:
                    result[3][3][0] = 1

        else:
            result[i] = default_matrices[i]
    return result
