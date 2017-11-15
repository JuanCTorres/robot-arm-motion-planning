# from src.cs1lib import start_graphics, draw_line
from time import sleep
from typing import List, Sized

import cs1lib
import random
from numpy import cos, sin, pi, sqrt
from shapely.geometry.polygon import Polygon, LineString
from collections import deque
from itertools import chain


class RobotArmMotion:
    l = 80  # length of robot arm

    def __init__(self, start, goal, k=5, step_count=20, obstacles=None):
        assert len(start) == len(goal)
        self.num_arms = len(start)
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.graph: dict = None
        self.k = k
        self.obstacles: List[Polygon] = obstacles
        self.step_count = step_count

        print('Start: %s' % start)
        print('Goal: %s' % goal)

    def angles_to_x_y(self, angles) -> list:
        """
        Translates a configuration (in angles) to a list of endpoints (x, y) for each robot arm.
        :param x: baseline x
        :param y: baseline y
        :param theta: angle
        :return: robot arm length x, robot arm length y
        >>> r = RobotArmMotion(1); r.angles_to_x_y([pi, pi, pi])
        """
        res = []
        for i in range(len(angles)):
            prev_angles = [angles[j] for j in range(i + 1)]
            prev_x, prev_y = res[i - 1] if i > 0 else (0, 0)
            x_res = prev_x + (self.l * cos(sum(prev_angles)))
            y_res = prev_y + (self.l * sin(sum(prev_angles)))
            res.append((x_res, y_res))
        return res

    # def angle_to_x_y(self, x, y, theta):
    #     """
    #     :param x: baseline x
    #     :param y: baseline y
    #     :param theta: angle
    #     :return: robot arm length x, robot arm length y
    #     >>> r = RobotArmMotion(1); r.angle_to_x_y(0, 0, 2 * pi)
    #
    #     """
    #     x_res = self.l * cos(theta)
    #     y_res = self.l * sin(theta)
    #     return x_res, y_res

    @DeprecationWarning
    def draw_config(self, angles: list):
        endpoints = self.angles_to_x_y(angles)
        i = 0
        while i < len(endpoints):
            x, y = (0, 0)
            cs1lib.draw_line(x, y, endpoints[i][0], endpoints[i][1])
            x, y = endpoints[i]
            i += 1

    def solve(self) -> list:
        """
        Finds a path for the robot
        """
        self.graph = self.generate_initial_configurations()
        self.populate_graph()
        res = self.bfs(self.start, self.goal)
        if res:
            return self.backtrack(res, self.goal)
        else:
            return []  # no solution

    def backtrack(self, dict, goal):
        """
        Takes a dictionary of nodes to parents created by bfs
        and finds a path from start to goal
        """
        path = [goal]
        curr = goal
        while curr is not None:
            path.append(curr)
            curr = dict[curr]
        return path[::-1]

    def bfs(self, start, goal) -> dict:
        """
        Breadth-first search.

        :param start: starting config of the search
        :param goal: goal config for the search
        :return: dictionary of configurations to parents used to backtrack and find a path
        """
        queue = deque([start])
        visited = set()
        parent_dict = {start: None}
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                if node == goal:
                    return parent_dict
                for n in self.get_valid_neighbors(node):
                    if n not in parent_dict:  # if neighbor doesn't have a parent already, node is its parent
                        parent_dict[n] = node
                        queue.append(n)
        return {}  # not found

    def get_valid_neighbors(self, node) -> list:
        """
        Returns the valid neighbors for a node among its k-closest ones.

        In this function, valid refers to the neighbors for which there is a transition
        from `node` to `neighbor` that does not collide with the obstacles in `self.obstacles`

        :param node: The node to get the valid neighbors for
        :return: List of valid neighbors
        """
        all_neighbor_list = self.graph[node]
        valid_neighbor_list = [neighbor for neighbor in all_neighbor_list if self.is_valid_transition(node, neighbor)]
        return valid_neighbor_list

    def is_valid_transition(self, config1: List, config2: List):
        """
        Is the transition between `config1` and `config2` valid?

        For a definition of valid, see `get_valid_neighbors`

        :param config1: Starting configuration for a robot arm (theta1, theta2, ...)
        :param config2: Target configuration for a robot arm (theta1, theta2, ...)
        :return: Is the transition between config1 and config2 valid?
        """
        assert len(config1) == len(config2)
        res = []
        for arm_index in range(len(config1)):
            # Define a range of angles to create a sampling on:
            # The sampling will have `step_count` steps between `range_start` and `range_stop`
            # `range_start` is the smallest of the two; range_stop is whichever `range_start` is not.
            range_start = min(config1[arm_index], config2[arm_index])
            range_stop = config2[arm_index] if (range_start == config1[arm_index]) else config1[arm_index]
            assert range_start != range_stop

            step_size = (range_stop - range_start) / self.step_count
            arm_current_intermediate_configs = [range_start + j * step_size for j in range(self.step_count)]
            res.append(arm_current_intermediate_configs)

        all_configs = [config1] + list(zip(*res)) + [config2]  # [theta, intermediate configs, theta']
        endpoint_list_all_configs = [self.angles_to_x_y(config_intermediate) for config_intermediate in all_configs]
        # endpoint_list_all_configs = [[[0] * 2] + endpoint_xy for endpoint_xy in endpoint_list_all_configs]
        endpoint_list_all_configs = [
            self.get_endpoints_with_start(endpoint_xy) for endpoint_xy in
            endpoint_list_all_configs
        ]
        for endpoint_list in endpoint_list_all_configs:
            # A line can be defined by 4 points: (x1, y1, x2, y2). These are the endpoints
            # of the previous arm and the endpoints of the current arm
            line_coords = [endpoint_list[i: i + 2] for i in range(len(endpoint_list) - 1)]
            line_objs = [LineString(coord) for coord in line_coords]
            if any([l.intersects(p) for l in line_objs for p in self.obstacles]):
                return False
        return True

    def get_endpoints_with_start(self, endpoint_list: list) -> list:
        """
        Transforms endpoints for each of the robot arms into endpoints for each robot arm AND a starting location (0, 0)
        :param endpoint_list: (x, y) endpoints for each arm of the robot
        """
        return [[0] * 2] + list(endpoint_list)

    def get_line_coordinates(self, endpoints_with_start: list) -> list:
        return [endpoints_with_start[i: i + 2] for i in range(len(endpoints_with_start) - 1)]

    def generate_initial_configurations(self) -> dict:
        d = dict.fromkeys(  # since it's a graph, a dictionary works best
            [
                tuple((random.uniform(0, 2 * pi) for _ in range(self.num_arms)))  # random configuration
                for __ in range(max(min(self.num_arms ** 7, 2000), 200))  # a large number of configurations
            ]
            + [self.start, self.goal]  # don't forget start and goal; they have to be in the graph
        )
        return d

    def populate_graph(self) -> None:
        """
        Populates the initial graph using k-closest neighbors
        """
        for config in self.graph:
            knn = self.get_knn(config, self.graph.keys(), self.k)
            self.graph[config] = knn

    def get_knn(self, p, neighbors, k):
        """
        Get k closest neighbors using angular distance

        :param p: point
        :param neighbors: neighbors
        :param k: how many of the closest neighbors to get
        :return: k-closest neighbors
        """
        dist = sorted([(self.angular_distance(p, neigh), neigh) for neigh in neighbors if p != neigh])
        # dist = sorted([(self.angular_distance(p, n), n) for n in neighbors if n != p])
        return [tup for d, tup in dist[: k]]

    @staticmethod
    def angular_distance(theta1: tuple, theta2: tuple) -> float:
        """
        :param theta1:
        :param theta2:
        :return:
        >>> a = (1, 2, 3); b = (4, 5, 6); print(RobotArmMotion.angular_distance(a, b))
        5.19615242271
        >>> a = (0, 0); b = (0, 0); print(RobotArmMotion.angular_distance(a, b))
        0.0
        """
        diffs = [RobotArmMotion.get_dist(a, b) for a, b in zip(theta1, theta2)]
        assert all([0 <= d <= pi for d in diffs])

        euclidean_dist = sqrt(
            sum(
                # [abs((t1 - t2) % pi_times_2) ** 2 for t1, t2 in zip(theta1, theta2)]
                [d ** 2 for d in diffs]
            )
        )
        return euclidean_dist

    @staticmethod
    def get_dist(a, b) -> float:
        pi_times_2 = 2 * pi

        dist = abs(a - b) % pi_times_2
        dist = min(dist, pi_times_2 - dist)
        assert 0 <= dist <= pi
        return dist


if __name__ == '__main__':
    # seed = 5
    # random.seed(seed)
    width = 700
    height = 500
    canvas_mid_x = width // 2
    canvas_mid_y = height // 2

    robot_arm_count = 3
    start_state = [random.uniform(0, 2 * pi) for _ in range(robot_arm_count)]
    goal_state = [random.uniform(0, 2 * pi) for _ in range(robot_arm_count)]
    # start_state = [1.393024861411571, 6.089091906185002, 4.719785764918847, 0.5262075278847457]
    # goal_state = [5.7269810625040165, 4.207226403531826, 0.07372503284423317, 2.783575613814917]

    obstacles1 = [
        Polygon([[50, 50], [50, 100], [100, 100], [100, 50]]),
        Polygon([[-80, -80], [-100, -100], [-80, -100], [-100, -80]])
    ]
    obstacles2 = [
        Polygon([[-100, -100], [-100, -80], [-80, -80], [-80, -100]]),
        Polygon([[100, 100], [100, 80], [80, 80], [80, 100]]),
        Polygon([[100, -100], [80, -100], [80, -80], [100, -80]]),
        Polygon([[-100, 80], [-100, 100], [-80, 80], [-100, 80]])
    ]

    # set up the system
    r = RobotArmMotion(
        start=start_state,
        goal=goal_state,
        k=20,
        step_count=75,
        obstacles=obstacles2
    )

    # solve it
    angle_config_result = r.solve()
    print('Result:')
    for i in range(len(angle_config_result)):
        cfg = angle_config_result[i]
        print(i, cfg)

    arm_endpoints = []
    for new_config in angle_config_result:
        arm_endpoints.append(r.angles_to_x_y(new_config))
    my_endpoints_with_start = [r.get_endpoints_with_start(config) for config in arm_endpoints]
    lines = [r.get_line_coordinates(endpoints_with_start) for endpoints_with_start in my_endpoints_with_start]

    print('Lines:')
    for i in range(len(lines)):
        l = lines[i]
        print(i, l)

    colors = [[random.uniform(0.2, 1) for i in range(3)] for _ in range(len(angle_config_result))]
    j = 0


    def on_click(mx, my):
        print("Mouse click! " + str(mx) + " " + str(my))


    def on_move(mx, my):
        pass


    def on_release(mx, my):
        print("Mouse up! " + str(mx) + " " + str(my))


    def on_keydown(key):
        print("Pressed " + key)


    def on_keyup(key):
        print("Released " + key)


    def get_x(my_x):
        return my_x + canvas_mid_x


    def get_y(my_y):
        # y is flipped because the canvas is flipped
        return -my_y + canvas_mid_y


    def draw():
        if not lines:
            print('No solution found')
            exit(0)

        cs1lib.set_clear_color(.2, .2, .2)
        cs1lib.clear()
        global j

        # for robot arms: enable stroke
        cs1lib.enable_stroke()

        robot_arms = lines[j]
        line_color = colors[j]
        cs1lib.set_stroke_color(*line_color)
        for arm in robot_arms:
            x, y, final_x, final_y = list(chain.from_iterable(arm))
            cs1lib.draw_line(get_x(x), get_y(y), get_x(final_x), get_y(final_y))

        # for obstacles: no stroke, enable fill
        cs1lib.set_fill_color(1, 1, 1)
        cs1lib.disable_stroke()

        for obs in r.obstacles:
            obs_x, obs_y, obs_xx, obs_yy = obs.bounds
            width = obs_xx - obs_x
            height = obs_yy - obs_y
            print(obs.bounds)
            print(get_x(obs_x), get_y(obs_y), get_x(obs_xx), get_y(obs_yy))
            cs1lib.draw_rectangle(get_x(obs_x), get_y(obs_y), width, height)

        # config = angle_config_result[j]
        cs1lib.enable_stroke()
        cs1lib.draw_text("%d" % j, 20, 20)
        j += 1
        if j == len(angle_config_result) - 1:  # goal state: show some message
            cs1lib.draw_text("GOAL", 20, 20)
        elif j >= len(angle_config_result):  # wrap around to get back to 0
            j %= len(angle_config_result)
            cs1lib.clear()
        sleep(1)


    cs1lib.start_graphics(draw, width=width, height=height, mouse_press=on_click, mouse_release=on_release,
                          mouse_move=on_move,
                          key_press=on_keydown)
