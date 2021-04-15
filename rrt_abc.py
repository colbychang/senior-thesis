"""

Path planning with RRTABC

author: Colby Chang, modified from Atsushi Sakai's RRT Star code

"""

import imageio
import math
import os
import random
import sys
import time
import xlsxwriter

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../RRT/")

try:
    from rrt import RRT
except ImportError:
    raise

show_animation = True
plot_evader = True
slow_factor = 0.7
create_gif = False


class RRTABC:
    """
    Class for RRT Star planning
    """

    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = [x]
            self.path_y = [y]
            self.parent = None
            self.cost = float("inf")
            self.lmc = float("inf")
            self.N0_in = []
            self.N0_out = []
            self.Nr_in = []
            self.Nr_out = []
            self.children = set()
            self.line = []
            self.dist = float("inf")
            self.reachable = True

    class Edge:
        def __init__(self, start_node, end_node):
            self.start_node = start_node
            self.end_node = end_node
            self.path_x = []
            self.path_y = []

    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=1.0,
                 path_resolution=0.2,
                 goal_sample_rate=2,
                 max_iter=2000,
                 connect_circle_dist=1.6,
                 epsilon=0.001,
                 search_until_max_iter=False,
                 hidden_obs_list=[],
                 fake_obs_inds=[],
                 moving_obs=None):
        """
        Setting parameters

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacle_list:obstacle Positions [(x,y,size),...]
        rand_area:Random Sampling Area [min,max]
        """
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.end.lmc = 0
        self.end.cost = 0
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.obs_plot_objs = []
        self.node_list = []
        self.connect_circle_dist = connect_circle_dist
        self.epsilon = epsilon
        self.search_until_max_iter = search_until_max_iter
        self.Q = []
        self.orphan_node_list = []
        self.dist_dict = {}
        self.edge_list = []
        self.bot_node = self.start
        self.goal_chosen = False
        self.hidden_obs_list = hidden_obs_list
        self.fake_obs_list = [obstacle_list[i] for i in fake_obs_inds]

        self.moving_obs = moving_obs
        self.mov_obs_circ = None

        self.mouse_x = 0
        self.mouse_y = 0
        self.dist_traveled = 0

    def planning(self, animation=True):
        """
        rrt star path planning

        animation: flag for animation on or off .
        """
        if animation:
            plt.ion()
            plt.xlim([-2.5, 15.5])
            plt.ylim([-2.5, 15.5])
            point = plt.plot(self.bot_node.x, self.bot_node.y, 'go')
            plt.plot(self.end.x, self.end.y, 'ro')

            plt.show()
            for x, y, r in self.obstacle_list:
                self.obs_plot_objs.append(self.plot_circle(x, y, r))
            if self.moving_obs is not None:
                self.mov_obs_circ = self.plot_circle(self.moving_obs[0],
                                                     self.moving_obs[1],
                                                     self.moving_obs[2])
                self.obstacle_list.append(self.moving_obs)

        goal_achieved = False
        self.node_list.append(self.end)

        filenames = []
        for i in range(self.max_iter):
            # shrinking ball radius
            # for now, not using shrinking ball radius
            # n = len(self.node_list)
            # self.connect_circle_dist = self.connect_circle_dist * math.log(n) / n

            # sampling node and determining its validity
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            near_node = self.node_list[nearest_ind]
            d, theta = self.calc_distance_and_angle(rnd, near_node)

            # replacing node to be within expand_dis
            min_dist = min(d, self.expand_dis * 0.99)
            try:
                rnd.x = near_node.x + (rnd.x - near_node.x) * min_dist / d
                rnd.y = near_node.y + (rnd.y - near_node.y) * min_dist / d
            except ZeroDivisionError:
                print("coords were ({}, {})".format(rnd.x, rnd.y))

            # if self.check_collision(rnd, self.obstacle_list):
            # extend the graph (attempt to add rnd to the graph)
            extended = self.extend(rnd)
            if animation and plot_evader:
                plt.plot(rnd.x, rnd.y, "b", marker='.', markersize=1)
            if extended:
                if animation and plot_evader:
                    # plotting the line from random point to parent
                    # plt.plot(rnd.x, rnd.y, "b", marker='.', markersize=3)
                    rnd.line = plt.plot([rnd.path_x[0], rnd.path_x[-1]],
                                        [rnd.path_y[0], rnd.path_y[-1]],
                                        "black", linewidth=0.5)

                # rewiring and reducing inconsistency
                self.rewire_neighbors(rnd)
                self.reduce_inconsistency()

                # plot goal line if goal reached
                if rnd.x == self.start.x and rnd.y == self.start.y:
                    starter_node = rnd
                    if animation:
                        goal_line = self.plot_from(starter_node)
                    goal_achieved = True
                    self.goal_chosen = True
                    prev_goal_cost = starter_node.cost
                    self.bot_node = starter_node

                # replot goal line if cost-to-goal has decreased
                if goal_achieved and starter_node.cost < prev_goal_cost:
                    if animation:
                        goal_line.pop(0).remove()
                        goal_line = self.plot_from(starter_node)
                    prev_goal_cost = starter_node.cost

            if create_gif and (i % 100 == 0 or i == self.max_iter-1):
                filename = "iter{}.png".format(i)
                plt.savefig(filename)
                filenames.append(filename)
        print("Iterations finished")

        if animation:
            plt.connect('motion_notify_event', self.mouse_move)
            plt.plot(self.end.x, self.end.y, 'ro')

            # plt objects to maintain bot location
            point[0].remove()
            point = plt.plot(self.bot_node.x, self.bot_node.y, 'go')
            circ = self.plot_circle(self.bot_node.x, self.bot_node.y, 1.5, color='-g')
            filename = 'moving_frame.png'
            plt.savefig(filename)
            filenames.append(filename)

            input("Press Enter to initiate robot move...")

        t = time.time()
        tot_time = 0
        i = 0
        tot_dist = 0
        pe_dists = []

        # moving the robot
        while self.bot_node != self.end:
            self.update_obstacles()

            px = self.moving_obs[0]
            py = self.moving_obs[1]
            pe_dists.append([tot_dist, math.hypot(px - self.bot_node.x, py - self.bot_node.y)])

            if animation:
                plt.plot([self.bot_node.path_x[0], self.bot_node.path_x[-1]],
                         [self.bot_node.path_y[0], self.bot_node.path_y[-1]],
                         color="black", linewidth=3)

            self.dist_traveled, _ = self.calc_distance_and_angle(self.bot_node, self.bot_node.parent)

            # TODO: DEBUG SITUATION WHERE BOT_NODE BECOMES NONE (because its parent is None?)
            self.bot_node = self.bot_node.parent
            if animation:
                # redrawing robot after one node of movement
                point[0].remove()
                point = plt.plot(self.bot_node.x, self.bot_node.y, 'go')
                circ[0].remove()
                circ = self.plot_circle(self.bot_node.x, self.bot_node.y, 1.5, color='-g')
                goal_line.pop(0).remove()
                goal_line = self.plot_from(self.bot_node)

            # gif creation
            if create_gif:
                filename = "moving_frame{}.png".format(i)
                filenames.append(filename)
                print(len(filenames))
                plt.savefig(filename)

            # timing each step
            new_t = time.time()
            diff = new_t - t
            if diff > 0.01:
                print("Time elapsed: {}".format(diff))
            tot_time += diff
            t = new_t
            i += 1
            if animation:
                val = 1 - diff
                if val > 0:
                    print("pausing for {}".format(val))
                    plt.pause(val)
                    t = time.time()

            tot_dist += self.dist_traveled
            print("Total distance: {} in frame {}".format(tot_dist, i-1))
            # input("Press Enter to continue...")

        if create_gif:
            writer = imageio.get_writer('rrtabc.mp4', fps=3)
            for filename in filenames:
                writer.append_data(imageio.imread(filename))
            writer.close()

        with xlsxwriter.Workbook('test.xlsx') as workbook:
            worksheet = workbook.add_worksheet()

            for row_num, data in enumerate(pe_dists):
                worksheet.write_row(row_num, 0, data)

        print("Total time elapsed: {}".format(tot_time))
        print("Average time elapsed: {}".format(tot_time / i))
        input("Press Enter to end...")

    def check_path(self):
        dist = 0
        start_cost = self.bot_node.cost
        node = self.bot_node
        while node.parent is not None:
            prev_node = node
            node = node.parent
            new_dist = self.dist_dict[(prev_node, node)]
            dist += new_dist
            dx = node.x - self.moving_obs[0]
            dy = node.y - self.moving_obs[1]
            d = math.hypot(dx, dy) * slow_factor

            # if moving obstacle can reach that node before bot can, need to change path
            if d < dist:
                # revert back to cost before that bad node and add bad node to orphans
                dist -= new_dist
                self.verify_orphan(node)
                low_cost = float('inf')
                best_node = None
                # check all the near out nodes
                for near_node in set(prev_node.N0_out) | set(prev_node.Nr_out):
                    if near_node in self.orphan_node_list or near_node.parent == prev_node:
                        continue
                    extra_dist = self.dist_dict[(prev_node, near_node)]

                    dx = near_node.x - self.moving_obs[0]
                    dy = near_node.y - self.moving_obs[1]
                    d = math.hypot(dx, dy) * slow_factor
                    if dist + extra_dist < d and near_node.cost < low_cost:
                        low_cost = near_node.cost
                        best_node = near_node
                if best_node is not None:
                    self.make_parent_of(prev_node, best_node)
                    dist += self.dist_dict[(prev_node, best_node)]
                    print("reformatted a path node")
                else:
                    print("it's donezo")

    def move_obstacle(self):
        """
        Moves an obstacle (pursuer) based on the mouse location
        """
        if self.moving_obs is None:
            return
        prev_obs = self.moving_obs

        # find new coords for the obstacle
        dx = self.mouse_x - prev_obs[0]
        dy = self.mouse_y - prev_obs[1]
        d = math.hypot(dx, dy)
        if d != 0:
            min_dist = min(d, self.dist_traveled * slow_factor)
            x = prev_obs[0] + (self.mouse_x - prev_obs[0]) * min_dist / d
            y = prev_obs[1] + (self.mouse_y - prev_obs[1]) * min_dist / d
        else:
            x = prev_obs[0]
            y = prev_obs[1]
        self.moving_obs = (x, y, prev_obs[2])
        self.add_new_obstacle(self.moving_obs)
        self.remove_obstacle(prev_obs, paint_over=False)

        # rewiring cascade based on reachability
        for node in self.node_list:
            was_reachable = node.reachable
            self.check_reachable(node)
            if not was_reachable and node.reachable:
                self.verify_queue(node)
            elif was_reachable and not node.reachable:
                if len(node.line) > 0:
                    node.line.pop(0).remove()
                # TODO: what else to do here?
                self.verify_orphan(node)

    def extend(self, new_node):
        """
        Attempts to insert a new node into the graph. Finds the best parent
        and populates its neighbor lists

        Arguments:
            new_node: Node
                The node to be (potentially) inserted

        Returns:
            extended: boolean
                Whether or not the node was successfully inserted
        """
        extended = False
        near_inds = self.find_near_nodes(new_node)
        near_nodes = [self.node_list[i] for i in near_inds]
        temp_edge_list = []

        # for a moving pursuer
        if self.moving_obs is not None:
            self.check_reachable(new_node)

        for near_node in near_nodes:
            # adding near node to new node's neighbor list and vice versa
            new_node.N0_out.append(near_node)
            new_node.N0_in.append(near_node)
            near_node.Nr_out.append(new_node)
            near_node.Nr_in.append(new_node)

            edge = self.steer(near_node, new_node)
            temp_edge_list.append(edge)
            d, _ = self.calc_distance_and_angle(new_node, near_node)
            if self.check_collision(edge, self.obstacle_list):
                # populating distance dictionary
                self.dist_dict[(new_node, near_node)] = d
                self.dist_dict[(near_node, new_node)] = d
                # if lowest lmc and inside connect circle dist, make new node's parent near node
                if d <= self.connect_circle_dist and new_node.lmc > d + near_node.lmc and new_node.reachable:
                    # TODO: SHOULD IT GET A LMC IF IT'S NOT REACHABLE?
                    # seems like maybe, just so it's put in the priority queue more accurately? But also maybe not?
                    new_node.parent = near_node
                    new_node.lmc = d + near_node.lmc
                    # this part seems clunky, but can't figure out a better way so
                    # this is what's going down for now. Simply replacing the paths
                    # because steer generates a path in a new node
                    new_node.path_x = edge.path_x
                    new_node.path_y = edge.path_y
            else:
                # populating distance dictionary
                self.dist_dict[(new_node, near_node)] = float("inf")
                self.dist_dict[(near_node, new_node)] = float("inf")
        # add the node to the graph
        if new_node.parent is not None:
            new_node.parent.children.add(new_node)
            extended = True

        self.node_list.append(new_node)
        self.edge_list += temp_edge_list # why not just add edges as they go?

        return extended

    def steer(self, from_node, to_node, extend_length=float("inf")):
        """
        Calculates a straight-line path between two nodes. Creates and
        returns an edge containing that path

        Arguments:
            from_node: Node
                The node that is being navigated from
            to_node: Node
                The node that is being navigated to
            extend_length: float
                The max length of an edge

        Returns:
            edge: Edge
                An edge generated from from_node to to_node with a planned path
        """
        edge = self.Edge(from_node, to_node)

        d, theta = self.calc_distance_and_angle(from_node, to_node)

        x = from_node.x
        y = from_node.y
        edge.path_x = [x]
        edge.path_y = [y]

        # I think this might not be necessary with the new code
        # Could get rid of it, but would have to change extend_length to d in following code
        if extend_length > d:
            extend_length = d

        # add points to the path of length self.path_resolution
        while extend_length > self.path_resolution:
            x += self.path_resolution * math.cos(theta)
            y += self.path_resolution * math.sin(theta)
            edge.path_x.append(x)
            edge.path_y.append(y)
            extend_length = math.hypot(to_node.x - x, to_node.y - y)

        if extend_length > 0:
            edge.path_x.append(to_node.x)
            edge.path_y.append(to_node.y)

        return edge

    def cull_neighbors(self, node):
        """
        Culls the running in/out-neighbors of a given node based on the decreasing r
        """
        for near_node in node.Nr_out:
            d = self.dist_dict[(node, near_node)]
            if self.connect_circle_dist < d and node.parent != near_node:
                node.Nr_out.remove(near_node)
                node.Nr_in.remove(near_node)

    def rewire_neighbors(self, node):
        """
        Rewires the in-neighbors of a given node to use that node as their parent
        (if this rewiring lowers their cost-to-goal)

        Arguments:
            node: Node
                The node whose in-neighbors are checked for rewiring
        """
        # need the NaN case for when the node has just been initialized since both values
        # start off as infinity
        # TODO: ADD MATH.ISNAN CHECKS FOR OTHER AREAS WHERE NODE COST AND LMC ARE CHECKED
        if node.cost - node.lmc > self.epsilon or math.isnan(node.cost - node.lmc):
            self.cull_neighbors(node)
            for near_node in set(node.N0_in) | set(node.Nr_in):
                if not near_node.reachable:
                    continue
                d = self.dist_dict[(near_node, node)]
                if near_node.lmc > d + node.lmc and node.parent != near_node:
                    if near_node.parent != node:
                        self.make_parent_of(near_node, node)
                    near_node.lmc = d + node.lmc
                    if near_node.cost - near_node.lmc > self.epsilon:
                        self.verify_queue(near_node)

    def reduce_inconsistency(self):
        """
        Manages rewiring cascade of nodes in the priority queue. Propagates cost-to-goal
        information and maintains E-consistency in the graph
        """
        while len(self.Q) > 0:  # TODO: and a bunch of other things, check alg 5 in paper
            node = self.Q.pop(0)
            if not node.reachable:
                continue
            if node.cost - node.lmc > self.epsilon:
                self.update_lmc(node)
                self.rewire_neighbors(node)
            node.cost = node.lmc

    def reduce_inconsistency_motion(self):
        """
        Manages rewiring cascade of nodes in the priority queue. Propagates cost-to-goal
        information and maintains E-consistency in the graph.

        Note: This method is used when the robot is in motion
        """
        # print("q len is {}".format(len(self.Q)))
        while len(self.Q) > 0:
            if not (self.get_key(self.Q[0]) < self.get_key(self.bot_node)
                    or self.bot_node.lmc != self.bot_node.cost
                    or self.bot_node.cost == float('inf')
                    or self.bot_node in self.Q):
                # print("returned, q len is still {}".format(len(self.Q)))
                return
            node = self.Q.pop(0)
            if not node.reachable:
                continue
            # if node == self.bot_node:
            #     print("node is bot node, q len is {}".format(len(self.Q)))
            if node.cost - node.lmc > self.epsilon:
                self.update_lmc(node)
                self.rewire_neighbors(node)
            node.cost = node.lmc

    def check_reachable(self, node, max_dist=5):
        """
        Identifies if a node is "reachable" based on L2-norm to the
        node from the robot and its pursuer

        Arguments:
            node: Node
                Node whose reachability is being checked
            max_dist: float
                Maximum distance from the pursuer's location at which
                the node can be considered not reachable
        """
        px = self.moving_obs[0]
        py = self.moving_obs[1]
        r = self.moving_obs[2]
        d = math.hypot(px - node.x, py - node.y)  # slow_factor
        bx = self.bot_node.x
        by = self.bot_node.y
        node.dist = math.hypot(bx - node.x, by - node.y)
        if d - r <= node.dist and d < max_dist:
            node.reachable = False
        else:
            node.reachable = True

    def update_obstacles(self, radius=1.5):
        """
        Updates the obstacles based on new sensor measurements from the
        robot's current location

        Arguments:
            radius: float
                Radius at which the robot can sense hidden obstacles
        """
        new_obs = False
        obs_removed = False
        if self.moving_obs is not None:
            # move pursuer set distance in direction of mouse
            self.move_obstacle()
            new_obs = True

        # example of removing an obstacle
        # if self.bot_node.x > 10 and not self.obs_removed:
        #     self.obs_removed = True
        #     obs = (7, 5, 2)
        #     self.remove_obstacle(obs)
        #     self.reduce_inconsistency_motion()
        x = self.bot_node.x
        y = self.bot_node.y
        obs_to_keep = []
        for (ox, oy, size) in self.fake_obs_list:
            d = math.hypot(ox - x, oy - y)
            if d <= radius + size:
                obs_removed = True
                self.remove_obstacle((ox, oy, size))
            else:
                obs_to_keep.append((ox, oy, size))
        if obs_removed:
            self.fake_obs_list = obs_to_keep
            self.reduce_inconsistency_motion()

        obs_to_keep = []
        for (ox, oy, size) in self.hidden_obs_list:
            d = math.hypot(ox - x, oy - y)
            if d <= radius + size:
                new_obs = True
                self.add_new_obstacle((ox, oy, size))
            else:
                obs_to_keep.append((ox, oy, size))
        if new_obs:
            self.hidden_obs_list = obs_to_keep
            self.propagate_descendants()
            self.verify_queue(self.bot_node)
            self.reduce_inconsistency_motion()

    def update_lmc(self, node):
        """
        Updates a node's lmc based on its out-neighbors

        Arguments:
            node: Node
                The node whose lmc is to be updated
        """
        self.cull_neighbors(node)
        for near_node in set(node.N0_out) | set(node.Nr_out):
            if near_node in self.orphan_node_list:
                continue
            d = self.dist_dict[(node, near_node)]
            if node.lmc > d + near_node.lmc and near_node.parent != node:
                if node.parent != near_node:
                    self.make_parent_of(node, near_node)
                node.lmc = d + near_node.lmc

    def propagate_descendants(self):
        """
        Performs a cost-to-goal increase cascade after an obstacle is added.
        Handles logistics behind removing nodes from the graph
        """
        # updating orphan_node_list to include all children of nodes in the list
        for node in self.orphan_node_list:
            self.orphan_node_list += list(node.children)
        for node in self.orphan_node_list:
            # TODO: NOT ENTIRELY SURE ABOUT THIS LOOP
            for near_node in set(node.N0_out) | set(node.Nr_out):
                if near_node not in self.orphan_node_list and near_node.reachable:  # TODO: NOT SURE ABOUT THE REACHABLE ART
                    near_node.cost = float("inf")
                    self.verify_queue(near_node)

            # could this be more efficient? Seems like if we did this before the first
            # for loop in propagate descendants then it would be less time consuming
            if node.parent not in self.orphan_node_list and node.parent is not None:
                node.parent.cost = float("inf")
                self.verify_queue(node.parent)
        # why does this have to be a separate for loop from the previous one?
        for node in self.orphan_node_list:
            # delete the lines for the orphan nodes
            if len(node.line) > 0:
                node.line.pop(0).remove()
            node.cost = float("inf")
            node.lmc = float("inf")
            if node.parent is not None:
                # might get an error if the node isn't in the parent node's children?
                # should always be there, but maybe add a try-except loop just in case?
                node.parent.children.remove(node)
                node.parent = None
        self.orphan_node_list = []

    def make_parent_of(self, node, new_parent):
        """
        Assigns a new parent for a provided node and plans a new path for
        that node. Also replots the node's line on the graph

        Arguments:
            node: Node
                The node whose parent is being reassigned
            new_parent: Node
                The new parent node
        """
        # handling change of parents and children
        if node.parent is not None:
            node.parent.children.remove(node)
        node.parent = new_parent
        new_parent.children.add(node)

        # replanning the path
        temp_edge = self.steer(new_parent, node)
        # TODO: FIGURE OUT IF NEED TO CHECK IF IN COLLISION
        # I don't think it is necessary to check if in collision, since the d value
        # should be inf if that edge is in collision
        node.path_x = temp_edge.path_x
        node.path_y = temp_edge.path_y

        if show_animation and plot_evader:
            # replotting the line
            if len(node.line) > 0:
                node.line.pop(0).remove()
            node.line = plt.plot([node.path_x[0], node.path_x[-1]],
                                 [node.path_y[0], node.path_y[-1]],
                                 "black", linewidth=0.5)

    def verify_queue(self, node):
        """
        Adding node to priority queue (self.Q) based on the key
        (min(cost, lmc), lmc) such that (a,b) < (c,d) iff
        a < c or (a==c and b<d)

        Arguments:
            node: Node
                The node to be added to the priority queue
        """
        key = self.get_key(node)
        if node in self.Q:
            return
        for i, curr_node in enumerate(self.Q):
            curr_key = self.get_key(curr_node)
            # TODO: FIGURE OUT IF THIS SHOULD BE < OR <=
            # got this error once with <=:
            # File "/Users/colbychang/PythonRobotics/PathPlanning/RRTStar/rrt_x.py", line 215, in planning
            #     point = plt.plot(self.bot_node.x, self.bot_node.y, 'mo')
            # AttributeError: 'NoneType' object has no attribute 'x'
            if key <= curr_key:
                self.Q.insert(i, node)
                return
        self.Q.append(node)

    def verify_orphan(self, node):
        """
        Adds a node to the orphan list

        Arguments:
            node: Node
                The node to be added to the orphan list
        """
        if node in self.Q:
            self.Q.remove(node)
        self.orphan_node_list.append(node)

    def add_new_obstacle(self, obstacle):
        """
        Adds a new obstacle to the obstacle list and checks for any orphan nodes
        created by the obstacle's addition

        Arguments:
            obstacle: tuple
                The new obstacle to be added in the form (x_pos, y_pos, radius)
        """
        # print('adding obstacle')
        self.obstacle_list.append(obstacle)
        if show_animation:
            if obstacle == self.moving_obs:
                self.mov_obs_circ[0].remove()
                self.mov_obs_circ = self.plot_circle(obstacle[0], obstacle[1], obstacle[2])
            else:
                self.plot_circle(obstacle[0], obstacle[1], obstacle[2])
        for edge in self.edge_list:
            if not self.check_collision(edge, [obstacle]):
                node1 = edge.start_node
                node2 = edge.end_node
                # seems a bit redundant to do it twice, but unsure if doing it once is a better way
                self.dist_dict[(node1, node2)] = float("inf")
                self.dist_dict[(node2, node1)] = float("inf")
                if node1.parent == node2:
                    self.verify_orphan(node1)
                if node2.parent == node1:
                    self.verify_orphan(node2)
                # TODO: if v_bot is in pi(v,u) then pi_bot = None (see line 6 of alg 11)

    def remove_obstacle(self, removed_obs, paint_over=True):
        """
        Removes an obstacle from the obstacle list and reinserts previously
        intersecting nodes back into the graph

        Arguments:
            removed_obs: tuple
                The obstacle to be removed in the form (x_pos, y_pos, radius)
            paint_over: bool
                Whether or not to create a white circle over the old obstacle location
        """
        # print('removing obstacle')
        idx = self.obstacle_list.index(removed_obs)
        self.obstacle_list.remove(removed_obs)
        if show_animation and paint_over:
            plot_obj = self.obs_plot_objs.pop(idx)
            plot_obj[0].remove()
            # self.plot_circle(removed_obs[0], removed_obs[1], removed_obs[2], color='w')
        affected_nodes = set()
        for edge in self.edge_list:
            if not self.check_collision(edge, [removed_obs]):
                if self.check_collision(edge, self.obstacle_list):
                    node1 = edge.start_node
                    node2 = edge.end_node
                    d, _ = self.calc_distance_and_angle(node1, node2)
                    # seems a bit redundant to do it twice, but unsure if doing it once is a better way
                    self.dist_dict[(node1, node2)] = d
                    self.dist_dict[(node2, node1)] = d
                    affected_nodes.add(node1)
                    affected_nodes.add(node2)
        for node in affected_nodes:
            if not node.reachable:
                continue
            self.update_lmc(node)
            if node.lmc != node.cost:
                self.verify_queue(node)

    def find_near_nodes(self, new_node):
        """
        Defines a ball centered on new_node and returns all nodes that
        are inside this ball

        Arguments:
            new_node: Node
                New randomly generated node, without collisions between
                it and its nearest node
        Returns:
            list
                List with the indices of the nodes inside the ball of
                radius r
        """
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist  # * math.sqrt((math.log(nnode) / nnode))
        # if expand_dist exists, search vertices in a range no more than
        # expand_dist
        if hasattr(self, 'expand_dis'):
            r = min(r, self.expand_dis)

        # TODO: GO THROUGH ALL CODE AND MAKE IT CONSISTENT, EITHER MATH.HYPOT OR SQUARED EUCLIDEAN DIST
        dist_list = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2
                     for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r ** 2]
        if len(near_inds) == 0:
            print("No near nodes")
            print("Min dist is {}".format(math.sqrt(min(dist_list))))
        return near_inds

    def get_random_node(self):
        """
        Generates a random node in the reachable space. Has a set
        frequency of sampling the start point

        Returns:
            Node
                A node with a random location
        """
        if not self.goal_chosen and random.randint(0, 100) <= self.goal_sample_rate:
            # goal point sampling
            rnd = self.Node(self.start.x, self.start.y)
        else:
            # normal random sampling
            rnd = self.Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand))
        return rnd

    def mouse_move(self, event):
        """
        Updates the mouse location
        """
        self.mouse_x = event.xdata
        self.mouse_y = event.ydata
        if self.mouse_x is None or self.mouse_y is None:
            self.mouse_x = self.bot_node.x
            self.mouse_y = self.bot_node.y

    @staticmethod
    def plot_from(node):
        """
        Plots a red line starting from a given node

        Arguments:
            node: Node
                The node to start the plot from
        """
        x = []
        y = []
        while node is not None:
            x.append(node.x)
            y.append(node.y)
            node = node.parent
        line = plt.plot(x, y, 'r')
        return line

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        """
        Returns the index of the nearest node to a given node

        Arguments:
            node_list: list[Node]
                List of nodes to check
            rnd_node: Node
                Node to find the nearest node

        Returns:
            int
                Index of node_list for the nearest node
        """
        dlist = [(node.x - rnd_node.x) ** 2 + (node.y - rnd_node.y) ** 2
                 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def check_collision(obj, obstacle_list):
        """
        Check for any collisions in the path for object provided and
        the obstacles in the space

        Arguments:
            obj: Node or Edge
                The node or edge whose path to analyze
            obstacle_list: list[tuple]
                List of obstacles of the form (xpos, ypos, radius)

        Returns:
            bool
                True if no collision, False if collision
        """

        if obj is None:
            return False

        for (ox, oy, size) in obstacle_list:
            dx_list = [ox - x for x in obj.path_x]
            dy_list = [oy - y for y in obj.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            if min(d_list) <= size ** 2:
                return False  # collision

        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        """
        Find and return the distance and angle between two nodes

        Arguments:
            from_node: Node
                The node to be navigated from
            to_node: Node
                The node to be navigated to

        Returns:
            tuple(float, float)
                Tuple containing (distance, angle)
        """
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        """
        Plots a circle
        """
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        circ = plt.plot(xl, yl, color)
        return circ

    @staticmethod
    def get_key(node):
        """
        Returns the key for a node for the priority queue
        """
        key = (min(node.cost, node.lmc), node.cost)
        return key


def main():
    print("Start " + __file__)

    # ====Search Path with RRT====
    obstacle_list = [
        # (5, 5, 1),
        # (3, 6, 2),
        # (3, 8, 2),
        # (3, 10, 2),
        # (7, 5, 2),
        # (9, 5, 2),
        # (8, 10, 1),
        # (6, 12, 1),
        (3, 8, 2),
        (3, 11, 1),
        (10, 10, 2),
        (10, 12.5, 1),
        (2.5, 3, 1.5),
        (3, 6, 2),
        (11, 7.5, 1),
    ]  # [x,y,size(radius)]

    hidden_obs_list = [
        (5, 5, 1),
        # (3, 8, 2),
        # (3, 11, 1),
        (7, 5, 2),
        (9, 5, 2),
        (11, 5, 1.5),
        (13.5, 5, 1.5)
        # (8, 10, 1),
        # (6, 12, 1),
    ]  # [x,y,size(radius)]

    # Set Initial parameters
    rrt_x = RRTABC(
        start=[1, 0],
        goal=[13, 12],
        rand_area=[-2, 15],
        obstacle_list=[],  #obstacle_list,
        # hidden_obs_list=hidden_obs_list,
        moving_obs=(1, 6, 2),
        # fake_obs_inds=[0,3]
    )
    rrt_x.planning(animation=show_animation)

    num_neighs = []
    for node in rrt_x.node_list:
        num_neighs.append(len(set(node.N0_in) | set(node.Nr_in)))
    # print(max(num_neighs))
    b = [0, 5, 10, 15, 20, 25, 30]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(num_neighs, bins=b)
    plt.show()


if __name__ == '__main__':
    main()
