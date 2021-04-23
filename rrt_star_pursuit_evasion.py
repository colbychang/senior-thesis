"""

Path planning code with RRT* for pursuit-evasion
    See "Incremental Sampling-based Algorithms for a class of
    Pursuit-Evasion Games" by Sertec Karaman and Emilio Frazzoli
    for the algorithm explanation


author: Colby Chang, modified from Atsushi Sakai's RRT Star code

"""

import math
import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../RRT/")

try:
    from rrt import RRT
except ImportError:
    raise

show_animation = True


class RRTStarPE(RRT):
    """
    Class for RRT Star planning
    """

    class Node(RRT.Node):
        def __init__(self, x, y):
            super().__init__(x, y)
            self.cost = 0.0

    def __init__(self,
                 start,
                 goal,
                 pursuer_start,
                 obstacle_list,
                 rand_area,
                 expand_dis=30.0,
                 path_resolution=1.0,
                 goal_sample_rate=20,
                 max_iter=2000,
                 connect_circle_dist=50.0,
                 search_until_max_iter=False,
                 min_radius = 10.0):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        pursuer_start:Pursuer's Start Position(s) [[x1,y1], ..., [xn, yn]]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]

        """
        super().__init__(start, goal, obstacle_list, rand_area, expand_dis,
                         path_resolution, goal_sample_rate, max_iter)
        self.evader_node_list = [self.start]
        self.pursuer_start = pursuer_start
        self.pursuer_node_list = [self.Node(x,y) for x, y in pursuer_start] # this might not work
        self.connect_circle_dist = connect_circle_dist
        self.goal_node = self.Node(goal[0], goal[1])
        self.search_until_max_iter = search_until_max_iter
        self.min_radius = min_radius

    def planning(self, animation=True):
        """
        rrt star path planning

        animation: flag for animation on or off .
        """
        # one iteration of expanding evader/pursuer's trees
        for i in range(self.max_iter):
            # print("Iter:", i, ", number of evader nodes:", len(self.evader_node_list))

            # generate new evader node
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.evader_node_list, rnd)
            near_node = self.evader_node_list[nearest_ind]
            new_evader_node = self.steer(near_node, rnd, self.expand_dis)
            new_evader_node.cost = near_node.cost + \
                            math.hypot(new_evader_node.x-near_node.x,
                                       new_evader_node.y-near_node.y)

            # add new node after checking validity/rewiring graph
            if self.check_collision(new_evader_node, self.obstacle_list):
                if self.validate_evader_node(new_evader_node):
                    near_inds = self.find_near_nodes(new_evader_node, self.evader_node_list)
                    node_with_updated_parent = self.choose_parent(
                        new_evader_node, near_inds, self.evader_node_list)
                    if node_with_updated_parent:
                        self.rewire(node_with_updated_parent, near_inds, self.evader_node_list)
                        self.evader_node_list.append(node_with_updated_parent)
                    else:
                        self.evader_node_list.append(new_evader_node)

            # generate new pursuer node
            rnd = self.get_random_node(pursuer=True)
            nearest_ind = self.get_nearest_node_index(self.pursuer_node_list, rnd)
            near_node = self.pursuer_node_list[nearest_ind]
            new_pursuer_node = self.steer(near_node, rnd, self.expand_dis)
            new_pursuer_node.cost = near_node.cost + \
                            math.hypot(new_pursuer_node.x - near_node.x,
                                       new_pursuer_node.y - near_node.y)

            if self.check_collision(new_pursuer_node, self.obstacle_list):
                # near_inds = self.find_near_nodes(new_pursuer_node, self.pursuer_node_list)
                # node_with_updated_parent = self.choose_parent(
                #     new_pursuer_node, near_inds, self.pursuer_node_list)
                # if node_with_updated_parent:
                #     self.rewire(node_with_updated_parent, near_inds, self.pursuer_node_list)
                #     self.pursuer_node_list.append(node_with_updated_parent)
                # else:
                #     self.pursuer_node_list.append(new_pursuer_node)
                self.pursuer_node_list.append(new_pursuer_node)
                self.curate_evader_nodes(self.pursuer_node_list[-1])

            if animation:
                self.draw_graph(rnd)

            if ((not self.search_until_max_iter)
                    and new_evader_node):  # if reaches goal
                last_index = self.search_best_goal_node()
                if last_index is not None:
                    return self.generate_final_course(last_index)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index is not None:
            return self.generate_final_course(last_index)

        return None

    def choose_parent(self, new_node, near_inds, node_list):
        """
        Computes the cheapest point to new_node contained in the list
        near_inds and set such a node as the parent of new_node.

        Arguments:
            new_node, Node
                randomly generated node with a path from its neared point
                There are not coalitions between this node and th tree.
            near_inds: list
                Indices of indices of the nodes what are near to new_node

        Returns:
            Node, a copy of new_node
        """
        if not near_inds:
            return None

        # search nearest cost in near_inds
        costs = []
        for i in near_inds:
            near_node = node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(t_node, self.obstacle_list):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.evader_node_list[min_ind], new_node)
        new_node.cost = min_cost

        return new_node

    def search_best_goal_node(self):
        """
        Checks if there exists an explored node within self.expand_dis
        to the goal node. If there are multiple explored nodes that fit
        this criterion, return the index of the node with the lowest cost
        to the goal
        """
        dist_to_goal_list = [
            self.calc_dist_to_goal(n.x, n.y) for n in self.evader_node_list
        ]
        goal_inds = [
            dist_to_goal_list.index(i) for i in dist_to_goal_list
            if i <= self.expand_dis
        ]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.evader_node_list[goal_ind], self.goal_node)
            if self.check_collision(t_node, self.obstacle_list):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        min_cost = min([self.evader_node_list[i].cost for i in safe_goal_inds])
        for i in safe_goal_inds:
            if self.evader_node_list[i].cost == min_cost:
                return i

        return None

    def curate_evader_nodes(self, pursuer_node):
        """
        For a given pursuer node, find the close evader nodes and remove any that
        have larger costs (times)

        Arguments:
            pursuer_node: Node
                The pursuer's node that was just inserted
        """
        near_inds = self.find_near_nodes(pursuer_node, self.evader_node_list)
        n = 0
        parent_nodes = []
        for ind in near_inds:
            evader_node = self.evader_node_list[ind]
            if evader_node.cost >= pursuer_node.cost:
                parent_nodes.append(evader_node)
                n += 1
        for parent_node in parent_nodes:
            self.remove_evader_children(parent_node)

    def remove_evader_children(self, parent_node):
        """
        Removes all of the children for a given evader node

        Arguments:
            parent_node: Node
                The node whose children should be removed
        """
        nodes_to_remove = []
        for i, node in enumerate(self.evader_node_list):
            if node.parent == parent_node:
                nodes_to_remove.append(node)
        # needs to be in a separate loop so to not mess with list traversal
        for node in nodes_to_remove:
            self.evader_node_list.remove(node)
            self.remove_evader_children(node)

    def validate_evader_node(self, evader_node):
        """
        Determine whether the new evader node is reachable by any pursuer node
        within the time needed for the evader to reach the new node

        Arguments:
            evader_node: Node
                The evader's node to be whose reachability is being checked
        """
        for node in self.pursuer_node_list:
            if node.cost <= evader_node.cost:
                if math.hypot(node.x - evader_node.x, node.y - evader_node.y) <= self.min_radius:
                    return False
        return True

    def find_near_nodes(self, new_node, node_list):
        """
        Defines a ball centered on new_node and returns indices of all other
        nodes that are inside this ball
        Arguments:
            new_node: Node
                new randomly generated node, without collisions between
                its nearest node
        Returns:
            list
                List with the indices of the nodes inside the ball of
                radius r
        """
        nnode = len(node_list) + 1
        r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
        # if expand_dist exists, search vertices in a range no more than
        # expand_dist
        if hasattr(self, 'expand_dis'):
            r = min(r, self.expand_dis)
        dist_list = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2
                     for node in node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r**2]
        return near_inds

    def rewire(self, new_node, near_inds, node_list):
        """
        For each node in near_inds, this will check if it is cheaper to
        arrive to them from new_node.
        In such a case, this will re-assign the parent of the nodes in
        near_inds to new_node.
        Arguments:
            new_node, Node
                Node randomly added which can be joined to the tree

            near_inds, list of uints
                A list of indices of the self.new_node which contains
                nodes within a circle of a given radius.

        Remark: parent is designated in choose_parent.
        """
        for i in near_inds:
            near_node = node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            no_collision = self.check_collision(edge_node, self.obstacle_list)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                near_node.x = edge_node.x
                near_node.y = edge_node.y
                near_node.cost = edge_node.cost
                near_node.path_x = edge_node.path_x
                near_node.path_y = edge_node.path_y
                near_node.parent = edge_node.parent
                self.propagate_cost_to_leaves(new_node, node_list)

    def calc_new_cost(self, from_node, to_node):
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node, node_list):
        for node in node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node, node_list)

    def draw_graph(self, rnd=None, path=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.evader_node_list:
            plt.plot(node.x, node.y, "b", marker='.', markersize=1)
            if node.parent:
                plt.plot([node.path_x[0], node.path_x[-1]],
                         [node.path_y[0], node.path_y[-1]],
                         color="black", linewidth=0.5)
        # for node in self.pursuer_node_list:
        #     if node.parent:
        #         plt.plot(node.path_x, node.path_y, "-r")
        for x, y in self.pursuer_start:
            plt.plot(x, y, 'bx')

        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)

        if path is not None:
            plt.plot([x for (x, y) in path], [y for (x, y) in path], 'r-')

        plt.plot(self.start.x, self.start.y, "go")
        plt.plot(self.end.x, self.end.y, "ro")
        # plt.axis("equal")
        # plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])
        # plt.grid(True)
        plt.pause(0.005)

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.evader_node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

def main():
    print("Start " + __file__)

    # ====Search Path with RRT====
    obstacle_list = [
        (5, 5, 1),
        (3, 6, 2),
        (3, 8, 2),
        (3, 10, 2),
        (7, 5, 2),
        (9, 5, 2),
        (8, 10, 1),
        (6, 12, 1),
    ]  # [x,y,size(radius)]

    obstacle_list = [
        (0, 6, 1),
        (7, 5.5, 2),
        (5, 0, 1),
    ]

    # Set Initial parameters
    rrt_star = RRTStarPE(
        start=[3, 3],
        goal=[12, 12],
        pursuer_start=[(16,-1),(2,15),(-1,-1)],
        rand_area=[-2, 18],
        obstacle_list=obstacle_list,
        expand_dis=1,
        search_until_max_iter=True,
        min_radius=0.5,
        max_iter=5000)
    path = rrt_star.planning(animation=False)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        # Draw final path
        if show_animation:
            rrt_star.draw_graph(path=path)
            # plt.plot([x for (x, y) in path], [y for (x, y) in path], 'r--')
            # plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
