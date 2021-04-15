"""

Path planning Sample Code with Randomized Rapidly-Exploring Random Trees (RRT)

author: AtsushiSakai(@Atsushi_twi)

"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

show_animation = True


class RRT3D:
    """
    Class for RRT planning
    """

    class Node:
        """
        RRT Node
        """

        def __init__(self, x, y, z):
            self.x = x
            self.y = y
            self.z = z
            self.path_x = []
            self.path_y = []
            self.path_z = []
            self.parent = None

    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=3.0,
                 path_resolution=0.5,
                 goal_sample_rate=5,
                 max_iter=500):
        """
        Setting Parameter

        start:Start Position [x,y,z]
        goal:Goal Position [x,y,z]
        obstacleList:obstacle Positions [[x,y,z,size],...]
        randArea:Random Sampling Area [min,max]

        """
        self.start = self.Node(start[0], start[1], start[2])
        self.end = self.Node(goal[0], goal[1], goal[2])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []

    def planning(self, animation=True):
        """
        rrt path planning

        animation: flag for animation on or off
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            print("rnd node coords: [{}, {}, {}]".format(rnd_node.x, rnd_node.y, rnd_node.z))
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node)
                print("        node was appended at coords: [{}, {}, {}]".format(new_node.x, new_node.y, new_node.z))

            if animation:
                self.draw_graph(rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1].x,
                                      self.node_list[-1].y,
                                      self.node_list[-1].z) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end,
                                        self.expand_dis)
                if self.check_collision(final_node, self.obstacle_list):
                    print("iterations: {}".format(i))
                    return self.generate_final_course(len(self.node_list) - 1)

        return None  # cannot find path

    def steer(self, from_node, to_node, extend_length=float("inf")):

        new_node = self.Node(from_node.x, from_node.y, from_node.z)
        d, theta, phi = self.calc_distance_and_angle(new_node, to_node)
        print("d, theta, phi are: {}, {}, {}".format(d, theta, phi))

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]
        new_node.path_z = [new_node.z]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            # new_node.x += self.path_resolution * math.cos(theta)
            # new_node.y += self.path_resolution * math.sin(theta)

            new_node.x += self.path_resolution * math.sin(phi) * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(phi) * math.sin(theta)
            new_node.z += self.path_resolution * math.cos(phi)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)
            new_node.path_z.append(new_node.z)

        d, _, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.path_z.append(to_node.z)
            new_node.x = to_node.x
            new_node.y = to_node.y
            new_node.z = to_node.z

        new_node.parent = from_node

        return new_node

    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y, self.end.z]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y, node.z])
            node = node.parent
        path.append([node.x, node.y, node.z])

        return path

    def calc_dist_to_goal(self, x, y, z):
        dx = x - self.end.x
        dy = y - self.end.y
        dz = z - self.end.z
        return math.hypot(dx, dy, dz)

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand),
                random.uniform(self.min_rand, self.max_rand))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y, self.end.z)
        return rnd

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis("equal")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        plt.pause(0.01)

    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 +
                 (node.z - rnd_node.z)**2 for node in node_list]
        minind = dlist.index(min(dlist))

        return minind

    @staticmethod
    def check_collision(node, obstacleList):

        if node is None:
            return False

        for (ox, oy, oz, size) in obstacleList:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            dz_list = [oz - z for z in node.path_z]
            d_list = [dx * dx + dy * dy + dz * dz for (dx, dy, dz) in zip(dx_list, dy_list, dz_list)]

            if min(d_list) <= size**2:
                return False  # collision

        return True  # safe

    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        """
        Returns cylindrical coordinates from from_node to to_node
        """
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        dz = to_node.z - from_node.z
        d = math.hypot(dx, dy, dz)
        if d == 0:
            print('d is 0')
            return 0, 0, 0
        theta = math.atan2(dy, dx)
        phi = math.acos(dz/d)

        return d, theta, phi


def main(gx=6.0, gy=10.0, gz=3.0):
    print("start " + __file__)

    # ====Search Path with RRT====
    # obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
    #                 (9, 5, 2), (8, 10, 1)]  # [x, y, radius]
    obstacleList = [(5, 5, 3, 1), (3, 6, 2, 2), (3, 8, 5, 2), (3, 10, 4, 2), (7, 5, 7, 2),
                    (9, 5, 3, 2), (8, 10, 2, 1)]  # [x, y, z, radius]
    # obstacleList = [(5, 5, 0, 1), (3, 6, 0, 2), (3, 8, 0, 2), (3, 10, 0, 2), (7, 5, 0, 2),
    #                 (9, 5, 0, 2), (8, 10, 0, 1)]  # [x, y, z, radius]
    # Set Initial parameters
    rrt = RRT3D(
        start=[0, 0, 0],
        goal=[gx, gy, gz],
        rand_area=[-2, 15],
        obstacle_list=obstacleList)
    path = rrt.planning(animation=False)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        # Draw final path
        if show_animation:
            print("path length is: {}".format(len(path)))
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            ax.plot(path[0][0], path[0][1], path[0][2], "xr")
            ax.plot(path[-1][0], path[-1][1], path[-1][2], "xr")

            for i in range(len(path)-1):
                x1, y1, z1 = path[i]
                x2, y2, z2 = path[i+1]
                ax.plot([x1, x2], [y1, y2], [z1, z2], color='g')
                print("z is [{}, {}]".format(z1, z2))

            u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]

            for ox, oy, oz, r in obstacleList:
                x = r * np.cos(u) * np.sin(v) + ox
                y = r * np.sin(u) * np.sin(v) + oy
                z = r * np.cos(v) + oz
                ax.plot_surface(x, y, z, color='b')

            plt.show()


            # rrt.draw_graph()
            # plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            # plt.grid(True)
            # plt.pause(0.01)  # Need for Mac
            # plt.show()


if __name__ == '__main__':
    main()
