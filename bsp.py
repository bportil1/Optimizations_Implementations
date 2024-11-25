import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

class BSP:
    #def __init__(self):

    def bisect(self, segments, line):
        segment_start = segments[..., 0, :]
        segment_end = segments[..., 1, :]

        v0 = segment_end - segment_start
        v1 = line[1] - line[0]

        numerator = np.cross((line[0] - segment_start), v1)
        denominator = np.cross(v0, v1)

        parallel = np.isclose(denominator, 0)
        not_parallel = np.logical_not(parallel)

        intersection = numerator / (denominator + parallel)

        ahead = np.logical_or(numerator > 0, np.logical_and(np.isclose(numerator, 0), denominator < 0))
        behind = np.logical_or(numerator < 0, np.logical_and(np.isclose(numerator, 0), denominator > 0))

        colinear = np.logical_and(parallel, np.isclose(numerator, 0))

        bisected = np.logical_and(not_parallel, np.logical_and(intersection > 0, intersection < 1))
        intersection_points = segment_start + intersection[..., np.newaxis] * v0

        l_segments = np.stack((segments[..., 0, :], intersection_points), axis=1)

        r_segments = np.stack((intersection_points, segments[..., 1, :]), axis=1)

        mask = numerator[..., np.newaxis, np.newaxis] > 0

        bisected_ahead = np.where(mask, l_segments, r_segments)[bisected]
        bisected_behind = np.where (np.logical_not(mask), l_segments, r_segments)[bisected]

        ahead_mask = np.logical_and(ahead, np.logical_not(bisected))
        behind_mask = np.logical_and(behind, np.logical_not(bisected))

        if bisected_ahead.size != 0:
            if np.any(ahead_mask):
                all_ahead = np.concatenate((segments[ahead_mask], bisected_ahead))
            else:
                all_ahead = bisected_ahead
        else:
            all_ahead = bisected_ahead

        if bisected_behind.size != 0:
            if np.any(behind_mask):
                all_behind = np.concatenate((segments[behind_mask], bisected_behind))
            else:
                all_behind = bisected_behind
        else:
            all_behind = segments[behind_mask]

        all_colinear = segments[colinear]

        return all_ahead, all_behind, all_colinear

    def build_tree(self, segments, starting_segment=None):
        
        def bsp_helper(segments, division_line):
            ahead, behind, colinear = self.bisect(segments, division_line)
            node_id = id(division_line)
            graph.add_node(node_id, line=division_line, colinear_segments=colinear)
            if behind.size != 0:
                node_behind = bsp_helper(behind, behind[0])
                graph.add_edge(node_id, node_behind, position=-1)
            if ahead.size != 0:
                node_ahead = bsp_helper(ahead, ahead[0])
                graph.add_edge(node_id, node_ahead, position=1)
            return node_id

        graph = nx.DiGraph()
        if starting_segment is None:
            starting_segment = segments[0]

        bsp_helper(segments, starting_segment)
        return nx.relabel.convert_node_labels_to_integers(graph)


    def draw_segments(self, tree, axis=None, *args, **kwargs):
        if axis is None:
            axis = plt.gca()
        all_segments = np.concatenate([value for value in dict(nx.get_node_attributes(tree, 'colinear_segments')).values()])
        for segment in all_segments:
            axis.plot(*(segment.T), *args, **kwargs)

    def segments_caller(self):
        segments = np.loadtxt('points.csv')
        segments = segments.reshape(int(segments.shape[0] / 4), 2, 2)

        tree = self.build_tree(segments)
        fig = plt.figure()
        axis = plt.subplot(2,1,1)
        axis.grid()
        for segment in segments:
            axis.plot(*(segment.T), "o-", color='k', linewidth=3, markersize=12)
        
        ax2 = plt.subplot(2,1,2)
        for _, segments in tree.nodes.data('colinear_segments'):
            for segment in segments:
                ax2.plot(*(segment.T), "o-", linewidth=3, markersize=12)

        ax2.grid()

        ax2.set_xlim(axis.get_xlim())
        ax2.set_ylim(axis.get_ylim())

        plt.show()

bsp = BSP()

bsp.segments_caller()
