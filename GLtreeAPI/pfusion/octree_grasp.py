import numpy as np
from queue import Queue
import time
from .interval_tree import RedBlackTree, Node, BLACK, RED, NIL
import random
import math

from .draw_utils import get_line_geometry
import open3d as o3d
from scipy.spatial.transform import Rotation as R

def rotation_matrix_to_axis_angle(matrix):
    epsilon = 1e-6
    angle = np.arccos(  np.clip(np.trace(matrix) - 1, -1, 1) / 2.0)

    return angle


def normalize(quat):
    norm = np.linalg.norm(quat)
    return quat / norm

def dot_product(q1, q2):
    return np.sum(np.array(q1) * np.array(q2))

def slerp(q1, q2, t, ensure_shortest_path=True):

    
    # 首先确保两个四元数都是归一化的
    q1 = normalize(q1)
    q2 = normalize(q2)
    
    # 计算两个四元数之间的点积
    cos_omega = np.dot(q1, q2)
    
    # 如果确保选择最短路径且点积为负，则反转其中一个四元数
    if ensure_shortest_path and cos_omega < 0.0:
        q2 = -q2
        cos_omega = -cos_omega
    
    # 如果两个四元数几乎平行，直接使用线性插值
    if cos_omega > 0.95:
        return normalize((1.0 - t) * np.array(q1) + t * np.array(q2))
    
    omega = np.arccos(cos_omega)
    so = np.sin(omega)
    
    weighted_quat = (np.sin((1.0 - t) * omega) / so) * np.array(q1) + (np.sin(t * omega) / so) * np.array(q2)

    
    return normalize(weighted_quat)



def remove_duplicates(set_of_tuples):

    seen = set()
    result = set()
    for t in set_of_tuples:
        sorted_t = tuple(sorted(t, key=lambda x: x))
        if sorted_t not in seen:
            seen.add(sorted_t)
            result.add(t)
    return result

class Grasp6d:
    def __init__(self, grasp_pos, grasp_ori, score):

        self.grasp_pos = grasp_pos
        self.grasp_ori = grasp_ori
        
        self.score = score
        self.fused_count = 0
        

        self.branch_array = [None, None, None, None, None, None, None, None]
        self.branch_distance = np.full((8),15)
        self.frame_id = 0


class GL_tree_grasp:

    def __init__(self, args):
        # self.opt = opt

        self.args= args

        self.x_rb_tree = RedBlackTree(self.args["interval_size"])
        self.y_rb_tree = RedBlackTree(self.args["interval_size"])
        self.z_rb_tree = RedBlackTree(self.args["interval_size"])

        self.scene_node = set()
        
        self.connected_num = 0
        self.score_total = 0
        self.fused_total = 0


        # self.observation_window = set()
        # self.observation_window_size = self.args["observation_window_size"]

        



    def reset_gltree(self):
        del self.x_rb_tree
        del self.y_rb_tree
        del self.z_rb_tree
        del self.scene_node

        self.x_rb_tree = RedBlackTree(self.args["interval_size"])
        self.y_rb_tree = RedBlackTree(self.args["interval_size"])
        self.z_rb_tree = RedBlackTree(self.args["interval_size"])
        self.scene_node = set()
        # self.observation_window = set()
        
        self.connected_num = 0
        self.score_total = 0
        self.fused_total = 0

    def init_grasp_node(self, grasp_coor):
        self.x_tree_node_list = []
        self.y_tree_node_list = []
        self.z_tree_node_list = []

        for p in range(grasp_coor.shape[0]):
            x_temp_node = self.x_rb_tree.add(grasp_coor[p,0])
            y_temp_node = self.y_rb_tree.add(grasp_coor[p,1])
            z_temp_node = self.z_rb_tree.add(grasp_coor[p,2])

            self.x_tree_node_list.append(x_temp_node)
            self.y_tree_node_list.append(y_temp_node)
            self.z_tree_node_list.append(z_temp_node)


    def add_grasps(self, grasp_group_array, frame_index):

        # activate_3d = True
        
        per_image_node_set = set()

        # per_image_node_num = 0
        # per_image_fused_num = 0
        # per_image_score_incremental = 0


        for p in range(grasp_group_array.shape[0]):
            
            x_set_union = self.x_tree_node_list[p].set_list
            y_set_union = self.y_tree_node_list[p].set_list
            z_set_union = self.z_tree_node_list[p].set_list
            set_intersection = x_set_union[0] & y_set_union[0] & z_set_union[0]
            temp_branch = [None, None, None, None, None, None, None, None]
            temp_branch_distance = np.full((8), self.args["max_octree_threshold"])
            is_find_nearest = False
            branch_record = set()
            list_intersection=list(set_intersection)
            random.shuffle(list_intersection)

            pos = grasp_group_array[p, 13:16]
            ori = grasp_group_array[p, 4:4+9].reshape(3,3)
            
            quat_ori = R.from_matrix(ori).as_quat()
            
            score = grasp_group_array[p, 0]
            

            for grasp_iter in list_intersection:

                if rotation_matrix_to_axis_angle(grasp_iter.grasp_ori @ ori.T) > self.args["min_ori_threshold"]:
                    continue

                # distance check
                distance = np.linalg.norm(grasp_iter.grasp_pos - pos)
                if distance < self.args["min_octree_threshold"]:
                    is_find_nearest = True
                    if frame_index!=grasp_iter.frame_id:
                        grasp_iter.frame_id=frame_index
                        grasp_iter.fused_count+=1
                        
                        weight = score / (grasp_iter.score + score)
                        
                        grasp_iter.grasp_pos = (grasp_iter.grasp_pos * (1-weight)  + pos * weight)                         
                        
                        quat_ori = slerp(quat_ori, R.from_matrix(grasp_iter.grasp_ori).as_quat(), 1 - weight)
                        grasp_iter.grasp_ori = R.from_quat(quat_ori).as_matrix()
                        
                        
                        
                        grasp_iter.score = grasp_iter.score + score
                        
                        
                        self.fused_total += 1

                        
                    per_image_node_set.add(grasp_iter)
                    break

                x = int(grasp_iter.grasp_pos[0] >= pos[0])
                y = int(grasp_iter.grasp_pos[1] >= pos[1])
                z = int(grasp_iter.grasp_pos[2] >= pos[2])
                branch_num= x * 4 + y * 2 + z

                if distance < grasp_iter.branch_distance[7-branch_num]:
                    branch_record.add((grasp_iter, 7 - branch_num, distance))

                    if distance < temp_branch_distance[branch_num]:
                        temp_branch[branch_num] = grasp_iter
                        temp_branch_distance[branch_num] = distance


            if not is_find_nearest:
                new_grasp = Grasp6d(pos, ori, score)
                new_grasp.frame_id = frame_index

                for grasp_branch in branch_record:
                    grasp_branch[0].branch_array[grasp_branch[1]] = new_grasp
                    grasp_branch[0].branch_distance[grasp_branch[1]] = grasp_branch[2]
                    self.connected_num += 1
                    self.fused_total += 1

                new_grasp.branch_array = temp_branch
                new_grasp.branch_distance = temp_branch_distance
                per_image_node_set.add(new_grasp)
                self.score_total += new_grasp.score
                
                for x_set in x_set_union:
                    x_set.add(new_grasp)
                for y_set in y_set_union:
                    y_set.add(new_grasp)
                for z_set in z_set_union:
                    z_set.add(new_grasp)

        # self.observation_window = self.observation_window.union(per_image_node_set)

        self.scene_node = self.scene_node.union(per_image_node_set)
        return per_image_node_set

    def all_grasp(self):
        return self.scene_node


    def get_node_numbder(self):
        return len(self.scene_node)
    
    def get_edge_number(self):
        return self.connected_num

    def get_fused_count(self):
        return self.fused_total
    
    def get_totla_score(self):
        return self.score_total


    def all_grasps_array(self):
        observation_grasp = np.zeros((len(self.scene_node), 3+9+1)) 

        for i, node in enumerate(self.scene_node):
            observation_grasp[i,:3] = node.grasp_pos
            observation_grasp[i,3:3+9] = node.grasp_ori.reshape(-1)
            observation_grasp[i,12] = node.score
        
        return observation_grasp


    def write_grasp_geometry(self, file_path):

        node_list = list(self.scene_node)

        connection_list = []
        # points_list = []

        for node in node_list:
            # print("======= node.branch_array ===========",node.branch_array)
            for branch_node in node.branch_array:
                if branch_node is not None:
                    connection_list.append(tuple(np.r_[node.grasp_pos, branch_node.grasp_pos]))
                    # print("-------------", tuple(np.r_[node.grasp_pos, branch_node.grasp_pos]))
                    # connection_list.append(tuple(node.grasp_pos[0], node.grasp_pos[1], node.grasp_pos[2], branch_node.grasp_pos[0], branch_node.grasp_pos[1], branch_node.grasp_pos[2]))
                            

        # print("========= connection_list ========", connection_list)

        if len(connection_list) == 0:
            return None

        # print("========= connection_list ========", connection_list)

        seen = remove_duplicates(connection_list)

        # construct points and connections
        points = []
        connections = []
        for i, node in enumerate(seen):
            # print("========= node ========", node)
            points.append((node[0], node[1], node[2]))
            points.append((node[3], node[4], node[5]))
            connections.append((2*i, 2*i+1))    

        tracking_mesh = get_line_geometry(np.asarray(points), np.asarray(connections))

        line_mesh_geo = tracking_mesh.cylinder_segments
        scene_mesh = line_mesh_geo[0]

        for j in range(1,len(line_mesh_geo)):
            scene_mesh = scene_mesh + line_mesh_geo[j]

        o3d.io.write_triangle_mesh(file_path, scene_mesh)





    # def sample_points(self):
    #     if len(self.observation_window) > self.observation_window_size:
    #         remove_node_list = random.sample(self.observation_window, len(self.observation_window) - self.observation_window_size)
    #         for node in remove_node_list:
    #             self.observation_window.remove(node)

    #     observation_grasp = np.zeros((4096, 3+9+1)) 
    #     for i, node in enumerate(self.observation_window):
    #         observation_grasp[i,:3] = node.grasp_pos
    #         observation_grasp[i,3:3+9] = node.grasp_ori.reshape(-1)
    #         observation_grasp[i,12] = node.score

        # return observation_grasp



    # def write_nodes(self, filename):

    #     lens = len(self.all_points())

    #     point_count=lens
    #     ply_file = open(filename, 'w')
    #     ply_file.write("ply\n")
    #     ply_file.write("format ascii 1.0\n")
    #     ply_file.write("element vertex " + str(point_count) + "\n")
    #     ply_file.write("property float x\n")
    #     ply_file.write("property float y\n")
    #     ply_file.write("property float z\n")
    #     ply_file.write("property uchar red\n")
    #     ply_file.write("property uchar green\n")
    #     ply_file.write("property uchar blue\n")

    #     ply_file.write("end_header\n")
    #     for node in self.all_points():
    #         ply_file.write(str(node.point_coor[0]) + " " +
    #                     str(node.point_coor[1]) + " " +
    #                     str(node.point_coor[2]) + " "
    #                     str(rgb_points[i, 0])+ " "+
    #                     str(rgb_points[i, 1])+ " "+
    #                     str(rgb_points[i, 2]))
    #         ply_file.write("\n")
    #     ply_file.close()
    #     print("save result to "+filename)
            

    # def write_points(self, filename, points, rgb_points):

    #     lens = points.shape[0]

    #     point_count=lens
    #     ply_file = open(filename, 'w')
    #     ply_file.write("ply\n")
    #     ply_file.write("format ascii 1.0\n")
    #     ply_file.write("element vertex " + str(point_count) + "\n")
    #     ply_file.write("property float x\n")
    #     ply_file.write("property float y\n")
    #     ply_file.write("property float z\n")
    #     ply_file.write("property uchar red\n")
    #     ply_file.write("property uchar green\n")
    #     ply_file.write("property uchar blue\n")

    #     ply_file.write("end_header\n")
    #     for i in range(lens):
    #         ply_file.write(str(points[i, 0]) + " " +
    #                     str(points[i, 1]) + " " +
    #                     str(points[i, 2]) + " "+ 
    #                     str(rgb_points[i, 0])+ " "+
    #                     str(rgb_points[i, 1])+ " "+
    #                     str(rgb_points[i, 2]))

    #         ply_file.write("\n")
    #     ply_file.close()
    #     print("save result to "+filename)
            
        

            
