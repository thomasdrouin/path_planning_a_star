import cv2
import numpy as np
import math
from queue import *
import copy
from vision.drawer import *

CELL_SIDE_LENGTH = 5

MAX_OBSTACLE_GRADIENT_VALUE = 255

MAX_CONTOUR_GRADIENT_VALUE = 255

OBSTACLE_GRADIENT_RADIUS = 100

TABLE_CONTOUR_GRADIENT_RADIUS = 60

CELL_DISPLACEMENT_VALUE = CELL_SIDE_LENGTH

MAX_GRADIENT_VALUE_TO_PASS = 100

NUMBER_OF_CONTOUR_POINTS_DRAWED_BY_CONTOUR_POINT = 1

OBSTACLE_DANGEROUS_DISTANCE = 140

NUMBER_OF_PATH_POSITIONS_BY_CHECKPOINT = 10


class Cell:
    def __init__(self, x_table_position, y_table_position, cell_id):
        self.id = cell_id
        self.x_table_position = x_table_position
        self.y_table_position = y_table_position
        self.center_position = self.get_cell_center_position()
        self.distance_from_goal_position = None
        self.gradient_value = 0

    def get_cell_distance_from_position(self, position):
        return math.hypot(position[0] - self.center_position[0], position[1] - self.center_position[1])

    def get_cell_center_position(self):
        return int(self.x_table_position * CELL_SIDE_LENGTH + CELL_SIDE_LENGTH / 2), int(self.y_table_position * CELL_SIDE_LENGTH + CELL_SIDE_LENGTH / 2)

    def set_cell_distance_from_goal_position(self, goal_position):
        self.distance_from_goal_position = math.hypot(abs(goal_position[0] - self.center_position[0]), abs(goal_position[1] - self.center_position[1]))

    def __eq__(self, other):
        return self.center_position == other.center_position

    def __hash__(self):
        pos = ''
        pos += str(self.x_table_position)
        pos += ':'
        pos += str(self.y_table_position)
        return pos.__hash__()

    def __lt__(self, other):
        self_priority = self.distance_from_goal_position + self.gradient_value
        other_priority = other.distance_from_goal_position + other.gradient_value
        return self_priority < other_priority

class PathPlanner:
    def __init__(self, resolution, obstacle_positions, table_contour):
        self.resolution = resolution
        self.start_position = None
        self.goal_position = None
        self.table_contour_point_positions = table_contour
        self.obstacle_positions = obstacle_positions
        self.cell_map_by_id = None
        self.cell_id_table = None
        self.image = None

        self.set_cell_list_and_table()
        self.max_x_cell_number = self.cell_id_table.shape[0] - 1
        self.max_y_cell_number = self.cell_id_table.shape[1] - 1

        self.put_contour_gradient()
        self.put_obstacles_gradient()

    def set_cell_list_and_table(self):
        x_resolution, y_resolution = self.resolution
        x_number_of_cells = math.ceil(x_resolution/CELL_SIDE_LENGTH)
        y_number_of_cells = math.ceil(y_resolution/CELL_SIDE_LENGTH)

        cell_id_table = np.zeros([x_number_of_cells, y_number_of_cells])
        cell_map_by_id = {}
        cell_id = 0

        for cell_x_table_position in range(0, x_number_of_cells):
            for cell_y_table_position in range(0, y_number_of_cells):
                cell = Cell(cell_x_table_position, cell_y_table_position, cell_id)
                cell_id_table[cell_x_table_position][cell_y_table_position] = cell_id
                cell_map_by_id[cell_id] = cell
                cell_id += 1

        self.cell_map_by_id = cell_map_by_id
        self.cell_id_table = cell_id_table

    def set_all_cell_distance_from_goal_position(self):
        for id, cell in self.cell_map_by_id.items():
            cell.set_cell_distance_from_goal_position(self.goal_position)

    def put_contour_gradient(self):
        contour_cells = set()
        for point_position in self.table_contour_point_positions:
            contour_cell = self.get_cell_from_position(point_position)
            contour_cells.add(contour_cell)

        contour_cells = list(contour_cells)
        for contour_cell in contour_cells:
            if self.max_x_cell_number/10 < contour_cell.x_table_position < (9 / 10)* self.max_x_cell_number and not (contour_cell.x_table_position%NUMBER_OF_CONTOUR_POINTS_DRAWED_BY_CONTOUR_POINT == 0):
                contour_cells.remove(contour_cell)
            elif self.max_y_cell_number/10 < contour_cell.y_table_position < (9 / 10)* self.max_y_cell_number and not (contour_cell.y_table_position%NUMBER_OF_CONTOUR_POINTS_DRAWED_BY_CONTOUR_POINT == 0):
                contour_cells.remove(contour_cell)

        for contour_cell in contour_cells:
            radius_layer_range = math.floor(TABLE_CONTOUR_GRADIENT_RADIUS / CELL_SIDE_LENGTH)
            for cell_layer_range in range(1, radius_layer_range):
                layer_cells = self.get_layer_of_cells_around_cell_by_range(contour_cell.x_table_position, contour_cell.y_table_position, cell_layer_range)
                for layer_cell in layer_cells:
                    gradient_value = (1-cell_layer_range/radius_layer_range) * MAX_CONTOUR_GRADIENT_VALUE
                    if gradient_value > layer_cell.gradient_value:
                        layer_cell.gradient_value = gradient_value


    def put_obstacles_gradient(self):
        for obstacle_position in self.obstacle_positions:
            self.put_obstacle_gradient(obstacle_position)

    def put_obstacle_gradient(self, obstacle_position):
        obstacle_cell = self.get_cell_from_position(obstacle_position)

        cells_around_obstacle = []
        radius_layer_range = math.floor(OBSTACLE_GRADIENT_RADIUS/CELL_SIDE_LENGTH)
        for cell_layer_range in range(1, radius_layer_range):
            layer_cells = self.get_layer_of_cells_around_cell_by_range(obstacle_cell.x_table_position,
                                                                       obstacle_cell.y_table_position, cell_layer_range)
            cells_around_obstacle.extend(layer_cells)

        for cell_around_obstacle_cell in cells_around_obstacle:
            cell_distance_from_obstacle = cell_around_obstacle_cell.get_cell_distance_from_position(obstacle_position)
            if cell_distance_from_obstacle < OBSTACLE_GRADIENT_RADIUS:
                gradient = (OBSTACLE_GRADIENT_RADIUS-cell_distance_from_obstacle) / OBSTACLE_GRADIENT_RADIUS * MAX_CONTOUR_GRADIENT_VALUE
                if gradient > cell_around_obstacle_cell.gradient_value:
                    cell_around_obstacle_cell.gradient_value = gradient

    def delete_last_obstacle_cell_gradients(self):
        obstacle_cell = self.get_cell_from_position(self.obstacle_positions[2])

        cells_around_obstacle = []
        for cell_layer_range in range(1, OBSTACLE_GRADIENT_RADIUS):
            layer_cells = self.get_layer_of_cells_around_cell_by_range(obstacle_cell.x_table_position,
                                                                       obstacle_cell.y_table_position, cell_layer_range)
            cells_around_obstacle.extend(layer_cells)

        for cell_around_obstacle_cell in cells_around_obstacle:
            cell_distance_from_obstacle = cell_around_obstacle_cell.get_cell_distance_from_position(self.obstacle_positions[2])
            if cell_distance_from_obstacle < OBSTACLE_GRADIENT_RADIUS:
                gradient = (OBSTACLE_GRADIENT_RADIUS-cell_distance_from_obstacle) / OBSTACLE_GRADIENT_RADIUS * MAX_OBSTACLE_GRADIENT_VALUE
                cell_around_obstacle_cell.gradient_value -= gradient
        self.put_obstacle_gradient(self.obstacle_positions[0])
        self.put_obstacle_gradient(self.obstacle_positions[1])

    def get_image_from_gradient(self):
        image = np.zeros((self.resolution[1], self.resolution[0], 3), np.uint8)
        for id, cell in self.cell_map_by_id.items():
            cell_gradient = cell.gradient_value
            gradient_color = (cell_gradient, cell_gradient, cell_gradient)
            top_left_corner = (cell.x_table_position*CELL_SIDE_LENGTH, cell.y_table_position*CELL_SIDE_LENGTH)
            bottom_corner = (top_left_corner[0]+CELL_SIDE_LENGTH, top_left_corner[1]+CELL_SIDE_LENGTH)
            cv2.rectangle(image, top_left_corner, bottom_corner, gradient_color, -1)
        return image

    def get_adjacent_layer_of_cells(self, cell_id):
        cell = self.cell_map_by_id[cell_id]
        adjacent_cells = self.get_layer_of_cells_around_cell_by_range(cell.x_table_position, cell.y_table_position, 1)
        return adjacent_cells

    def get_adjacent_cells(self, cell_id):
        cell = self.cell_map_by_id[cell_id]
        cell_x_table_position, cell_y_table_position = cell.x_table_position, cell.y_table_position
        adjacent_safe_cells = []
        if cell_x_table_position != 0:
            left_cell_id = self.cell_id_table[cell_x_table_position-1][cell_y_table_position]
            left_cell = self.cell_map_by_id[left_cell_id]
            adjacent_safe_cells.append(left_cell)
        if cell_y_table_position != 0:
            top_cell_id = self.cell_id_table[cell_x_table_position][cell_y_table_position-1]
            top_cell = self.cell_map_by_id[top_cell_id]
            adjacent_safe_cells.append(top_cell)
        if cell_x_table_position != self.max_x_cell_number:
            right_cell_id = self.cell_id_table[cell_x_table_position+1][cell_y_table_position]
            right_cell = self.cell_map_by_id[right_cell_id]
            adjacent_safe_cells.append(right_cell)
        if cell_y_table_position != self.max_y_cell_number:
            bottom_cell_id = self.cell_id_table[cell_x_table_position][cell_y_table_position+1]
            bottom_cell = self.cell_map_by_id[bottom_cell_id]
            adjacent_safe_cells.append(bottom_cell)
        return adjacent_safe_cells

    def get_cell_from_position(self, position):
        x_position, y_position = position
        cell_x_table_position = math.floor(x_position/CELL_SIDE_LENGTH)
        cell_y_table_position = math.floor(y_position/CELL_SIDE_LENGTH)
        cell_id = self.cell_id_table[cell_x_table_position][cell_y_table_position]
        cell = self.cell_map_by_id[cell_id]
        return cell

    def get_layer_of_cells_around_cell_by_range(self, x_table_position, y_table_position, cell_range):
        cell_list = []
        cell_x_table_position, cell_y_table_position = x_table_position, y_table_position

        for i in range(-cell_range, cell_range + 1):
            for j in range(-cell_range, cell_range + 1):
                if i == -cell_range or i == cell_range or j == -cell_range or j == cell_range:
                    if 0 <= cell_x_table_position + i <= self.max_x_cell_number and 0 <= cell_y_table_position + j <= self.max_y_cell_number:
                        cell_id = self.cell_id_table[cell_x_table_position + i][cell_y_table_position + j]
                        cell = self.cell_map_by_id[cell_id]
                        cell_list.append(cell)
        return cell_list

    def show_path_window_from_path_with_tolerance(self, path):
            if self.image is None:
                self.image = self.get_image_from_gradient()
            image_name = "gradients"
            image = copy.deepcopy(self.image)
            if path:
                print("show path start point = " + str(path[0]))
                print("show path end point = " + str(path[len(path) - 1]))
                for position in path:
                    position_x, position_y = position[0]
                    if position == path[0]:
                        draw_point_on_image((position_x, position_y), image, "vert")
                    elif position == path[len(path)-1]:
                        draw_point_on_image((position_x, position_y), image, "rouge")
                    else:
                        draw_point_on_image((position_x, position_y), image, "blanc")

            cv2.namedWindow(image_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
            cv2.setMouseCallback(image_name, self.mouse_callback)
            cv2.resizeWindow(image_name, (800, 450))
            cv2.imshow(image_name, image)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyWindow(image_name)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            start_position = (x, y)
            path = self.find_new_path_from_start_and_goal_position(start_position, self.goal_position)
            self.show_path_window_from_path_with_tolerance(path)

        elif event == cv2.EVENT_RBUTTONUP:
            goal_position = (x, y)
            path = self.find_new_path_from_start_and_goal_position(self.start_position, goal_position)
            self.show_path_window_from_path_with_tolerance(path)

        elif event == cv2.EVENT_LBUTTONDBLCLK:
            print("obstacle = " + str(x) + ' , ' + str(y))
            self.delete_last_obstacle_cell_gradients()
            self.obstacle_positions.pop(2)
            self.obstacle_positions.insert(0, (x, y))
            self.put_obstacles_gradient()
            self.put_contour_gradient()
            self.image = self.get_image_from_gradient()

    def find_first_path(self, start_position, goal_position):
        self.start_position = start_position
        self.goal_position = goal_position
        self.set_all_cell_distance_from_goal_position()

        cell_path_value_cell_priority_queue = PriorityQueue()
        cell_center_path_positions = []
        previous_cell_map = {}
        start_cell = self.get_cell_from_position(self.start_position)
        goal_cell = self.get_cell_from_position(self.goal_position)
        cell_path_value_cell_priority_queue.put((9999, start_cell))

        previous_cell_map[start_cell] = None

        goal_found = False

        while not cell_path_value_cell_priority_queue.empty() and not goal_found:
            current_cell = cell_path_value_cell_priority_queue.get()[1]
            adjacent_cells = self.get_adjacent_layer_of_cells(current_cell.id)
            for adjacent_cell in adjacent_cells:
                if not goal_found and adjacent_cell.gradient_value < MAX_GRADIENT_VALUE_TO_PASS:
                    if previous_cell_map.get(adjacent_cell) is None:
                        cell_path_value_cell_priority_queue.put((adjacent_cell.distance_from_goal_position + adjacent_cell.gradient_value, adjacent_cell))
                        previous_cell_map[adjacent_cell] = (current_cell, adjacent_cell.distance_from_goal_position + adjacent_cell.gradient_value)
                    else:
                        if adjacent_cell.distance_from_goal_position + adjacent_cell.gradient_value < previous_cell_map[adjacent_cell][1]:
                            previous_cell_map[adjacent_cell] = (current_cell, adjacent_cell.distance_from_goal_position + adjacent_cell.gradient_value)
                    if adjacent_cell.distance_from_goal_position == goal_cell.distance_from_goal_position:
                        cell_center_path_positions.append(adjacent_cell.center_position)
                        previous_cell = previous_cell_map[adjacent_cell][0]
                        while previous_cell != start_cell:
                            cell_center_path_positions.append(previous_cell.center_position)
                            previous_cell = previous_cell_map[previous_cell][0]
                        cell_center_path_positions.append(previous_cell.center_position)
                        cell_center_path_positions.reverse()
                        goal_found = True
        if goal_found:
            return cell_center_path_positions
        else:
            return None

    def find_new_path_from_start_and_goal_position(self, start_position, goal_position):
        first_first_path = self.find_first_path(start_position, goal_position)
        if first_first_path == None:
            print("goal not found...")
            return []
        shortest_path = self.find_shortest_path(first_first_path)
        self.goal_position = goal_position
        path_with_tolerance = create_position_tolerance_tuples_from_real_path(shortest_path, self.obstacle_positions)
        return path_with_tolerance

    def find_shortest_path(self, path):
        for i in range(0, len(path), 5):
            start_position = path[0]
            goal_position = path[i]
            possible_shorter_path = self.find_first_path(start_position, goal_position)
            print("before start point = " + str(path[0]))
            print("before end point = " + str(path[len(path) - 1]))
            if possible_shorter_path is not None and get_path_total_length(possible_shorter_path) < get_path_total_length(path[0:i+1])-20:
                path = possible_shorter_path + path[i+1:len(path)]
                print("shorter start point = " + str(path[0]))
                print("shorter end point = " + str(path[len(path) - 1]))
                return self.find_shortest_path(path)
        return path


def get_path_total_length(path):
    total_distance = 0
    for i in range(0, len(path)-1):
        present_cell = path[i]
        next_cell = path[i+1]
        distance = math.hypot(next_cell[0] - present_cell[0], next_cell[1] - present_cell[1])
        total_distance += distance
    return total_distance

def create_position_tolerance_tuples_from_real_path(real_path, obstacle_positions):
    checkpoint_path = []
    goal_position = real_path[-1]
    goal_position_tolerance = 15
    goal_position_tuple = [goal_position, goal_position_tolerance]
    for i in range(0, len(real_path) - 1):
        if i % NUMBER_OF_PATH_POSITIONS_BY_CHECKPOINT == 0:
            checkpoint_path.append([real_path[i], 60])

    for position_and_tolerance_tuple in checkpoint_path:
        for obstacle_position in obstacle_positions:
            if math.hypot(obstacle_position[0] - position_and_tolerance_tuple[0][0], obstacle_position[1] - position_and_tolerance_tuple[0][1]) < OBSTACLE_DANGEROUS_DISTANCE:
                position_and_tolerance_tuple[1] = 20
                break
    checkpoint_path.append(goal_position_tuple)
    return checkpoint_path
