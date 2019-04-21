from path_planning.path_planner import *

TABLE_NUMBER = 5
CAMERA_INDEX = 0


def path_tester_offline():
    table_contour_point_positions = load_table_contours()
    obstacle_positions = [(500, 200), (500, 300), (500, 400)]
    robot_position = (87, 107)
    goal_position = (697, 312)
    center_position = (482, 102)

    path_planner = PathPlanner((800, 450), obstacle_positions, table_contour_point_positions)

    # path = path_planner.find_new_path_from_start_and_goal_position(robot_position, goal_position)
    path_with_tolerance = path_planner.find_new_path_from_start_and_goal_position(robot_position, goal_position)
    path_planner.show_path_window_from_path_with_tolerance(path_with_tolerance)


def load_table_contours():
    file_name = "table_contour.npz"

    restored_contour = np.load(file_name)

    return restored_contour["table_contour"]


path_tester_offline()
