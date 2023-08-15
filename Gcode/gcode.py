import numpy as np
import easygui
import trimesh

def divide_trajectory(origin, destination, size):
    # Compute the direction vector and distance between the two points
    direction = destination - origin
    distance = np.linalg.norm(direction)

    # Normalize the direction vector
    direction_normalized = direction / distance

    # Compute the number of chunks needed to divide the trajectory into 0.5 distance chunks
    num_chunks = int(distance // 0.5)

    # Compute the positions of the points along the trajectory
    positions = []
    for i in range(num_chunks):
        distance_to_chunk = (i + 1) * 0.5
        position = origin + distance_to_chunk * direction_normalized
        positions.append(position)

    return positions

def gcode_non_planar_modif(gcode_fid=None, comp_mesh_id=None):
    if gcode_fid is None:
        gcode_fid = easygui.fileopenbox('Please select gcode file', 'Please select gcode file')
    if comp_mesh_id is None:
        comp_mesh_id = easygui.fileopenbox('Please select compensated mesh', 'Please select compensated mesh')
    comp_mesh = trimesh.load(comp_mesh_id)

    Nearestpoint = trimesh.proximity.ProximityQuery(comp_mesh)

    with open(gcode_fid, 'r') as f:
        gcode = f.readlines()

    skin = False
    prev_point = None
    z_val = 0
    for line in gcode:
        if line[0] == ';':
            if 'SKIN' in line:
                skin = True
        else:
            if 'z' in line:
                z_val = float(line.split('Z')[1].split(' ')[0])
                prev_point = None
            if skin:
                x_val = float(line.split('X')[1].split(' ')[0])-105
                y_val = float(line.split('Y')[1].split(' ')[0])-105
                if prev_point is None:
                    prev_point = np.array(((x_val, y_val, z_val))).reshape((1,3))
                else:
                    diff = np.array(((x_val, y_val, z_val))).reshape((1,3)) - prev_point
                    if diff > 0.1:
                        div = divide_trajectory(prev_point, np.array(((x_val, y_val, z_val))).reshape((1,3)), 0.1)

                # point, _, _ = Nearestpoint.on_surface(np.array((x_val, y_val, z_val)).reshape((1,3)))
                # print(point)


if __name__ == '__main__':
    gcode_non_planar_modif()