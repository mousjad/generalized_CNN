import numpy as np

def z_morph(gcode_fid, interp):


    with open(gcode_fid, 'r') as f:
        gcode = f.readlines()

    for line in gcode:
        if 'G1' in line or 'G0' in line:
            if 'X' in line:
                X = float(line.split('X')[1].split(' ')[0])
            if 'Y' in line:
                Y = float(line.split('Y')[1].split(' ')[0])
            if 'Z' in line:
                Z = float(line.split('Z')[1].split(' ')[0])


def interpolate(ref_mesh, dist):
    id = np.where(ref_mesh.vertex_normals[:,2] == 1)[0]

def main():



    return None

if __name__ == '__main__':
    main()