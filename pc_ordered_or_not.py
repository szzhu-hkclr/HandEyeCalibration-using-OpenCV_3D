import numpy as np
from plyfile import PlyData

def read_ply_header(file_path):
    with open(file_path, 'rb') as f:
        line = f.readline().decode('ascii').strip()
        header = []
        while line != 'end_header':
            header.append(line)
            line = f.readline().decode('ascii').strip()
    return header

def check_point_order(ply_data):
    vertices = np.vstack([ply_data['vertex']['x'], ply_data['vertex']['y'], ply_data['vertex']['z']]).T
    if 'width' in ply_data['vertex'].properties and 'height' in ply_data['vertex'].properties:
        width = ply_data['vertex']['width']
        height = ply_data['vertex']['height']
        if vertices.shape[0] == width * height:
            return True
    return False

def main(file_path):
    header = read_ply_header(file_path)
    print("PLY Header:")
    for line in header:
        print(line)
    
    ply_data = PlyData.read(file_path)
    if check_point_order(ply_data):
        print("The point cloud is ordered.")
    else:
        print("The point cloud is unordered.")

if __name__ == "__main__":
    file_path = "pc01.ply"
    main(file_path)