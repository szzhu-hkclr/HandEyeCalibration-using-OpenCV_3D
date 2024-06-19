import cv2
import numpy as np
import glob
import open3d as o3d
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import copy, csv

class MultiCameraRegistration(object):
    def __init__(self, rgb_intrinsic, relative_mat):
        self.project_idx_x = None
        self.project_idx_y = None
        self.rgb_intrinsic = rgb_intrinsic
        self.relative_mat = relative_mat
        self.eps = 1e-10

    def point_projection_vector(self, point):
        cam_fx = self.rgb_intrinsic[0, 0]
        cam_fy = self.rgb_intrinsic[1, 1]
        cam_cy = self.rgb_intrinsic[1, 2]
        cam_cx = self.rgb_intrinsic[0, 2]
        point_x = point[:, 0] / (point[:, 2] + self.eps)
        point_y = point[:, 1] / (point[:, 2] + self.eps)
        point_x = point_x * cam_fx + cam_cx
        point_y = point_y * cam_fy + cam_cy
        return point_x.astype(np.int32), point_y.astype(np.int32)

    def calculate_project_index(self, pc_array):
        pc_array = np.concatenate([pc_array, np.ones((pc_array.shape[0], 1))], axis=1)
        pc_rgb_array = np.linalg.inv(self.relative_mat).dot(pc_array.T).T[:, :3]
        self.project_idx_x, self.project_idx_y = self.point_projection_vector(pc_rgb_array)

    def get_rbgd_pointcloud(self, pc_array, rgb_img):
        self.calculate_project_index(pc_array)
        rgb_pc = rgb_img[np.clip(self.project_idx_y, 0, rgb_img.shape[0] - 1),
                         np.clip(self.project_idx_x, 0, rgb_img.shape[1] - 1), :]
        rgbd_pcd = o3d.geometry.PointCloud()
        rgbd_pcd.points = o3d.utility.Vector3dVector(pc_array)
        rgbd_pcd.colors = o3d.utility.Vector3dVector(rgb_pc[:, ::-1]/255.0)
        return rgbd_pcd

    def get_img_to_3d_mapping(self, pc_array, rgb_img):
        """
        map each (x,y) in rgb_img to real world x,y,z
        """
        self.calculate_project_index(pc_array)
        maxy, maxx = rgb_img.shape[:2]
        depth_img = np.zeros([maxy, maxx, 3], dtype=float)
        depth_img[:] = np.finfo(float).max
        for (point, rgbx, rgby) in zip(pc_array, self.project_idx_x, self.project_idx_y):
            if not ((0 <= rgbx < maxx) and (0 <= rgby < maxy)): continue
            if point[2] < depth_img[rgby, rgbx, 2]: depth_img[rgby, rgbx] = point
        return depth_img

    def imgXYs_to_3d_mapping(self, imgXYs, pc_array):
        """
        map (x,y) in rgb_img to real world x,y,z
        """
        self.calculate_project_index(pc_array)
        output = []
        for imgX, imgY in imgXYs:
            xy_diff = abs(imgX - self.project_idx_x) + abs(imgY - self.project_idx_y)
            output.append(pc_array[np.argmin(xy_diff)])
        return output

    def get_project_index(self):
        return self.project_idx_x, self.project_idx_y

def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]  # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    H = np.matmul(np.transpose(AA),BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.matmul(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2, :] *= -1
        R = np.matmul(Vt.T,U.T)

    t = -np.matmul(R, centroid_A) + centroid_B
    # err = B - np.matmul(A,R.T) - t.reshape([1, 3])
    return R, t

def undistort(frame, k, d):
    # 相机坐标系到像素坐标系的转换矩阵 k
    # 畸变系数 d
    h, w = frame.shape[:2]
    mapx, mapy = cv2.initUndistortRectifyMap(k, d, None, k, (w, h), 5)
    return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

# 找棋盘格角点
# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 150, 0.001) # 阈值
#棋盘格模板规格
w = 11   # 12 - 1
h = 8   # 9  - 1
# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
objp = np.zeros((w*h,3), np.float32)
objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
# print(objp)
# exit()
objp = objp*0.015  # mm
# print(objp)

# 储存棋盘格角点的世界坐标和图像坐标对
objpoints = [] # 在世界坐标系中的三维点
imgpoints = [] # 在图像平面的二维点

root_folder = './06-19_hand_eye' 
other_dir = root_folder + "/"

camera_other = np.array([[3579.0434570313, 0.0,             1246.4215087891], 
                         [0.0,             3578.8725585938, 1037.0089111328],
                         [0.0,             0.0,             1.0]]) 
dis_coe = np.array([-0.0672596246, 0.1255717576, 0.0008782994, -0.0014749423, -0.0597795881])

imgpoints_2 = [] # 在图像平面的二维点
img3dpoints = []
avg_sqrt_proj_list = []
R_all_chess_to_cam, T_all_chess_to_cam = [], []
images_2 = sorted(glob.glob(f'{other_dir}/*.bmp'))  #   拍摄的十几张棋盘图片所在目录
if len(images_2)==0:
    images_2 = sorted(glob.glob(f'{other_dir}/*.png'))  #   拍摄的十几张棋盘图片所在目录
pcs_2 = sorted(glob.glob(f'{other_dir}/*.ply'))  #   拍摄的十几张棋盘图片所在目录
if len(pcs_2)==0:
    pcs_2 = sorted(glob.glob(f'{other_dir}/*.xml'))  #   拍摄的十几张棋盘图片所在目录
print(len(images_2))
i=0
for m in range(len(images_2)):

    print(images_2[m])
    img = cv2.imread(images_2[m])
    # img = undistort(img, camera_other, dis_coe)

    # 获取画面中心点
    #获取图像的长宽
    h1, w1 = img.shape[0], img.shape[1]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    u, v = img.shape[:2]
    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (w,h),None)
    # 如果找到足够点对，将其存储起来
    if ret == True:
        print("i:", i)
        i = i+1
        ## 在原角点的基础上寻找亚像素角点
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        #追加进入世界三维点和平面二维点中
        imgpoints_2.append(corners.reshape(-1,2))

        if pcs_2[0].split('.')[-1]=='ply':
            pcd = o3d.io.read_point_cloud(pcs_2[m])
            pcd_array = np.asarray(pcd.points)
        if pcs_2[0].split('.')[-1]=='xml':
            data = cv2.FileStorage(pcs_2[m], cv2.FILE_STORAGE_READ)
            data_1 = data.getNode("test_filter").mat()
            pcd_array = data_1[:,:,:3].reshape(-1,3)
        
        if np.mean(pcd_array[:,2])>100:
            pcd_array /= 1000
        selected_pc = []
        selected_imgpoints = []
        selected_objps = []
        mz = 0
        multi_camera_registration = MultiCameraRegistration(camera_other, np.eye(4))
        scene_pcd = multi_camera_registration.get_rbgd_pointcloud(pcd_array, img)

        ## mat location indexing

        print("mat location indexing")
        for p in imgpoints_2[i-1]:
            p_id = v* (np.round(p[1])) + np.round(p[0])
            if pcd_array[int(p_id)][2]>0:
                selected_pc.append(pcd_array[int(p_id)])
                selected_objps.append(objp[mz])
            mz += 1

        ## re-projection indexing
        print("re-projection indexing")
        project_index_x, project_index_y = multi_camera_registration.get_project_index()
        for p in imgpoints_2[i-1]:
            corner_2d_inindex = np.where((project_index_y - p[1])**2 +
                                    (project_index_x - p[0])**2 < 0.5**2)[0]
            corner_pcd = scene_pcd.select_by_index(corner_2d_inindex)
            corner_pcd_array = np.asarray(corner_pcd.points)
            if len(corner_pcd_array)!=0:
                corner_3d = np.mean(corner_pcd_array, axis=0)
                selected_pc.append(corner_3d)
                selected_objps.append(objp[mz])
            mz += 1

        print(len(selected_pc))

        a = np.array(selected_objps)
        b = np.array(selected_pc)
        r, t = rigid_transform_3D(a, b)

        bb = np.matmul(a, r.T) + t.reshape([1, 3])
        avg_sqrt_proj_list.append(np.mean(np.sqrt((b - bb)**2), axis=0))

        cam_corner3d = o3d.geometry.PointCloud()
        cam_corner3d.points = o3d.utility.Vector3dVector(np.array(selected_pc))
        chess_corner3d = o3d.geometry.PointCloud()
        chess_corner3d.points = o3d.utility.Vector3dVector(np.array(selected_objps))
        trans_init = np.eye(4)
        trans_init[:3, :3] = r
        trans_init[:3, 3] = t 

        ## visualization
        ## camera accuracy verification

        source_temp = copy.deepcopy(chess_corner3d)
        target_temp = copy.deepcopy(cam_corner3d)
        source_temp.paint_uniform_color([0, 0, 1])
        target_temp.paint_uniform_color([1, 0, 0])
        source_temp.transform(trans_init)
        o3d.visualization.draw_geometries([source_temp, target_temp])
        
        o3d.io.write_point_cloud(other_dir+'/real_board_pcd.ply', source_temp)
        o3d.io.write_point_cloud(other_dir+'/measured_board_pcd.ply', target_temp)

        evaluation = o3d.pipelines.registration.evaluate_registration(chess_corner3d, cam_corner3d, 0.01, trans_init)
        print(evaluation)
        reg_p2p = o3d.pipelines.registration.registration_icp(chess_corner3d, cam_corner3d, 0.005, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
        print(reg_p2p)
        R_all_chess_to_cam.append(trans_init[:3,:3])
        T_all_chess_to_cam.append(trans_init[:3, 3].reshape((3,1)))
        

        ##---------------------visualization------------------------------
        
        cam_corner3d.paint_uniform_color([1,0,0])
        O = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05)
        init_pose = np.eye(4).astype(np.float32)
        init_pose[:3, 3] = selected_pc[1]
        O.transform(init_pose)
        o3d.visualization.draw_geometries([cam_corner3d, scene_pcd, O])

        proj_matrix = np.zeros((3, 4))
        proj_matrix[2, 3] = 1
        proj_matrix[:3, :3] = camera_other
        pc_array = np.concatenate([np.array(selected_pc), 
                                np.ones((len(selected_pc), 1))], axis=1) 
        pc_rgb_array = np.eye(4).dot(pc_array.T).T[:, :3]
        pc_rgb_array /= pc_rgb_array[:, 2:]
        proj_rgb_xy = np.round(camera_other.dot(pc_rgb_array.T).T[:, :2]).astype('int64')
        proj_rgb_xy_2 = np.round(imgpoints_2[i-1]).astype('int64')
        img[proj_rgb_xy[:, 1], proj_rgb_xy[:, 0]] = np.array([0,0,255])
        img[proj_rgb_xy_2[:, 1], proj_rgb_xy_2[:, 0]] = np.array([255,0,0])
        plt.figure()
        plt.imshow(img[:, :, ::-1])
        plt.show()
        ##---------------------visualization------------------------------

#         
print('avg sqrt b-bb:', np.mean(avg_sqrt_proj_list, axis=0))

end_to_base_quat = []
with open(other_dir + "/JointStateSteps.csv", 'r') as file:
  csvreader = csv.reader(file)
  for row in csvreader:
    if 'marker' in row:
       continue
    row_list = [float(x) for x in row[0].split(',')]
    end_to_base_quat.append(row_list)       
R_all_base_to_end, T_all_base_to_end = [], []
end_to_base_quat = np.array(end_to_base_quat)

for i in range(len(end_to_base_quat)):
    rot = Rot.from_quat(end_to_base_quat[i][3:]).as_matrix() #quat scalar-last (x, y, z, w) format
    homo_matrix = np.eye(4)
    homo_matrix[:3, :3] = rot
    homo_matrix[:3, 3] = end_to_base_quat[i][0:3]
    # eye to hand
    # inv_homo_matrix = np.linalg.inv(homo_matrix)
    # R_all_base_to_end.append(inv_homo_matrix[:3, :3])
    # T_all_base_to_end.append(inv_homo_matrix[:3, 3].reshape((3,1)))
    # eye in hand
    R_all_base_to_end.append(homo_matrix[:3, :3])
    T_all_base_to_end.append(homo_matrix[:3, 3].reshape((3,1)))

R, T = cv2.calibrateHandEye(R_all_base_to_end, T_all_base_to_end, R_all_chess_to_cam, T_all_chess_to_cam, method=1)#手眼标定
np.set_printoptions(suppress = True)
print("hand-eye-calibration R: \n", np.array2string(R, separator=","))
print("hand-eye-calibration T: \n", np.array2string(T, separator=","))
RT_quat = np.zeros(7)
RT_quat[0:3] = T[:, 0]
RT_quat[3:] = Rot.from_matrix(R).as_quat()
print("hand-eye-calibration Quat: \n", np.array2string(RT_quat, separator=","))

rotvec_list = []
translation_list = []

for i in range(len(end_to_base_quat)):

    RT_base_to_end = np.column_stack((R_all_base_to_end[i], T_all_base_to_end[i]))
    RT_base_to_end = np.row_stack((RT_base_to_end, np.array([0,0,0,1])))
    # print(RT_base_to_end)

    RT_chess_to_cam = np.column_stack((R_all_chess_to_cam[i],T_all_chess_to_cam[i]))
    RT_chess_to_cam = np.row_stack((RT_chess_to_cam,np.array([0,0,0,1])))
    # print(RT_chess_to_cam)

    RT_cam_to_base = np.column_stack((R,T))
    RT_cam_to_base = np.row_stack((RT_cam_to_base, np.array([0,0,0,1])))
    # print(RT_cam_to_end)

    RT_chess_to_end = RT_base_to_end@RT_cam_to_base@RT_chess_to_cam#即为固定的棋盘格相对于机器人基坐标系位姿
    rotvec = Rot.from_matrix(RT_chess_to_end[:3, :3]).as_rotvec()
    # print(i,'th:', "rotation vector:", rotvec, "translation:", RT_chess_to_end[:3, 3])
    rotvec_list.append(rotvec)
    translation_list.append(RT_chess_to_end[:3, 3])
np.set_printoptions(suppress = False)
print("rotation vector variance:", np.var(rotvec_list, axis=0), "translation variance:", np.var(translation_list, axis=0))
print("rotation vector change range:", np.max(rotvec_list, axis=0) - np.min(rotvec_list, axis=0), 
      "translation change range:", np.max(translation_list, axis=0) - np.min(translation_list, axis=0))