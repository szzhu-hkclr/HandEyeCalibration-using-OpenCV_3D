import cv2
import numpy as np
import glob
import open3d as o3d
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as Rot

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

root_folder = './10-13_twocamera' 
realsen_dir = root_folder + "/realsense_1920"
other_dir = root_folder + "/photoneo"

images_1 = sorted(glob.glob(f'{realsen_dir}/*.png'))  #   拍摄的十几张棋盘图片所在目录
print(len(images_1))
i=0
for fname in images_1:
    print(fname)
    img = cv2.imread(fname)
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
        # 在原角点的基础上寻找亚像素角点
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        #追加进入世界三维点和平面二维点中
        objpoints.append(objp)
        imgpoints.append(corners.reshape(-1,2))

# last_img = cv2.imread(images_1[-1])
# gray = cv2.cvtColor(last_img, cv2.COLOR_BGR2GRAY)
# ret, cameraMatrix, distCoeffs, rvecs, tvecs = \
#     cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# print(cameraMatrix)
# print(distCoeffs)
# print(ret)



imgpoints_2 = [] # 在图像平面的二维点
img3dpoints_2d = []
img3dpoints_2_2d = []
img3dpoints = []
img3dpoints_num_list = [0]
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

            # pcd_ = o3d.geometry.PointCloud()
            # pcd_.points = o3d.utility.Vector3dVector(np.array(pcd_array/1000))
            # o3d.io.write_point_cloud('..\Test_hkclrcamera\pc.ply', pcd_)
        
        print(pcs_2[m])
        if np.mean(pcd_array[:,2])>100:
            pcd_array /= 1000
        selected_pc = []
        selected_imgpoints = []
        selected_imgpoints_2 = []
        mz = 0
        for p in imgpoints_2[i-1]:
            p_id = v* (np.round(p[1])) + np.round(p[0])
            if pcd_array[int(p_id)][2]>0:
                selected_pc.append(pcd_array[int(p_id)])
                selected_imgpoints.append(imgpoints[i-1][mz])
                selected_imgpoints_2.append(p)
            mz += 1
        img3dpoints.extend(selected_pc)
        img3dpoints_2d.extend(selected_imgpoints)
        img3dpoints_2_2d.extend(selected_imgpoints_2)
        print(len(selected_pc))
        img3dpoints_num_list.append(len(selected_pc)+img3dpoints_num_list[-1])
        
        # vis_corner3d = o3d.geometry.PointCloud()
        # vis_corner3d.points = o3d.utility.Vector3dVector(np.array(selected_pc))
        # vis_corner3d.paint_uniform_color([1,0,0])
        # O = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05)
        # init_pose = np.eye(4).astype(np.float32)
        # init_pose[:3, 3] = selected_pc[1]
        # O.transform(init_pose)
        # o3d.visualization.draw_geometries([vis_corner3d, pcd, O])
       
# cameraMatrix_realsense_1920 = np.array([[1333.17, 0., 971.978], 
#                          [0., 1333.17, 549.778],
#                          [0., 0., 1.]])
cameraMatrix_realsense_1920 = np.array([[967.427, 0., 972.016], 
                         [0., 967.427, 539.719],
                         [0., 0., 1.]])
cameraMatrix_realsense_1280 = np.array([[888.78, 0., 647.985], 
                         [0., 888.78, 366.518],
                         [0., 0., 1.]])
cameraMatrix_photoneo = np.array([[2334.96, 0., 1054.91], 
                         [0., 2335.22, 759.454],
                         [0., 0., 1.]])
distCoeffs_photoneo = np.array([-0.240671, 0.162922, 0.000728072, 0.000214813, -0.0681146])

# camera_other = np.array([[2342.98, 0, 1049.54], #16w photoneo
#                         [0, 2343.49, 732.944],
#                         [0, 0, 1]])
# distCoeffs_other = np.array([-0.236918, 0.135125, -8.18265e-05, 0.000873692, -0.0144721]) #16w photoneo

camera_other = np.array([[2334.96, 0, 1054.91], #Maishuo photoneo
                        [0, 2335.22, 759.454],
                        [0, 0, 1]]) 
distCoeffs_other = np.array([-0.240671, 0.162922, 0.000728072, 0.000214813, -0.0681146]) #Maishuo photoneo

cameraMatrix = cameraMatrix_realsense_1920
distCoeffs = np.zeros((1, 5))

success, rotation_vector, translation_vector = cv2.solvePnP(np.array(img3dpoints), np.array(img3dpoints_2d), cameraMatrix, distCoeffs)

np.set_printoptions(suppress = True)
rot_matrix = Rot.from_rotvec(rotation_vector.flatten()).as_matrix()
print(rot_matrix) 
realsense_to_pc_mat =  np.eye(4)
realsense_to_pc_mat[:3, :3] = rot_matrix 
realsense_to_pc_mat[:3, 3:] = translation_vector
pc_to_realsense_mat = np.linalg.inv(realsense_to_pc_mat)
print('pc_to_realsense_mat', np.array2string(pc_to_realsense_mat, separator=","))


#------------------------------------------------------------------------
## registration for rgbd
multi_camera_registration = MultiCameraRegistration(cameraMatrix, pc_to_realsense_mat)
id = 1
img = cv2.imread(images_1[id])
img = cv2.undistort(img, cameraMatrix, distCoeffs)
if pcs_2[0].split('.')[-1]=='ply':
    scene_pcd = o3d.io.read_point_cloud(pcs_2[id])
    scene_pc_array = np.asarray(scene_pcd.points)
if pcs_2[0].split('.')[-1]=='xml':
    data = cv2.FileStorage(pcs_2[id], cv2.FILE_STORAGE_READ)
    data_1 = data.getNode("test_filter").mat()
    scene_pc_array = data_1[:,:,:3].reshape(-1,3)
if np.mean(scene_pc_array[:,2])>100:
    scene_pc_array /= 1000
scene_pcd = multi_camera_registration.get_rbgd_pointcloud(scene_pc_array, img)
o3d.visualization.draw_geometries([scene_pcd])

proj_matrix = np.zeros((3, 4))
proj_matrix[2, 3] = 1
proj_matrix[:3, :3] = cameraMatrix
pc_array = np.concatenate([np.array(img3dpoints[img3dpoints_num_list[id]: img3dpoints_num_list[id+1]]), 
                           np.ones((img3dpoints_num_list[id+1]-img3dpoints_num_list[id], 1))], axis=1) 
pc_rgb_array = realsense_to_pc_mat.dot(pc_array.T).T[:, :3]
pc_rgb_array /= pc_rgb_array[:, 2:]
proj_rgb_xy = np.round(cameraMatrix.dot(pc_rgb_array.T).T[:, :2]).astype('int64')
# pc_array_center = np.mean(np.concatenate([np.array(img3dpoints[img3dpoints_num_list[id]: img3dpoints_num_list[id+1]]), 
#                                           np.ones((img3dpoints_num_list[id+1]-img3dpoints_num_list[id], 1))], axis=1), axis=0, keepdims=True)
# pc_array_center = realsense_to_pc_mat.dot(pc_array_center.T).T[:, :3]
# pc_array_center /= pc_array_center[:, 2:]
# proj_rgb_xy_center = np.round(cameraMatrix.dot(pc_array_center.T).T[:, :2]).astype('int64').squeeze()
img[proj_rgb_xy[:, 1], proj_rgb_xy[:, 0]] = np.array([0,0,255])
# img[proj_rgb_xy_center[1], proj_rgb_xy_center[0]] = np.array([0,255,0])
plt.figure()
plt.imshow(img[:, :, ::-1])
plt.show()
#------------------------------------------------------------------------



## test ordered pointcloud mat for hkclr scanner
# camera_other = np.array([[2.0107621997829399e+03, 0., 1.0191838297247446e+03], 
#                          [0., 2.0096828574901108e+03, 5.4904360707454725e+02],
#                          [0., 0., 1.]])
# pc_to_2d_mat = np.identity(4)
# multi_camera_registration = MultiCameraRegistration(camera_other, pc_to_2d_mat)
# id=1
# img = cv2.imread(images_2[id])
# if pcs_2[0].split('.')[-1]=='xml':
#     data = cv2.FileStorage(pcs_2[id], cv2.FILE_STORAGE_READ)
#     data_1 = data.getNode("test_filter").mat()
#     scene_pc_array = data_1[:,:,:3].reshape(-1,3)
# else: 
#     pcd = o3d.io.read_point_cloud(pcs_2[m])
#     scene_pc_array = np.asarray(pcd.points)
# if np.mean(scene_pc_array[:,2])>100:
#     scene_pc_array /= 1000
# scene_pcd = multi_camera_registration.get_rbgd_pointcloud(scene_pc_array, img)
# o3d.visualization.draw_geometries([scene_pcd])
# # o3d.io.write_point_cloud('..\Test_hkclrcamera\pc.ply', scene_pcd)

# proj_matrix = np.zeros((3, 4))
# proj_matrix[2, 3] = 1
# proj_matrix[:3, :3] = camera_other
# pc_array = np.concatenate([np.array(img3dpoints[img3dpoints_num_list[id]: img3dpoints_num_list[id+1]]), 
#                            np.ones((img3dpoints_num_list[id+1]-img3dpoints_num_list[id], 1))], axis=1) 
# pc_rgb_array = pc_to_2d_mat.dot(pc_array.T).T[:, :3]
# pc_rgb_array /= pc_rgb_array[:, 2:]
# proj_rgb_xy = np.round(camera_other.dot(pc_rgb_array.T).T[:, :2]).astype('int64')
# pc_array_center = np.mean(np.concatenate([np.array(img3dpoints[img3dpoints_num_list[id]: img3dpoints_num_list[id+1]]), 
#                                           np.ones((img3dpoints_num_list[id+1]-img3dpoints_num_list[id], 1))], axis=1), axis=0, keepdims=True)
# pc_array_center = pc_to_2d_mat.dot(pc_array_center.T).T[:, :3]
# pc_array_center /= pc_array_center[:, 2:]
# proj_rgb_xy_center = np.round(camera_other.dot(pc_array_center.T).T[:, :2]).astype('int64').squeeze()
# proj_rgb_xy_2 = np.round(img3dpoints_2_2d[img3dpoints_num_list[id]: img3dpoints_num_list[id+1]]).astype('int64')
# img[proj_rgb_xy[:, 1], proj_rgb_xy[:, 0]] = np.array([0,0,255])
# img[proj_rgb_xy_2[:, 1], proj_rgb_xy_2[:, 0]] = np.array([255,0,0])
# plt.figure()
# plt.imshow(img[:, :, ::-1])
# plt.show()
