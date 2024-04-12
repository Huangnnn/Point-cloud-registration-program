# 面部多视角ICP点云配准实验
# 统一图像的左中右。0号相机：右 1号相机：中 2号相机：左
# 区分左中右的时候使用0,1,2也行，能直接对应到图像数据
# 功能 能够生成初步匹配后的多视角图
# 2023-02-23
# 可以输出点云粗配准和精配准后的R,t信息

import os
import numpy as np
import cv2
import open3d
from Multi_Camera_functions import *
from depth_fill_hole.fill_hole_2 import depth_hole_filling

# mode 共两种，1.single_mode 计算单个人的
#            2.dir_mode 计算一个文件夹下所有人


image_02_dir = r'E:\datasets\or_po\test\images'
image_1_dir = r'F:\or_datasets\test\images'
depth_dir = r'E:\datasets\or_po\depths'
json_02_dir = r'E:\datasets\or_po\test\labelme_json_mp'
json_1_dir = r'F:\or_datasets\test\labelme_json_mp'
face_json_dir = r'E:\datasets\or_po\test\face_json'

filenames = os.listdir(image_02_dir)
print(filenames)

angles_0 = []
angles_2 = []
angles_0_fine = []
angles_2_fine = []
test_0_coarse_distances = []
test_2_coarse_distances = []
test_0_fine_distances = []
test_2_fine_distances = []
uids = []
for idx, filename in enumerate(filenames):
    if '.png' in filename:
        uid = filename.split('_color_')[0]
        uids.append(uid)
    else:
        pass

    # uids = list(set(uids))
    # print(idx+1, len(uids), filename)

uids = list(set(uids))

uids.sort()
uids.remove('2021_1025_098')
uids.remove('2021_1025_098z')

index_list = [11, 12, 27, 29, 44, 64, 76, 79, 87, 89, 91, 92, 113]
index_list.reverse()
for idx in index_list:
    uids.pop(idx)

# uids = uids[113:114]
# uids = uids[76:77]
# uids = uids[64:65]
# uids = uids[12:13]
# uids = uids[79:80]
# for uid in uids[20:21]:
# for uid in uids[117:118]:
# for uid in uids[77:78]:
# for uid in uids[98:99]:
# for uid in uids[0:30]:
# for uid in uids[-10:]:
# for uid in uids[0:73] + uids[75:]:

# uids = uids[0:1]

for uid in uids:

    image_0_path = os.path.join(image_02_dir, uid + '_color_0.png')
    depth_0_path = os.path.join(depth_dir, uid + '_depth_0.png')
    image_1_path = os.path.join(image_1_dir, uid + '_color_1.png')
    depth_1_path = os.path.join(depth_dir, uid + '_depth_1.png')
    image_2_path = os.path.join(image_02_dir, uid + '_color_2.png')
    depth_2_path = os.path.join(depth_dir, uid + '_depth_2.png')

    json_0_path = os.path.join(json_02_dir, uid + '_color_0.json')
    json_1_path = os.path.join(json_1_dir, uid + '_color_1.json')
    json_2_path = os.path.join(json_02_dir, uid + '_color_2.json')

    face_json_0_path = os.path.join(face_json_dir, uid + '_color_0.json')
    face_json_1_path = os.path.join(face_json_dir, uid + '_color_1.json')
    face_json_2_path = os.path.join(face_json_dir, uid + '_color_2.json')

    print('json_0_path:', json_0_path)
    print('json_1_path:', json_1_path)
    print('json_2_path:', json_2_path)

    json_0 = json.load(open(json_0_path, 'r'), encoding='utf-8')
    json_1 = json.load(open(json_1_path, 'r'), encoding='utf-8')
    json_2 = json.load(open(json_2_path, 'r'), encoding='utf-8')

    face_json_0 = json.load(open(face_json_0_path, 'r'), encoding='utf-8')
    face_json_1 = json.load(open(face_json_1_path, 'r'), encoding='utf-8')
    face_json_2 = json.load(open(face_json_2_path, 'r'), encoding='utf-8')

    # print('json_0:', json_0['shapes'][0]['points'])
    # print('json_1:', json_1['shapes'][0]['points'])
    # print('json_2:', json_2['shapes'][0]['points'])

    # print('face_json_0', face_json_0)
    face_mask_points_0 = face_json_0['shapes'][0]['points']
    face_mask_points_1 = face_json_1['shapes'][0]['points']
    face_mask_points_2 = face_json_2['shapes'][0]['points']
    face_mask_points_0 = np.round(face_mask_points_0).astype(np.int64)
    face_mask_points_1 = np.round(face_mask_points_1).astype(np.int64)
    face_mask_points_2 = np.round(face_mask_points_2).astype(np.int64)

    face_mask_0 = np.zeros((720, 1280, 1), dtype=np.uint8)
    face_mask_1 = np.zeros((720, 1280, 1), dtype=np.uint8)
    face_mask_2 = np.zeros((720, 1280, 1), dtype=np.uint8)

    face_mask_0 = cv2.fillPoly(face_mask_0, [face_mask_points_0], (255,))
    face_mask_1 = cv2.fillPoly(face_mask_1, [face_mask_points_1], (255,))
    face_mask_2 = cv2.fillPoly(face_mask_2, [face_mask_points_2], (255,))

    # 显示人脸面部区域的mask
    # cv2.imshow('face_mask_0', face_mask_0)
    # cv2.imshow('face_mask_1', face_mask_1)
    # cv2.imshow('face_mask_2', face_mask_2)
    # cv2.waitKey(0)

    json_0_14p = []
    json_1_14p = []
    json_2_14p = []

    # for循环里面的7和5指的是json标记文件中14个点在文件中的偏移位置
    for idx in range(14):
        json_0_14p.append(json_0['shapes'][idx+7]['points'][0])
        json_1_14p.append(json_1['shapes'][idx+5]['points'][0])
        json_2_14p.append(json_2['shapes'][idx+7]['points'][0])

    json_0_14p = np.round(json_0_14p).astype(np.uint16).tolist()
    json_1_14p = np.round(json_1_14p).astype(np.uint16).tolist()
    json_2_14p = np.round(json_2_14p).astype(np.uint16).tolist()

    # 读取图像
    color_0 = cv2.imdecode(np.fromfile(image_0_path, dtype=np.uint8), -1)
    depth_0 = cv2.imdecode(np.fromfile(depth_0_path, dtype=np.uint8), -1)
    color_1 = cv2.imdecode(np.fromfile(image_1_path, dtype=np.uint8), -1)
    depth_1 = cv2.imdecode(np.fromfile(depth_1_path, dtype=np.uint8), -1)
    color_2 = cv2.imdecode(np.fromfile(image_2_path, dtype=np.uint8), -1)
    depth_2 = cv2.imdecode(np.fromfile(depth_2_path, dtype=np.uint8), -1)

    # cv2.imshow('color_0', color_0)

    # 显示带特征点的彩色图像
    # radius = 2
    # color = (255, 255, 0)
    # thickness = 2
    #
    # for idx, point in enumerate(np.array(json_0_14p)[[2,3,6,7,9,11,12,13]]):
    #     center_coordinates = point
    #     print(center_coordinates)
    #     color_0 = cv2.circle(color_0, center_coordinates, radius, color, thickness)
    #
    # for idx, point in enumerate(np.array(json_1_14p)[[2,3,6,7,9,11,12,13]]):
    #     center_coordinates = point
    #     print(center_coordinates)
    #     color_1 = cv2.circle(color_1, center_coordinates, radius, color, thickness)
    #
    # for idx, point in enumerate(np.array(json_2_14p)[[0,1,4,5,8,10,12,13]]):
    #     center_coordinates = point
    #     print(center_coordinates)
    #     color_2 = cv2.circle(color_2, center_coordinates, radius, color, thickness)
    #
    # for idx, point in enumerate(np.array(json_1_14p)[[0,1,4,5,8,10,12,13]]):
    #     center_coordinates = point
    #     print(center_coordinates)
    #     color_1 = cv2.circle(color_1, center_coordinates, radius, color, thickness)
    #
    # cv2.imshow('color_0', color_0)
    # cv2.imshow('color_1', color_1)
    # cv2.imshow('color_2', color_2)
    # cv2.waitKey(0)

    # 深度图孔洞填充
    for idx in range(14):
        i = json_0_14p[idx][1]
        j = json_0_14p[idx][0]
        if (depth_0[i][j] == 0):
            # print(i,j)
            tmp = depth_0[(i - 1):(i + 2), (j - 1):(j + 2)]
            # print(tmp)
            if (len(tmp[np.nonzero(tmp)]) > 0):
                depth_0[i][j] = tmp[np.nonzero(tmp)].mean()
            else:
                tmp = depth_0[(i - 2):(i + 3), (j - 2):(j + 3)]
                if (len(tmp[np.nonzero(tmp)]) > 0):
                    depth_0[i][j] = tmp[np.nonzero(tmp)].mean()

        i = json_1_14p[idx][1]
        j = json_1_14p[idx][0]
        if (depth_1[i][j] == 0):
            # print(i,j)
            tmp = depth_1[(i - 1):(i + 2), (j - 1):(j + 2)]
            # print(tmp)
            if (len(tmp[np.nonzero(tmp)]) > 0):
                depth_1[i][j] = tmp[np.nonzero(tmp)].mean()
            else:
                tmp = depth_1[(i - 2):(i + 3), (j - 2):(j + 3)]
                if (len(tmp[np.nonzero(tmp)]) > 0):
                    depth_1[i][j] = tmp[np.nonzero(tmp)].mean()

        i = json_2_14p[idx][1]
        j = json_2_14p[idx][0]
        if (depth_2[i][j] == 0):
            # print(i,j)
            tmp = depth_2[(i - 1):(i + 2), (j - 1):(j + 2)]
            # print(tmp)
            if (len(tmp[np.nonzero(tmp)]) > 0):
                depth_2[i][j] = tmp[np.nonzero(tmp)].mean()
            else:
                tmp = depth_2[(i - 2):(i + 3), (j - 2):(j + 3)]
                if (len(tmp[np.nonzero(tmp)]) > 0):
                    depth_2[i][j] = tmp[np.nonzero(tmp)].mean()

    # print('json_0_14p', json_0_14p)
    # depth_0 = depth_hole_filling(depth_0)
    # depth_0 = depth_hole_filling(depth_0)
    # depth_1 = depth_hole_filling(depth_1)
    # depth_1 = depth_hole_filling(depth_1)
    # depth_2 = depth_hole_filling(depth_2)
    # depth_2 = depth_hole_filling(depth_2)

    # 深度图转点云
    xyz0 = cv2.rgbd.depthTo3d(depth_0, camera_0_depth_intrinsics)
    xyz1 = cv2.rgbd.depthTo3d(depth_1, camera_1_depth_intrinsics)
    xyz2 = cv2.rgbd.depthTo3d(depth_2, camera_2_depth_intrinsics)
    # 通过cv2的rgbd函数转点云，深度为0的点会转为nan，这样在open3d中是无法显示的，可以把为nan的点转为[0,0,0]，或后面显示前删除
    xyz0 = np.nan_to_num(xyz0) * 100
    xyz1 = np.nan_to_num(xyz1) * 100
    xyz2 = np.nan_to_num(xyz2) * 100


    # 坐标系变换，x不变，y和z轴取反，在这里xyz的shape为m*n*3
    xyz0[:, :, 1:3] = -xyz0[:, :, 1:3]
    xyz1[:, :, 1:3] = -xyz1[:, :, 1:3]
    xyz2[:, :, 1:3] = -xyz2[:, :, 1:3]

    # 获取n*3格式点云
    points_0 = xyz0.reshape((-1, 3))
    points_1 = xyz1.reshape((-1, 3))
    points_2 = xyz2.reshape((-1, 3))
    # points = np.vstack((points_0, points_1, points_2))

    # byr转rgb，转成n行3列的格式
    rgb_0 = cv2.cvtColor(color_0, cv2.COLOR_BGR2RGB)
    rgb_1 = cv2.cvtColor(color_1, cv2.COLOR_BGR2RGB)
    rgb_2 = cv2.cvtColor(color_2, cv2.COLOR_BGR2RGB)
    rgb_0 = rgb_0.reshape(-1, 3)
    rgb_1 = rgb_1.reshape(-1, 3)
    rgb_2 = rgb_2.reshape(-1, 3)

    # 删除非mask区域的点
    index = np.where(face_mask_0.reshape(-1, 1) == 0)
    rgb_0 = np.delete(rgb_0, index, axis=0)
    points_0 = np.delete(points_0, index, axis=0)

    index = np.where(face_mask_1.reshape(-1, 1) == 0)
    rgb_1 = np.delete(rgb_1, index, axis=0)
    points_1 = np.delete(points_1, index, axis=0)

    index = np.where(face_mask_2.reshape(-1, 1) == 0)
    rgb_2 = np.delete(rgb_2, index, axis=0)
    points_2 = np.delete(points_2, index, axis=0)

    face_points_0 = open3d.geometry.PointCloud()
    face_points_1 = open3d.geometry.PointCloud()
    face_points_2 = open3d.geometry.PointCloud()
    face_points_0.points = open3d.utility.Vector3dVector(points_0)
    face_points_1.points = open3d.utility.Vector3dVector(points_1)
    face_points_2.points = open3d.utility.Vector3dVector(points_2)
    face_points_0.paint_uniform_color([1, 0, 0])
    face_points_1.paint_uniform_color([0, 1, 0])
    face_points_2.paint_uniform_color([0, 0, 1])
    # face_points_0.colors = open3d.utility.Vector3dVector(rgb_0/255)
    # face_points_1.colors = open3d.utility.Vector3dVector(rgb_1/255)
    # face_points_2.colors = open3d.utility.Vector3dVector(rgb_2/255)

    # open3d.visualization.draw_geometries([face_points_0, face_points_1, face_points_2])

    test_0 = [json_0['shapes'][25]['points'][0], json_0['shapes'][26]['points'][0],
              json_0['shapes'][28]['points'][0], json_0['shapes'][27]['points'][0]]
    test_2 = [json_2['shapes'][21]['points'][0], json_2['shapes'][22]['points'][0],
              json_2['shapes'][24]['points'][0], json_2['shapes'][23]['points'][0]]
    mask_test_0 = np.zeros((720, 1280, 1), dtype=np.uint8)
    mask_test_2 = np.zeros((720, 1280, 1), dtype=np.uint8)
    test_0 = np.round(test_0).astype(np.int32).reshape(-1, 1, 2)
    test_2 = np.round(test_2).astype(np.int32).reshape(-1, 1, 2)

    cv2.fillPoly(mask_test_0, [test_0], (255))
    cv2.fillPoly(mask_test_2, [test_2], (255))

    # cv2.imshow('mask_test_0', mask_test_0)
    # cv2.imshow('mask_test_2', mask_test_2)
    # cv2.waitKey(0)

    # index = np.where(mask_test_0.reshape(-1, 1) != 0)
    # aa = xyz0.reshape((-1, 3))[index[0], :]
    index_test_0 = np.where(mask_test_0.flatten() != 0)
    test_0_np = xyz0.reshape((-1, 3))[index_test_0[0], :]
    index_test_2 = np.where(mask_test_2.flatten() != 0)
    test_2_np = xyz2.reshape((-1, 3))[index_test_2[0], :]

    points_test_0 = open3d.geometry.PointCloud()
    points_test_2 = open3d.geometry.PointCloud()
    points_test_0.points = open3d.utility.Vector3dVector(test_0_np)
    points_test_0.paint_uniform_color([1, 1, 0])
    points_test_0.paint_uniform_color([0.5, 1, 0.5])
    # open3d.visualization.draw_geometries([points_test_0, face_points_1])


    points_14_0 = np.zeros([14, 3])
    points_14_1 = np.zeros([14, 3])
    points_14_2 = np.zeros([14, 3])
    for idx in range(14):
        points_14_0[idx, :] = xyz0[json_0_14p[idx][1], json_0_14p[idx][0], :]
        points_14_1[idx, :] = xyz1[json_1_14p[idx][1], json_1_14p[idx][0], :]
        points_14_2[idx, :] = xyz2[json_2_14p[idx][1], json_2_14p[idx][0], :]

    # 中间的以两个眼角点的中点原点，两侧的以一个眼角点或鼻下点为原点进行显示
    points_init_base_0 = points_14_0[5,:]
    points_init_base_1 = points_14_1[[5,6],:].mean(axis=0)
    points_init_base_2 = points_14_2[6,:]
    # points_init_0 = points_14_0[[2,3,6,7,9,11,12,13],:] - points_init_base_0
    # points_init_01 = points_14_1[[2,3,6,7,9,11,12,13],:] - points_init_base_1
    # points_init_2 = points_14_2[[0,1,4,5,8,10,12,13],:] - points_init_base_2
    # points_init_21 = points_14_1[[0,1,4,5,8,10,12,13],:] - points_init_base_1
    points_init_0 = points_14_0[[2, 3, 6, 7, 9, 11, 12, 13], :]
    points_init_01 = points_14_1[[2, 3, 6, 7, 9, 11, 12, 13], :]
    points_init_2 = points_14_2[[0, 1, 4, 5, 8, 10, 12, 13], :]
    points_init_21 = points_14_1[[0, 1, 4, 5, 8, 10, 12, 13], :]

    o3d_p0_init = open3d.geometry.PointCloud()
    o3d_p01_init = open3d.geometry.PointCloud()
    o3d_p2_init = open3d.geometry.PointCloud()
    o3d_p21_init = open3d.geometry.PointCloud()

    o3d_p0_init.points = open3d.utility.Vector3dVector(points_init_0)
    o3d_p01_init.points = open3d.utility.Vector3dVector(points_init_01)
    o3d_p2_init.points = open3d.utility.Vector3dVector(points_init_2)
    o3d_p21_init.points = open3d.utility.Vector3dVector(points_init_21)
    o3d_p0_init.paint_uniform_color([1, 0, 0])
    o3d_p01_init.paint_uniform_color([0, 1, 0])
    o3d_p2_init.paint_uniform_color([0, 0, 1])
    o3d_p21_init.paint_uniform_color([0, 1, 1])

    coord = open3d.geometry.TriangleMesh.create_coordinate_frame()
    # open3d.visualization.draw_geometries([coord, o3d_p0_init, o3d_p1_init, o3d_p2_init])
    # open3d.visualization.draw_geometries([coord, o3d_p0_init, o3d_p01_init])
    # open3d.visualization.draw_geometries([coord, o3d_p2_init, o3d_p21_init])
    o3d_p0_init.paint_uniform_color([0, 1, 1])
    # open3d.visualization.draw_geometries([coord, o3d_p01_init, face_points_1, o3d_p01_init])
    # open3d.visualization.draw_geometries([coord, o3d_p0_init, face_points_0, o3d_p0_init, o3d_p01_init])

    mat0, vector0 = Face_ICP(points_src=points_init_0, points_dst=points_init_01)
    mat2, vector2 = Face_ICP(points_src=points_init_2, points_dst=points_init_21)

    # print(mat0, vector0)
    # print(mat2, vector2)
    angles0 = rotationMatrixToEulerAngles(mat0)
    angles2 = rotationMatrixToEulerAngles(mat2)
    print(np.rad2deg(angles0))
    print(np.rad2deg(angles2))
    angles_0.append(angles0)
    angles_2.append(angles2)

    points_0 = np.matmul(mat0, points_0.T).T + vector0
    points_2 = np.matmul(mat2, points_2.T).T + vector2
    face_points_0.points = open3d.utility.Vector3dVector(points_0)
    face_points_2.points = open3d.utility.Vector3dVector(points_2)
    open3d.visualization.draw_geometries([face_points_0, face_points_1, face_points_2], window_name='粗配准结果')

    # 3D-3D ICP 进行R，t迭代优化
    user_path = os.path.join(image_02_dir, uid)
    _source_0 = points_0
    _source_2 = points_2
    _target = points_1

    print('user_path', uid, user_path)
    Homo_0, Homo_2 = ICP_3D_3D(user_path, _source_0, _source_2, _target)
    angle_0_fine = np.rad2deg(rotationMatrixToEulerAngles(np.matmul(Homo_0[0:3, 0:3], mat0)))
    angle_2_fine = np.rad2deg(rotationMatrixToEulerAngles(np.matmul(Homo_2[0:3, 0:3], mat2)))
    # print('angle_0_fine', angle_0_fine)
    # print('angle_2_fine', angle_2_fine)
    angles_0_fine.append(angle_0_fine)
    angles_2_fine.append(angle_2_fine)

    test_0_coarse_np = np.matmul(mat0, test_0_np.T).T + vector0
    test_2_coarse_np = np.matmul(mat2, test_2_np.T).T + vector2
    test_0_fine_np = np.matmul(Homo_0[0:3, 0:3], (np.matmul(mat0, test_0_np.T).T + vector0).T).T + Homo_0[0:3, 3]
    test_2_fine_np = np.matmul(Homo_2[0:3, 0:3], (np.matmul(mat2, test_2_np.T).T + vector2).T).T + Homo_2[0:3, 3]
    points_test_0.points = open3d.utility.Vector3dVector(test_0_fine_np)
    points_test_2.points = open3d.utility.Vector3dVector(test_2_fine_np)
    open3d.visualization.draw_geometries([points_test_0, points_test_2, face_points_0, face_points_1, face_points_2], window_name='精配准结果')
    # open3d.visualization.draw_geometries([points_test_0, face_points_1])

    # test_0_coarse_distance = []
    # for idx in range(len(test_0_coarse_np)):
    #     aa = np.linalg.norm(test_0_coarse_np[idx] - points_1, axis=1)
    #     aa.sort()
    #     test_0_coarse_distance.append(aa[0])
    #
    # test_2_coarse_distance = []
    # for idx in range(len(test_2_coarse_np)):
    #     aa = np.linalg.norm(test_2_coarse_np[idx] - points_1, axis=1)
    #     aa.sort()
    #     test_2_coarse_distance.append(aa[0])
    #
    # test_0_fine_distance = []
    # for idx in range(len(test_0_fine_np)):
    #     aa = np.linalg.norm(test_0_fine_np[idx] - points_1, axis=1)
    #     aa.sort()
    #     test_0_fine_distance.append(aa[0])
    #
    # test_2_fine_distance = []
    # for idx in range(len(test_2_fine_np)):
    #     aa = np.linalg.norm(test_2_fine_np[idx] - points_1, axis=1)
    #     aa.sort()
    #     test_2_fine_distance.append(aa[0])
    #
    # test_0_coarse_distance_mean = np.array(test_0_coarse_distance).mean()
    # test_2_coarse_distance_mean = np.array(test_2_coarse_distance).mean()
    # print('test_0_coarse_distance_mean:', test_0_coarse_distance_mean)
    # print('test_2_coarse_distance_mean:', test_2_coarse_distance_mean)
    # test_0_coarse_distances.append(test_0_coarse_distance_mean)
    # test_2_coarse_distances.append(test_2_coarse_distance_mean)
    #
    # test_0_fine_distance_mean = np.array(test_0_fine_distance).mean()
    # test_2_fine_distance_mean = np.array(test_2_fine_distance).mean()
    # print('test_0_distance_fine_mean:', test_0_fine_distance_mean)
    # print('test_2_distance_fine_mean:', test_2_fine_distance_mean)
    # test_0_fine_distances.append(test_0_fine_distance_mean)
    # test_2_fine_distances.append(test_2_fine_distance_mean)

# 绘图 开始

# import matplotlib.pyplot as plt
# plt.figure(dpi=300, figsize=(8, 6)), plt.plot(test_0_coarse_distances)
# plt.savefig('img/test_0_coarse_distances.png')
# plt.figure(dpi=300, figsize=(8, 6)), plt.plot(test_2_coarse_distances)
# plt.savefig('img/test_2_coarse_distances.png')
# plt.figure(dpi=300, figsize=(8, 6)), plt.plot(test_0_fine_distances)
# plt.savefig('img/test_0_fine_distances.png')
# plt.figure(dpi=300, figsize=(8, 6)), plt.plot(test_2_fine_distances)
# plt.savefig('img/test_2_fine_distances.png')
#
# # coarse
# plt.figure(dpi=300, figsize=(8, 6))
# plt.title('angles_0'), plt.plot(angles_0), plt.show()
# plt.savefig('img/angles_0.png')
#
# plt.figure(dpi=300, figsize=(8, 6))
# plt.title('angles_2'), plt.plot(angles_2), plt.show()
# plt.savefig('img/angles_2.png')
#
# # fine
# plt.figure(dpi=300, figsize=(8, 6))
# plt.title('angles_0_fine'), plt.plot(angles_0_fine), plt.show()
# plt.savefig('img/angles_0_fine.png')
#
# plt.figure(dpi=300, figsize=(8, 6))
# plt.title('angles_2_fine'), plt.plot(angles_2_fine), plt.show()
# plt.savefig('img/angles_2_fine.png')
#
# # 两个相机点云的平均距离误差，粗配准
# test_0_coarse_distances = np.array(test_0_coarse_distances)
# test_2_coarse_distances = np.array(test_2_coarse_distances)
# print('test_0_coarse_distances.mean():', test_0_coarse_distances.mean().round(3)*10)
# print('test_0_coarse_distances.std() :', test_0_coarse_distances.std().round(3)*10)
# print('test_0_coarse_distances.min() :', test_0_coarse_distances.min().round(3)*10)
# print('test_0_coarse_distances.max() :', test_0_coarse_distances.max().round(3)*10)
#
# print('test_2_coarse_distances.mean():', test_2_coarse_distances.mean().round(3)*10)
# print('test_2_coarse_distances.std() :', test_2_coarse_distances.std().round(3)*10)
# print('test_2_coarse_distances.min() :', test_2_coarse_distances.min().round(3)*10)
# print('test_2_coarse_distances.max() :', test_2_coarse_distances.max().round(3)*10)
# coarse_distances = np.hstack((test_0_coarse_distances, test_2_coarse_distances))
# # 输出相机0和2的粗配准结果
# print('输出相机0和2的粗配准结果:', coarse_distances.min().round(3)*10, coarse_distances.max().round(3)*10,
#       coarse_distances.mean().round(3)*10, coarse_distances.std().round(3)*10)
#
# # 两个相机点云的平均距离误差，精配准
# test_0_fine_distances = np.array(test_0_fine_distances)
# test_2_fine_distances = np.array(test_2_fine_distances)
# print('test_0_fine_distances.mean():', test_0_fine_distances.mean().round(3)*10)
# print('test_0_fine_distances.std() :', test_0_fine_distances.std().round(3)*10)
# print('test_0_fine_distances.min() :', test_0_fine_distances.min().round(3)*10)
# print('test_0_fine_distances.max() :', test_0_fine_distances.max().round(3)*10)
#
# print('test_2_fine_distances.mean():', test_2_fine_distances.mean().round(3)*10)
# print('test_2_fine_distances.std() :', test_2_fine_distances.std().round(3)*10)
# print('test_2_fine_distances.min() :', test_2_fine_distances.min().round(3)*10)
# print('test_2_fine_distances.max() :', test_2_fine_distances.max().round(3)*10)
# fine_distances = np.hstack((test_0_fine_distances, test_2_fine_distances))
# # 输出相机0和2的粗配准结果
# print('输出相机0和2的粗配准结果:', fine_distances.min().round(3)*10, fine_distances.max().round(3)*10,
#       fine_distances.mean().round(3)*10, fine_distances.std().round(3)*10)
#
#
# # 点云粗配准后 角度误差
# angles_0_coarse = np.rad2deg(angles_0)
# angles_2_coarse = np.rad2deg(angles_2)
# print('angles_0_coarse.mean(axis=0):', angles_0_coarse.mean(axis=0).round(2))
# print('angles_0_coarse.std(axis=0) :', angles_0_coarse.std(axis=0).round(2))
# print('angles_0_coarse.min(axis=0) :', angles_0_coarse.min(axis=0).round(2))
# print('angles_0_coarse.max(axis=0) :', angles_0_coarse.max(axis=0).round(2))
#
# print('angles_2_coarse.mean(axis=0):', angles_2_coarse.mean(axis=0).round(2))
# print('angles_2_coarse.std(axis=0) :', angles_2_coarse.std(axis=0).round(2))
# print('angles_2_coarse.min(axis=0) :', angles_2_coarse.min(axis=0).round(2))
# print('angles_2_coarse.max(axis=0) :', angles_2_coarse.max(axis=0).round(2))
#
# # 点云精配准后 角度误差
# angles_0_fine = np.array(angles_0_fine)
# angles_2_fine = np.array(angles_2_fine)
# print('angles_0_fine.mean(axis=0):', angles_0_fine.mean(axis=0).round(2))
# print('angles_0_fine.std(axis=0) :', angles_0_fine.std(axis=0).round(2))
# print('angles_0_fine.min(axis=0) :', angles_0_fine.min(axis=0).round(2))
# print('angles_0_fine.max(axis=0) :', angles_0_fine.max(axis=0).round(2))
#
# print('angles_2_fine.mean(axis=0):', angles_2_fine.mean(axis=0).round(2))
# print('angles_2_fine.std(axis=0) :', angles_2_fine.std(axis=0).round(2))
# print('angles_2_fine.min(axis=0) :', angles_2_fine.min(axis=0).round(2))
# print('angles_2_fine.max(axis=0) :', angles_2_fine.max(axis=0).round(2))

# 绘图 结束


# # 获取多视角面部landmark并绘制到图像上
# landmarks0 = mediapipe_get_landmarks(color_0)
# landmarks1 = mediapipe_get_landmarks(c olor_1)
# landmarks2 = mediapipe_get_landmarks(color_2)
# color_0_show = mediapipe_features_draw(color_0, landmarks_mp=landmarks0)
# color_1_show = mediapipe_features_draw(color_1, landmarks_mp=landmarks1)
# color_2_show = mediapipe_features_draw(color_2, landmarks_mp=landmarks2)
# cv2.imshow('color_0_show', color_0_show)
# cv2.imshow('color_1_show', color_1_show)
# cv2.imshow('color_2_show', color_2_show)
#
# cv2.waitKey(0)

# # 通过彩色图像获取多视角面部mask
# face_mask0, masked_image1 = get_mask_from_json(image_0_path, json_0_path)
# face_mask1, masked_image2 = get_mask_from_json(image_1_path, json_1_path)
# face_mask2, masked_image0 = get_mask_from_json(image_2_path, json_2_path)
#
# # leftface_mask = get_leftface_mask(landmarks2)
# # rightface_mask = get_rightface_mask(landmarks0)
# # centerface_mask = get_centerface_mask(landmarks1)
#
# face_points_0 = [70, 46, 225, 224, 223, 222, 221, 189, 245, 188, 174, 236, 134, 220, 237, 241, 242, 97, 167, 37, 72,
#                    38, 82, 87, 86, 85, 84, 83, 201, 208, 171, 140, 170, 169, 210, 214, 192, 213, 147, 123, 116, 143,
#                    156]
# face_points_2 = [300, 276, 445, 444, 443, 442, 441, 413, 465, 412, 399, 456, 363, 440, 457, 461, 462, 326, 393, 267,
#                     302, 268, 312, 317, 316, 315, 314, 313, 421, 428, 396, 369, 395, 394, 430, 434, 416, 433, 376, 352,
#                     345, 372, 383]
#
# # 面部区域图像，通过标注的json文件
# img_face_region_0 = cv2.add(color_0, np.zeros(np.shape(color_0), dtype=np.uint8), mask=face_mask0)
# img_face_region_1 = cv2.add(color_1, np.zeros(np.shape(color_1), dtype=np.uint8), mask=face_mask1)
# img_face_region_2 = cv2.add(color_2, np.zeros(np.shape(color_2), dtype=np.uint8), mask=face_mask2)
#
# # 通过特征点框选面部区域
# circle_face_0 = img_circle_points(img_face_region_0, landmarks_mp=landmarks0[face_points_0, :])
# circle_face_2 = img_circle_points(img_face_region_2, landmarks_mp=landmarks2[face_points_2, :])
# circle_face_1_0 = img_circle_points(img_face_region_1, landmarks_mp=landmarks1[face_points_0, :])
# circle_face_1_2 = img_circle_points(img_face_region_1, landmarks_mp=landmarks1[face_points_2, :])
#
# # cv2.imshow('circle_face_0', circle_face_0)
# # cv2.imshow('circle_face_1_0', circle_face_1_0)
# # cv2.imshow('circle_face_2', circle_face_2)
# # cv2.imshow('circle_face_1_2', circle_face_1_2)
# # cv2.waitKey(0)
#
# # 获取部分面部区域点，用于计算视角之间相对位置和姿态
# points0 = get_camera_face_pts_mp_left(xyz0, landmarks0)
# points1_0 = get_camera_face_pts_mp_left(xyz1, landmarks1)
# points2 = get_camera_face_pts_mp_right(xyz2, landmarks2)
# points1_2 = get_camera_face_pts_mp_right(xyz1, landmarks1)
#
# # print('points0', points0)
# # print('points1_0', points1_0)
# # print('points2', points2)
# # print('points1_2', points1_2)
#
# # 三个视角相对姿态进行求解
# matR_0, vecT_0 = Face_ICP(points0, points1_0)
# matR_2, vecT_2 = Face_ICP(points2, points1_2)
# print('matR_0', rotationMatrixToEulerAngles(matR_0), matR_0)
# print('matR_2', rotationMatrixToEulerAngles(matR_2), matR_2)
#
# rgb_0 = cv2.cvtColor(color_0, cv2.COLOR_BGR2RGB).reshape(-1, 3)
# rgb_1 = cv2.cvtColor(color_1, cv2.COLOR_BGR2RGB).reshape(-1, 3)
# rgb_2 = cv2.cvtColor(color_2, cv2.COLOR_BGR2RGB).reshape(-1, 3)
#
# xyz0 = xyz0.reshape(-1, 3)
# xyz1 = xyz1.reshape(-1, 3)
# xyz2 = xyz2.reshape(-1, 3)
#
# face_mask0 = face_mask0.reshape(-1, 3)
# face_mask1 = face_mask1.reshape(-1, 3)
# face_mask2 = face_mask2.reshape(-1, 3)
#
# xyz0 = np.matmul(matR_0, (xyz0).T).T + vecT_0
# xyz2 = np.matmul(matR_2, (xyz2).T).T + vecT_2
#
# # 删除坐标值为0的点
# index = np.where(img_face_region_0.reshape(-1, 3)[:, 0] == 0)
# colors_0 = np.delete(rgb_0, index, axis=0)
# xyz_0 = np.delete(xyz0, index, axis=0)
# # img_face_region_0.reshape(-1,3)[index].shape
#
# index = np.where(img_face_region_1.reshape(-1, 3)[:, 0] == 0)
# colors_1 = np.delete(rgb_1, index, axis=0)
# xyz_1 = np.delete(xyz1, index, axis=0)
#
# index = np.where(img_face_region_2.reshape(-1, 3)[:, 0] == 0)
# colors_2 = np.delete(rgb_2, index, axis=0)
# xyz_2 = np.delete(xyz2, index, axis=0)
#
#
# # colors = np.vstack((colors_0, colors_1, colors_2))
# # points = np.vstack((xyz_0, xyz_1, xyz_2))
#
# # 获取鼻尖点三维空间坐标
# nose_coordinate = xyz1.reshape(720,1280,3)[landmarks1[4][1],landmarks1[4][0]]
# # mediapipe鼻子上点和鼻子下点的特征点编号：98, 327;;235,455
# nose_98 = xyz1.reshape(720,1280,3)[landmarks1[98][1],landmarks1[98][0]]
# nose_327 = xyz1.reshape(720,1280,3)[landmarks1[327][1],landmarks1[327][0]]
#
#
# print('nose_98', nose_98)
# print('nose_327', nose_327)
# nose_98_327_center = (nose_98 + nose_327) / 2
# nose_base_tmp = (nose_327[0] - nose_98[0]) / 4.0
# # 通过通用三维人脸模型获取
# nose_98_327_center_base = np.array([0, 1.4, 6.16])
# # 鼻子上点和鼻子下点的均值相对于原点为nose_98_327_center_base * nose_base_tmp，下面计算计算结果就是当前面部中，坐标原点所处的位置
# coordinate_origin = nose_98_327_center - nose_98_327_center_base * nose_base_tmp
#
#
# # 这里的points是进行坐标变换后的
# # points = points - nose_coordinate
# xyz_0 = xyz_0 - coordinate_origin
# xyz_1 = xyz_1 - coordinate_origin
# xyz_2 = xyz_2 - coordinate_origin
#
# # 点云0和点云2只保留z轴在-10cm到10cm区间范围内的数据
# colors_0 = colors_0[xyz_0[:,2] < 10, :]
# xyz_0 = xyz_0[xyz_0[:,2] < 10, :]
# colors_0 = colors_0[xyz_0[:,2] > -10, :]
# xyz_0 = xyz_0[xyz_0[:,2] > -10, :]
#
# colors_2 = colors_2[xyz_2[:,2] < 10, :]
# xyz_2 = xyz_2[xyz_2[:,2] < 10, :]
# colors_2 = colors_2[xyz_2[:,2] > -10, :]
# xyz_2 = xyz_2[xyz_2[:,2] > -10, :]
#
# # 点云1只保留z轴在-2cm到10cm区间范围内的数据
# colors_1 = colors_1[xyz_1[:,2] < 10, :]
# xyz_1 = xyz_1[xyz_1[:,2] < 10, :]
# colors_1 = colors_1[xyz_1[:,2] > -2, :]
# xyz_1 = xyz_1[xyz_1[:,2] > -2, :]
#
# # 设置各个视角点云的颜色
# # colors_0[:,:] = [255, 0, 0]
# # colors_1[:,:] = [0, 255, 0]
# # colors_2[:,:] = [0, 0, 255]
#
#
#
# point_cloud_0 = open3d.geometry.PointCloud()
# point_cloud_0.points = open3d.utility.Vector3dVector(xyz_0)
# point_cloud_0.colors = open3d.utility.Vector3dVector(colors_0 / 255)
# open3d.io.write_point_cloud('0.ply', point_cloud_0)
#
# point_cloud_1 = open3d.geometry.PointCloud()
# point_cloud_1.points = open3d.utility.Vector3dVector(xyz_1)
# point_cloud_1.colors = open3d.utility.Vector3dVector(colors_1 / 255)
# open3d.io.write_point_cloud('1.ply', point_cloud_1)
#
# point_cloud_2 = open3d.geometry.PointCloud()
# point_cloud_2.points = open3d.utility.Vector3dVector(xyz_2)
# point_cloud_2.colors = open3d.utility.Vector3dVector(colors_2 / 255)
# open3d.io.write_point_cloud('2.ply', point_cloud_2)
#
# # 删除在规定面部区域外的点
#
# # Homo_0 = np.array([[1.00000,0.00001,0.00286,-0.05172],
# #                     [0.00006,0.99973,-0.02320,0.16416],
# #                     [-0.00286,0.02320,0.99973,-0.02102],
# #                     [0.00000,0.00000,0.00000,1.00000]])
# #
# # Homo_2 = np.array([[0.99999,-0.00350,-0.00336,0.02973],
# # [0.00353,0.99995,0.00924,-0.05528],
# # [0.00333,-0.00925,0.99995,-0.02272],
# # [0.00000,0.00000,0.00000,1.00000]])
#
# import json
# labelme_json_0 = json.load(open(json_0_path, encoding='utf-8'))
# labelme_json_2 = json.load(open(json_2_path, encoding='utf-8'))
# points_tr_0 = np.round(labelme_json_0['shapes'][1]['points']).astype(np.uint64)
# points_tr_2 = np.round(labelme_json_2['shapes'][1]['points']).astype(np.uint64)
# print('points_tr_0', points_tr_0)
# print('points_tr_2', points_tr_2)
#
#
# src_tr_al = np.zeros([4,3])
# dst_tr_al = np.array([[-72643.3, 11413.3, 11140.1],
#                         [-16981.4, -3046.09, 111901],
#                         [71894.4, 11245.4, 11118.2],
#                         [16732.3, -2973.56, 111813]])/10000
#
# xyz0 = xyz0 - coordinate_origin
# xyz0 = np.matmul(Homo_0[0:3,0:3], xyz0.T).T + Homo_0[0:3,3]
#
# xyz0 = xyz0.reshape(720, 1280, 3)
# # aa = xyz0[landmarks0[234,1],landmarks0[234,0],:]
# aa = xyz0[points_tr_0[0,1],points_tr_0[0,0],:]
# bb = xyz0[landmarks0[102,1],landmarks0[102,0],:]
# src_tr_al[0, :] = aa
# src_tr_al[1, :] = bb
#
# cc = aa - bb
# angle = np.arctan(cc[1]/np.abs(cc[2]))
# print('角度是：', angle, angle*57.3)
#
# xyz2 = xyz2 - coordinate_origin
# xyz2 = np.matmul(Homo_2[0:3,0:3], xyz2.T).T + Homo_2[0:3,3]
#
# xyz2 = xyz2.reshape(720, 1280, 3)
# aa = xyz2[points_tr_2[0,1],points_tr_2[0,0],:]
# # aa = xyz2[landmarks2[454,1],landmarks2[454,0],:]
# bb = xyz2[landmarks2[331,1],landmarks2[331,0],:]
# src_tr_al[2,:] = aa
# src_tr_al[3,:] = bb
#
# cc = aa - bb
# angle2 = np.arctan(cc[1]/np.abs(cc[2]))
# print('角度是2：', angle2, angle2*57.3)
#
# print('src_tr_al:', src_tr_al)
# print('dst_tr_al:', dst_tr_al)
# matR_tr, vecT_tr = Face_ICP(src_tr_al, dst_tr_al)
# # print(matR_0, vecT_0)
# print('matR_tr', rotationMatrixToEulerAngles(matR_tr), rotationMatrixToEulerAngles(matR_tr)*180/np.pi)
#
# lines = [[0, 1], [1, 3], [3, 2], [2,0]] #连接的顺序，封闭链接
# color = [[1, 0, 0] for i in range(len(lines))]  # 红色
#
# lines_pcd = open3d.geometry.LineSet()
# lines_pcd.lines = open3d.utility.Vector2iVector(lines)
# lines_pcd.colors = open3d.utility.Vector3dVector(color) #线条颜色
# lines_pcd.points = open3d.utility.Vector3dVector(src_tr_al)
#
# xyz_0 = np.matmul(Homo_0[0:3,0:3], xyz_0.T).T + Homo_0[0:3,3]
# xyz_2 = np.matmul(Homo_2[0:3,0:3], xyz_2.T).T + Homo_2[0:3,3]
#
# # 耳屏点 234, 454
# # 鼻翼点 (129,102) (358,331)
#
#
# # 把多个视角的点云进行合并
# colors = np.vstack((colors_0, colors_1, colors_2))
# points = np.vstack((xyz_0, xyz_1, xyz_2))
# # points = np.vstack((xyz0.reshape(-1,3), xyz1, xyz2.reshape(-1,3)))
# # points = np.matmul(matR_tr, points.T).T + vecT_tr
#
# point_cloud = open3d.geometry.PointCloud()
# point_cloud.points = open3d.utility.Vector3dVector(points)
# point_cloud.colors = open3d.utility.Vector3dVector(colors / 255)
# # point_cloud.paint_uniform_color([0,1,0])
# # 坐标轴size单位为厘米，红、绿、蓝分别代表x,y,z。
# axis_1 = open3d.geometry.TriangleMesh.create_coordinate_frame(size=6, origin=[0, 0, 0])
# open3d.visualization.draw_geometries([point_cloud, axis_1, lines_pcd], window_name='Open3D', width=1536, height=864, left=50, top=50)
#
# # 左侧视图
# open3d.visualization.draw_geometries([point_cloud, axis_1], window_name='Open3D',
#                                   zoom=0.8,
#                                   front=[-1, 0, 0],
#                                   # lookat=[0, 0, 6],
#                                   lookat=[0,0,0],
#                                   up=[0, 1, 0]
#                                   )
#
# # 右侧视图
# open3d.visualization.draw_geometries([point_cloud, axis_1], window_name='Open3D',
#                                   zoom=0.8,
#                                   front=[1, 0, 0],
#                                   # lookat=[0, 0, 6],
#                                   lookat=[0,0,0],
#                                   up=[0, 1, 0]
#                                   )
#
# # 正视图
# open3d.visualization.draw_geometries([point_cloud, axis_1], window_name='Open3D',
#                                   zoom=0.8,
#                                   front=[0, 0, 1],
#                                   # lookat=[0, 0, 6],
#                                   lookat=[0,0,0],
#                                   up=[0, 1, 0]
#                                   )
#
# # 0号相机用于匹配的数据和1号相机用于和0号相机进行匹配的数据
# mask_match_0 = get_mask_from_points(landmarks0, face_points_0)
# mask_match_1_0 = get_mask_from_points(landmarks1, face_points_0)
#
# # cv2.imshow('mask_match_0', mask_match_0)
# # cv2.imshow('mask_match_1_0', mask_match_1_0)
# # cv2.waitKey(0)
#
# # points_match_0 = depth2xyz(depth_0, camera_0_depth_intrinsics, flatten=False, depth_scale=10)
# # points_left = points_match_0.reshape(-1, 3)
# index = np.where(mask_match_0.reshape(-1, 1)[:, 0] == 0)
# colors_match_0 = np.delete(rgb_0, index, axis=0)
# points_match_0 = np.delete(points_0, index, axis=0)
# index = np.where(mask_match_1_0.reshape(-1, 1)[:, 0] == 0)
# colors_match_1_0 = np.delete(rgb_1, index, axis=0)
# points_match_1_0 = np.delete(points_1, index, axis=0)
#
# # 如果没有进行全面的深度图孔洞填充，选取的区域会有部分点云为[0,0,0]，需要把这部分点删除掉，否则影响后面的计算和显示。
# index = np.where(points_match_0[:, 2] == 0)
# colors_match_0 = np.delete(colors_match_0, index, axis=0)
# points_match_0 = np.delete(points_match_0, index, axis=0)
# index = np.where(points_match_1_0[:, :2] == 0)
# colors_match_1_0 = np.delete(colors_match_1_0, index, axis=0)
# points_match_1_0 = np.delete(points_match_1_0, index, axis=0)
#
# # 对选取的0号点云进行旋转和平移
# points_match_0 = np.matmul(matR_0, points_match_0.T).T + vecT_0
#
# points_match = np.vstack((points_match_0, points_match_1_0))
# points_match = points_match - points_match.mean(axis=0)
# colors_match_0[:,:] = [255,0,0]
# colors_match_1_0[:,:] = [0,255,0]
# colors_match = np.vstack((colors_match_0, colors_match_1_0))
#
# point_cloud = open3d.geometry.PointCloud()
# point_cloud.points = open3d.utility.Vector3dVector(points_match)
# point_cloud.colors = open3d.utility.Vector3dVector(colors_match / 255)
# # point_cloud.paint_uniform_color([0, 1, 0])
# axis_1 = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
# open3d.visualization.draw_geometries([point_cloud, axis_1], window_name='Open3D', width=1536, height=864, left=50, top=50)
#
