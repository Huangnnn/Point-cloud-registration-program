import os
import numpy as np
import cv2
import math
import json
import open3d
import matplotlib.pyplot as plt

# camera 0 depth intrinsics: [ 1280x720  p[636.849 370.579]  f[900.584 900.584]  Brown Conrady [0 0 0 0 0] ]
# camera 0 color intrinsics: [ 1280x720  p[610.604 380.316]  f[920.925 920.796]  Inverse Brown Conrady [0 0 0 0 0] ]
# camera 1 depth intrinsics: [ 1280x720  p[632.04 379.481]  f[897.938 897.938]  Brown Conrady [0 0 0 0 0] ]
# camera 1 color intrinsics: [ 1280x720  p[627.635 374.616]  f[917.981 917.144]  Inverse Brown Conrady [0 0 0 0 0] ]
# camera 2 depth intrinsics: [ 1280x720  p[648.907 365.319]  f[902.059 902.059]  Brown Conrady [0 0 0 0 0] ]
# camera 2 color intrinsics: [ 1280x720  p[648.011 363.285]  f[915.012 912.372]  Inverse Brown Conrady [0 0 0 0 0] ]



camera_0_color_intrinsics = np.array([[920.925, 0, 610.604],
                    [0, 920.796, 380.316],
                    [0, 0, 0]])

camera_0_depth_intrinsics = np.array([[900.584, 0,636.849],
                    [0, 900.584, 370.579],
                    [0, 0, 0]])

camera_1_color_intrinsics = np.array([[917.981, 0, 627.635],
                    [0, 917.144, 374.616],
                    [0, 0, 0]])

camera_1_depth_intrinsics = np.array([[897.938, 0,632.04],
                    [0, 897.938, 379.481],
                    [0, 0, 0]])

camera_2_color_intrinsics = np.array([[915.012, 0, 648.011],
                    [0, 912.372, 363.285],
                    [0, 0, 0]])

camera_2_depth_intrinsics = np.array([[902.059, 0,648.907],
                    [0, 902.059, 365.319],
                    [0, 0, 0]])


face_points_0 = [70, 46, 225, 224, 223, 222, 221, 189, 245, 188, 174, 236, 134, 220, 237, 241, 242, 97, 167, 37, 72,
                   38, 82, 87, 86, 85, 84, 83, 201, 208, 171, 140, 170, 169, 210, 214, 192, 213, 147, 123, 116, 143,
                   156]
face_points_2 = [300, 276, 445, 444, 443, 442, 441, 413, 465, 412, 399, 456, 363, 440, 457, 461, 462, 326, 393, 267,
                    302, 268, 312, 317, 316, 315, 314, 313, 421, 428, 396, 369, 395, 394, 430, 434, 416, 433, 376, 352,
                    345, 372, 383]

def mediapipe_get_landmarks(rgb):
    import mediapipe as mp
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        rgb.flags.writeable = False

        results = face_mesh.process(rgb)
        # print(results)

    # landmark 个数为478
    # 水平方向为x轴，垂直方向为y轴
    # 0-467为face mesh 对应的468个特征点
    # 468-477为了两个黑色眼球区域的10个点，iris

    # 获取landmarks，数字范围为0-1, 如果是想输出0-1的mediapipe 3d点，直接return landmarks_mp_3d即可。
    # landmarks_mp_3d = np.array(results.multi_face_landmarks[0].landmark)
    landmarks_mp_3d = results.multi_face_landmarks[0].landmark

    landmarks_mp = np.zeros([478, 2], dtype=np.int32)

    for idx in range(478):
        x = np.round(landmarks_mp_3d[idx].x * 1280).astype(np.int32)
        y = np.round(landmarks_mp_3d[idx].y * 720).astype(np.int32)
        landmarks_mp[idx, 0] = x  # 对应列
        landmarks_mp[idx, 1] = y  # 对应行
        # landmarks_mp[idx, 0] = y
        # landmarks_mp[idx, 1] = x

    # print(landmarks_mp.shape, landmarks_mp)

    return landmarks_mp

# 放大原始图片并显示mediapipe face_mesh 特征点和标号
def mediapipe_features_draw(img, landmarks_mp, scale=1):
    point_size = 1
    point_color = (0, 255, 0)  # BGR
    thickness = 0  # 可以为 0、4、8

    img_output = cv2.resize(img, (1280 * scale, 720 * scale))
    for idx,landmark in enumerate(landmarks_mp):
        # print('landmark', idx, ':', landmark[0], landmark[1])
        x = landmark[0] * scale
        y = landmark[1] * scale
        cv2.circle(img_output, (x, y), point_size, point_color, thickness)
        # 图片 添加的文字 位置 字体 字体大小 字体颜色 字体粗细
        # cv2.putText(img_output, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return img_output

# 放大原始图片并显示mediapipe face_mesh 特征点和标号
def img_circle_points(img, landmarks_mp, scale=1):
    point_size = 1
    point_color = (0, 255, 0)  # BGR
    thickness = 0  # 可以为 0、4、8

    img_output = cv2.resize(img, (1280 * scale, 720 * scale))
    for idx,landmark in enumerate(landmarks_mp):
        # print('landmark', idx, ':', landmark[0], landmark[1])
        x = landmark[0] * scale
        y = landmark[1] * scale
        cv2.circle(img_output, (x,y), point_size, point_color, thickness)
        # 图片 添加的文字 位置 字体 字体大小 字体颜色 字体粗细
        # cv2.putText(img_output, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return img_output

# depth_scale=10时，长度单位为厘米，元素深度数据是毫米，乘以10为厘米
def depth2xyz(depth_map, depth_cam_matrix, flatten=False, depth_scale=10):
    fx, fy = depth_cam_matrix[0, 0], depth_cam_matrix[1, 1]
    cx, cy = depth_cam_matrix[0, 2], depth_cam_matrix[1, 2]
    h, w = np.mgrid[0:depth_map.shape[0], 0:depth_map.shape[1]]
    z = depth_map / depth_scale
    x = (w - cx) * z / fx
    y = (h - cy) * z / fy
    xyz = np.dstack((x, y, z)) if flatten == False else np.dstack((x, y, z)).reshape(-1, 3)
    # xyz=cv2.rgbd.depthTo3d(depth_map,depth_cam_matrix)
    return xyz

def get_mask_from_json(img_path, json_path):
    # pass
    labelme_json = json.load(open(json_path, encoding='utf-8'))
    img = cv2.imread(img_path)

    mask = np.zeros((720, 1280, 1), dtype=np.uint8)
    mask2 = np.zeros((720, 1280, 3), dtype=np.uint8)
    points = labelme_json['shapes'][0]['points']
    points = np.array(points)
    points = points.reshape(-1, 1, 2)
    points = points.astype(np.int32)

    cv2.fillConvexPoly(mask, points, (255,))
    cv2.fillConvexPoly(mask2, points, (255,255,255))
    # print('points', points)
    # scale = 2
    # mask2 = cv2.resize(mask2, (1280 * scale, 720 * scale))
    # for idx,point in enumerate(points):
    #     point = point * scale
    #     cv2.circle(mask2, point[0], 1, [255,0,0])
    #     cv2.putText(mask2, str(idx), point[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    # cv2.imshow('mask2', mask2)
    # 显示mask
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)

    # masked_image = cv2.bitwise_and(img, mask2)
    # 显示mask区域图像
    # cv2.imshow('new_image', masked_image)
    # cv2.waitKey(0)
    # return mask, masked_image
    return mask, mask

def get_mask_from_points(landmarks_mp, points_idxs):
    mask = np.full((720,1280,1), 0, dtype=np.uint8)

    # cv2.fillConvexPoly(mask, cv2.convexHull((landmarks_mp[[10, 109, 67, 103, 54, 21, 162, 127, 93, 234, 132, 58, 172, 136,
    #                                                 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 367, 288, 435, 361,
    #                                                             401, 323, 366, 454, 356, 389, 251, 284, 332, 297, 338, 10]])), (255,))
    cv2.fillConvexPoly(mask, cv2.convexHull((landmarks_mp[points_idxs])), (255,))
    return mask

def get_leftface_mask(landmarks_mp):
    leftface_mask = np.zeros(shape=[720, 1280], dtype=np.uint8)
    points_list = []
    # 右颐
    # for index in [206,205,50,123,213,135,214,216]:
    for index in [338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 175, 199, 200,
               18, 313, 406, 335, 273, 287, 410,
               322, 426, 266, 329, 277, 343, 412, 351, 168, 8, 9, 151, 10]:
    # for index in [36, 50, 187, 192, 216, 206]:
        # for index in [36, 50, 187, 207, 216, 206]:
        points_list.append(landmarks_mp[index])

    # 绘制填充的多边形
    points_list = np.array(points_list).reshape(-1, 1, 2)
    cv2.fillPoly(leftface_mask, [points_list], (255))

    return leftface_mask

def get_rightface_mask(landmarks_mp):
    rightface_mask = np.zeros(shape=[720, 1280], dtype=np.uint8)
    points_list = []
    # 右颐
    # for index in [206,205,50,123,213,135,214,216]:
    for index in [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 175, 199,
                  200, 18, 83, 182, 106, 43, 57, 92, 165, 203, 36, 100, 47, 114, 188, 122, 6,
                  168, 8, 9, 151]:
    # for index in [36, 50, 187, 192, 216, 206]:
        # for index in [36, 50, 187, 207, 216, 206]:
        points_list.append(landmarks_mp[index])

    # 绘制填充的多边形
    points_list = np.array(points_list).reshape(-1, 1, 2)
    cv2.fillPoly(rightface_mask, [points_list], (255))

    return rightface_mask

def get_centerface_mask(landmarks_mp):
    centerface_mask = np.zeros(shape=[720, 1280], dtype=np.uint8)
    points_list = []
    # 右颐
    # for index in [206,205,50,123,213,135,214,216]:
    for index in [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                    176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]:
        # for index in [36, 50, 187, 192, 216, 206]:
        # for index in [36, 50, 187, 207, 216, 206]:
        points_list.append(landmarks_mp[index])

    # 绘制填充的多边形
    points_list = np.array(points_list).reshape(-1, 1, 2)
    cv2.fillPoly(centerface_mask, [points_list], (255))

    return centerface_mask

# 通过相机获取面部14个用于姿态估计的3维特征点，特征点检测方法采用mediapipe
def get_camera_face_pts_10_mp(points, landmarks_mp):
    image_pts = np.zeros([10, 2], dtype=np.int32)
    for pts_idx, idx in enumerate([107,336,55,285,133,362,263,33,57,287]):

        image_pts[pts_idx] = landmarks_mp[idx]

    camera_face_pts_14 = np.zeros([10, 3])
    for idx, point in enumerate(image_pts):
        camera_face_pts_14[idx, :] = points[point[1], point[0]]

    return camera_face_pts_14

# 通过相机获取面部14个用于姿态估计的3维特征点，特征点检测方法采用mediapipe
def get_camera_face_pts_mp_left(points, landmarks_mp):
    image_pts = np.zeros([43, 2], dtype=np.int32)
    for pts_idx, idx in enumerate([70,46,225,224,223,222,221,189,245,188,174,236,134,220,237,241,242,97,167,37,72,38,82,
                                   87,86,85,84,83,201,208,171,140,170,169,210,214,192,213,147,123,116,143,156]):

        image_pts[pts_idx] = landmarks_mp[idx]

    camera_face_pts_14 = np.zeros([43, 3])
    for idx, point in enumerate(image_pts):
        camera_face_pts_14[idx, :] = points[point[1], point[0]]

    return camera_face_pts_14

# 通过相机获取面部14个用于姿态估计的3维特征点，特征点检测方法采用mediapipe
def get_camera_face_pts_mp_right(points, landmarks_mp):
    image_pts = np.zeros([43, 2], dtype=np.int32)
    for pts_idx, idx in enumerate([300,276,445,444,443,442,441,413,465,412,399,456,363,440,457,461,462,326,393,267,302,
                                   268,312,317,316,315,314,313,421,428,396,369,395,394,430,434,416,433,376,352,345,372,383]):

        image_pts[pts_idx] = landmarks_mp[idx]

    camera_face_pts_14 = np.zeros([43, 3])
    for idx, point in enumerate(image_pts):
        camera_face_pts_14[idx, :] = points[point[1], point[0]]

    return camera_face_pts_14

# 姿态估计
def Face_ICP(points_src, points_dst):

    points1_mean = points_src.mean(axis=0)
    points2_mean = points_dst.mean(axis=0)

    srcDat = points_src - points1_mean
    dstDat = points_dst - points2_mean
    # print(srcDat)
    # print(dstDat)

    matS = srcDat.T.dot(dstDat)

    w, u, v = cv2.SVDecomp(matS)

    matTemp = u.dot(v)

    det = cv2.determinant(matTemp)

    matM = np.eye(3, dtype=np.float64)
    matM[2,2] = det

    matR = v.T.dot(matM).dot(u.T)

    vecT = points2_mean - np.matmul(matR, (points1_mean).T).T

    # print('matR:', matR)
    # print('vecT', vecT)
    return matR, vecT

# 进行旋转矩阵和欧拉角的转换
def isRotationMatrix(R) :
  Rt = np.transpose(R)
  shouldBeIdentity = np.dot(Rt, R)
  I = np.identity(3, dtype = R.dtype)
  n = np.linalg.norm(I - shouldBeIdentity)
  return n < 1e-6

def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

def ICP_3D_3D(user_path, _source_0, _source_2, _target):
    obj_dir = os.path.basename(user_path)
    analysis_dir = os.path.join(os.path.dirname(user_path), 'analysis')
    user_dir = os.path.join(analysis_dir, obj_dir)
    dir_path_0 = os.path.join(user_dir, 'np_files_0_1')
    dir_path_2 = os.path.join(user_dir, 'np_files_2_1')

    if not os.path.exists(analysis_dir):
        os.mkdir(analysis_dir)

    if not os.path.exists(user_dir):
        os.mkdir(user_dir)

    if not os.path.exists(dir_path_0):
        os.mkdir(dir_path_0)

    if not os.path.exists(dir_path_2):
        os.mkdir(dir_path_2)


    # 大量计算
    norms_0 = np.zeros(_source_0.shape[0], dtype=np.float64)
    min_idxs_0 = np.zeros(_source_0.shape[0], dtype=np.uint64)
    norms_2 = np.zeros(_source_2.shape[0], dtype=np.float64)
    min_idxs_2 = np.zeros(_source_2.shape[0], dtype=np.uint64)

    iter_num = 10
    matR_0 = np.zeros([20, 3, 3])
    vecT_0 = np.zeros([20, 3])
    Homo_mat_0 = np.zeros([20, 4, 4])
    angle_0 = np.zeros([20, 3])
    matR_2 = np.zeros([20, 3, 3])
    vecT_2 = np.zeros([20, 3])
    Homo_mat_2 = np.zeros([20, 4, 4])
    angle_2 = np.zeros([20, 3])
    Homo_0 = np.eye(4)
    Homo_2 = np.eye(4)

    for iter_index in range(iter_num):
        # print('iter_index:%2g' % iter_index)
        norms_path_0 = os.path.join(dir_path_0, 'norms_' + str(iter_index) + '.npy')
        min_idxs_path_0 = os.path.join(dir_path_0, 'min_idxs_' + str(iter_index) + '.npy')
        norms_path_2 = os.path.join(dir_path_2, 'norms_' + str(iter_index) + '.npy')
        min_idxs_path_2 = os.path.join(dir_path_2, 'min_idxs_' + str(iter_index) + '.npy')
        if os.path.exists(min_idxs_path_0):
            # print('\tTrue', end='')
            # 通过读取文件获取normsS
            norms_0 = np.load(norms_path_0)
            min_idxs_0 = np.load(min_idxs_path_0)
        else:
            for index in range(_source_0.shape[0]):
                _diff = _source_0[index, :] - _target
                norms_0[index] = np.linalg.norm(_diff, axis=1).min()
                min_idxs_0[index] = np.linalg.norm(_diff, axis=1).argmin()
                # print('_diff', _diff)

                # 保存文件
            np.save(norms_path_0, norms_0)
            np.save(min_idxs_path_0, min_idxs_0)

        if os.path.exists(min_idxs_path_2):
            # print('\tTrue', end='')
            # 通过读取文件获取normsS
            norms_2 = np.load(norms_path_2)
            min_idxs_2 = np.load(min_idxs_path_2)
        else:
            for index in range(_source_2.shape[0]):
                _diff = _source_2[index, :] - _target
                norms_2[index] = np.linalg.norm(_diff, axis=1).min()
                min_idxs_2[index] = np.linalg.norm(_diff, axis=1).argmin()
                # print('_diff', _diff)

                # 保存文件
            np.save(norms_path_2, norms_2)
            np.save(min_idxs_path_2, min_idxs_2)

        min_idxs_0 = min_idxs_0.astype(np.uint64)
        min_idxs_2 = min_idxs_2.astype(np.uint64)

        # 获取距离最小的35000个点的阈值
        _norms_0 = norms_0.copy()
        _norms_2 = norms_2.copy()
        _norms_0.sort()
        _norms_2.sort()
        if len(_norms_0) < 35000:
            threshold_0 = _norms_0[30000]
        else:
            threshold_0 = _norms_0[35000]
        if len(_norms_2) < 35000:
            threshold_2 = _norms_2[30000]
        else:
            threshold_2 = _norms_2[35000]
        # plt.plot(_norms_0)
        # plt.show()

        # 获取用于ICP的src点和dst点
        points_src_0 = _source_0[norms_0 < threshold_0]
        points_src_2 = _source_2[norms_2 < threshold_2]
        points_dst_0 = _target[min_idxs_0[norms_0 < threshold_0], :]
        points_dst_2 = _target[min_idxs_2[norms_2 < threshold_2], :]

        matR_0[iter_index, :, :], vecT_0[iter_index, :] = Face_ICP(points_src_0, points_dst_0)
        matR_2[iter_index, :, :], vecT_2[iter_index, :] = Face_ICP(points_src_2, points_dst_2)
        _source_0 = np.matmul(matR_0[iter_index, :, :], (_source_0).T).T + vecT_0[iter_index, :]
        _source_2 = np.matmul(matR_2[iter_index, :, :], (_source_2).T).T + vecT_2[iter_index, :]

        Homo_mat_0[iter_index, 0:3, 0:3] = matR_0[iter_index, :, :]
        Homo_mat_0[iter_index, 0:3, 3] = vecT_0[iter_index, :]
        Homo_mat_0[iter_index, 3, 3] = 1
        Homo_mat_2[iter_index, 0:3, 0:3] = matR_2[iter_index, :, :]
        Homo_mat_2[iter_index, 0:3, 3] = vecT_2[iter_index, :]
        Homo_mat_2[iter_index, 3, 3] = 1

        Homo_0 = np.matmul(Homo_mat_0[iter_index, :, :], Homo_0)
        Homo_2 = np.matmul(Homo_mat_2[iter_index, :, :], Homo_2)

        angle_0[iter_index, :] = rotationMatrixToEulerAngles(matR_0[iter_index, :, :])
        angle_2[iter_index, :] = rotationMatrixToEulerAngles(matR_2[iter_index, :, :])
        # print('matR_0:', matR_0[iter_index], 'vecT_0', vecT_0[iter_index])
        # print('matR_2:', matR_2[iter_index], 'vecT_2', vecT_2[iter_index])

    return Homo_0, Homo_2

angles = []
def hpe_all_main(user_path):
    image_0_path = user_path + '_color_0.png'
    depth_0_path = user_path + '_depth_0.png'
    image_1_path = user_path + '_color_1.png'
    depth_1_path = user_path + '_depth_1.png'
    image_2_path = user_path + '_color_2.png'
    depth_2_path = user_path + '_depth_2.png'

    json_0_path = user_path + '_color_0.json'
    json_1_path = user_path + '_color_1.json'
    json_2_path = user_path + '_color_2.json'

    # 读取图像
    color_0 = cv2.imdecode(np.fromfile(image_0_path, dtype=np.uint8), -1)
    depth_0 = cv2.imdecode(np.fromfile(depth_0_path, dtype=np.uint8), -1)
    color_1 = cv2.imdecode(np.fromfile(image_1_path, dtype=np.uint8), -1)
    depth_1 = cv2.imdecode(np.fromfile(depth_1_path, dtype=np.uint8), -1)
    color_2 = cv2.imdecode(np.fromfile(image_2_path, dtype=np.uint8), -1)
    depth_2 = cv2.imdecode(np.fromfile(depth_2_path, dtype=np.uint8), -1)

    # 深度图转点云
    xyz0 = cv2.rgbd.depthTo3d(depth_0, camera_0_depth_intrinsics)
    xyz1 = cv2.rgbd.depthTo3d(depth_1, camera_1_depth_intrinsics)
    xyz2 = cv2.rgbd.depthTo3d(depth_2, camera_2_depth_intrinsics)
    # 通过cv2的rgbd函数转点云，深度为0的点会转为nan，如果数据中包含nan在open3d中是无法显示的，可以把为nan的点转为[0,0,0]，或后面显示前删除
    # 乘以100 单位变为厘米
    xyz0 = np.nan_to_num(xyz0) * 100
    xyz1 = np.nan_to_num(xyz1) * 100
    xyz2 = np.nan_to_num(xyz2) * 100

    # 坐标系变换，x不变，y和z轴取反，在这里xyz的shape为m*n*3
    xyz0[:, :, 1:3] = -xyz0[:, :, 1:3]
    xyz1[:, :, 1:3] = -xyz1[:, :, 1:3]
    xyz2[:, :, 1:3] = -xyz2[:, :, 1:3]

    # 获取多视角面部landmark并绘制到图像上
    landmarks0 = mediapipe_get_landmarks(color_0)
    landmarks1 = mediapipe_get_landmarks(color_1)
    landmarks2 = mediapipe_get_landmarks(color_2)
    # color_0_show = mediapipe_features_draw(color_0, landmarks_mp=landmarks0)
    # color_1_show = mediapipe_features_draw(color_1, landmarks_mp=landmarks1)
    # color_2_show = mediapipe_features_draw(color_2, landmarks_mp=landmarks2)
    # cv2.imshow('color_0_show', color_0_show)
    # cv2.imshow('color_1_show', color_1_show)
    # cv2.imshow('color_2_show', color_2_show)

    # cv2.waitKey(1000)
    # print(landmarks2)
    # print(landmarks2.shape, landmarks2[477, :])
    # print(xyz2.shape)
    # print(xyz2[landmarks2[477,1], landmarks2[477,0], :])
    # print(xyz2[landmarks2[476,1], landmarks2[476,0], :])
    # print(xyz2[landmarks2[475,1], landmarks2[475,0], :])
    # print(xyz2[landmarks2[474,1], landmarks2[474,0], :])
    # print(xyz2[landmarks2[473,1], landmarks2[473,0], :])

    xyz2_tmp = np.zeros([478,3])
    # print(xyz2_tmp.shape)
    for idx,landmark2 in enumerate(landmarks2):
        xyz2_tmp[idx, :] = xyz2[landmarks2[idx,1], landmarks2[idx,0], :]

    # print(xyz2_tmp)

    # 通过彩色图像获取多视角面部mask
    face_mask0, masked_image0 = get_mask_from_json(image_0_path, json_0_path)
    face_mask1, masked_image1 = get_mask_from_json(image_1_path, json_1_path)
    face_mask2, masked_image2 = get_mask_from_json(image_2_path, json_2_path)

    # 面部区域图像，通过标注的json文件
    img_0_face_region = cv2.add(color_0, np.zeros(np.shape(color_0), dtype=np.uint8), mask=face_mask0)
    img_1_face_region = cv2.add(color_1, np.zeros(np.shape(color_1), dtype=np.uint8), mask=face_mask1)
    img_2_face_region = cv2.add(color_2, np.zeros(np.shape(color_2), dtype=np.uint8), mask=face_mask2)

    # cv2.imshow('face_mask2', face_mask2)
    # cv2.imshow('img_2_face_region', img_2_face_region)

    # 获取部分面部区域点，用于计算视角之间相对位置和姿态
    points0 = get_camera_face_pts_mp_left(xyz0, landmarks0)
    points1_0 = get_camera_face_pts_mp_left(xyz1, landmarks1)
    points2 = get_camera_face_pts_mp_right(xyz2, landmarks2)
    points1_2 = get_camera_face_pts_mp_right(xyz1, landmarks1)

    # 三个视角相对姿态进行求解
    matR_0, vecT_0 = Face_ICP(points0, points1_0)
    matR_2, vecT_2 = Face_ICP(points2, points1_2)
    # print('matR_0', matR_0, rotationMatrixToEulerAngles(matR_0))
    # print('matR_2', matR_2, rotationMatrixToEulerAngles(matR_2))

    # 三维空间位置和颜色进行reshape
    xyz0 = xyz0.reshape(-1, 3)
    xyz1 = xyz1.reshape(-1, 3)
    xyz2 = xyz2.reshape(-1, 3)

    rgb_0 = cv2.cvtColor(color_0, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    rgb_1 = cv2.cvtColor(color_1, cv2.COLOR_BGR2RGB).reshape(-1, 3)
    rgb_2 = cv2.cvtColor(color_2, cv2.COLOR_BGR2RGB).reshape(-1, 3)

    xyz0 = np.matmul(matR_0, (xyz0).T).T + vecT_0
    xyz2 = np.matmul(matR_2, (xyz2).T).T + vecT_2

    # 删除坐标值为0的点
    index = np.where(img_0_face_region.reshape(-1, 3)[:, 0] == 0)
    colors_0 = np.delete(rgb_0, index, axis=0)
    xyz_0 = np.delete(xyz0, index, axis=0)

    index = np.where(img_1_face_region.reshape(-1, 3)[:, 0] == 0)
    colors_1 = np.delete(rgb_1, index, axis=0)
    xyz_1 = np.delete(xyz1, index, axis=0)

    index = np.where(img_2_face_region.reshape(-1, 3)[:, 0] == 0)
    colors_2 = np.delete(rgb_2, index, axis=0)
    xyz_2 = np.delete(xyz2, index, axis=0)

    # # 显示粗配准的人脸
    # colors = np.vstack((colors_0, colors_1, colors_2))
    # points = np.vstack((xyz_0, xyz_1, xyz_2))
    # point_cloud = open3d.geometry.PointCloud()
    # point_cloud.points = open3d.utility.Vector3dVector(xyz_2)
    # point_cloud.colors = open3d.utility.Vector3dVector(colors_2 / 255)
    # # point_cloud.paint_uniform_color([0,1,0])
    # # 坐标轴size单位为厘米，红、绿、蓝分别代表x,y,z。
    # axis_1 = open3d.geometry.TriangleMesh.create_coordinate_frame(size=6, origin=[0, 0, 0])
    # open3d.visualization.draw_geometries([point_cloud, axis_1], window_name='Open3D')


    # mediapipe鼻子上点和鼻子下点的特征点编号：98, 327;;235,455
    nose_98 = xyz1.reshape(720, 1280, 3)[landmarks1[98][1], landmarks1[98][0]]
    nose_327 = xyz1.reshape(720, 1280, 3)[landmarks1[327][1], landmarks1[327][0]]
    # print('nose_98', nose_98)
    # print('nose_327', nose_327)
    nose_98_327_center = (nose_98 + nose_327) / 2
    nose_base_tmp = (nose_327[0] - nose_98[0]) / 4.0
    # 通过通用三维人脸模型获取
    nose_98_327_center_base = np.array([0, 1.4, 6.16])
    # 鼻子上点和鼻子下点的均值相对于原点为nose_98_327_center_base * nose_base_tmp，下面计算计算结果就是当前面部中，坐标原点所处的位置
    coordinate_origin = nose_98_327_center - nose_98_327_center_base * nose_base_tmp

    # 这里的points是进行坐标变换后的
    xyz_0 = xyz_0 - coordinate_origin
    xyz_1 = xyz_1 - coordinate_origin
    xyz_2 = xyz_2 - coordinate_origin

    # 点云0和点云2只保留z轴在-10cm到10cm区间范围内的数据
    colors_0 = colors_0[xyz_0[:, 2] < 10, :]
    xyz_0 = xyz_0[xyz_0[:, 2] < 10, :]
    colors_0 = colors_0[xyz_0[:, 2] > -10, :]
    xyz_0 = xyz_0[xyz_0[:, 2] > -10, :]

    colors_2 = colors_2[xyz_2[:, 2] < 10, :]
    xyz_2 = xyz_2[xyz_2[:, 2] < 10, :]
    colors_2 = colors_2[xyz_2[:, 2] > -10, :]
    xyz_2 = xyz_2[xyz_2[:, 2] > -10, :]

    # 点云1只保留z轴在-2cm到10cm区间范围内的数据
    colors_1 = colors_1[xyz_1[:, 2] < 10, :]
    xyz_1 = xyz_1[xyz_1[:, 2] < 10, :]
    colors_1 = colors_1[xyz_1[:, 2] > -2, :]
    xyz_1 = xyz_1[xyz_1[:, 2] > -2, :]

    # # 显示粗配准的人脸
    # colors = np.vstack((colors_0, colors_1, colors_2))
    # points = np.vstack((xyz_0, xyz_1, xyz_2))
    # point_cloud = open3d.geometry.PointCloud()
    # point_cloud.points = open3d.utility.Vector3dVector(points)
    # point_cloud.colors = open3d.utility.Vector3dVector(colors / 255)
    # # point_cloud.paint_uniform_color([0,1,0])
    # # 坐标轴size单位为厘米，红、绿、蓝分别代表x,y,z。
    # axis_1 = open3d.geometry.TriangleMesh.create_coordinate_frame(size=6, origin=[0, 0, 0])
    # open3d.visualization.draw_geometries([point_cloud, axis_1], window_name='Open3D')



    # 3D-3D ICP 进行R，t迭代优化
    _source_0 = xyz_0
    _source_2 = xyz_2
    _target = xyz_1

    Homo_0, Homo_2 = ICP_3D_3D(user_path, _source_0, _source_2, _target)

    # print('matR_0', matR_0)
    # print('matR_2', matR_2)
    # print('Homo_0:', 57.3*rotationMatrixToEulerAngles(np.matmul(Homo_0[0:3,0:3], matR_0)))
    # print('Homo_2:', 57.3*rotationMatrixToEulerAngles(np.matmul(Homo_2[0:3,0:3], matR_2)))

    import json
    labelme_json_0 = json.load(open(json_0_path, encoding='utf-8'))
    labelme_json_2 = json.load(open(json_2_path, encoding='utf-8'))
    points_tr_0 = np.round(labelme_json_0['shapes'][1]['points']).astype(np.uint64)
    points_tr_2 = np.round(labelme_json_2['shapes'][1]['points']).astype(np.uint64)
    # print('points_tr_0', points_tr_0)
    # print('points_tr_2', points_tr_2)

    src_tr_al = np.zeros([4, 3])
    dst_tr_al = np.array([[-72643.3, 11413.3, 11140.1],
                          [-16981.4, -3046.09, 111901],
                          [71894.4, 11245.4, 11118.2],
                          [16732.3, -2973.56, 111813]]) / 10000

    # 此处的xyz0只是经过了第一次旋转平移矫正， 下面主要是为了通过原始点的m*n*3数据提取到al和t特征点并进行相应的旋转和平移。
    xyz0 = xyz0 - coordinate_origin
    xyz2 = xyz2 - coordinate_origin
    xyz0 = np.matmul(Homo_0[0:3, 0:3], xyz0.T).T + Homo_0[0:3, 3]
    xyz2 = np.matmul(Homo_2[0:3, 0:3], xyz2.T).T + Homo_2[0:3, 3]

    # 耳屏点 234, 454
    # 鼻翼点 (129,102) (358,331)
    xyz0 = xyz0.reshape(720, 1280, 3)
    xyz2 = xyz2.reshape(720, 1280, 3)
    src_tr_al[0, :] = xyz0[points_tr_0[0, 1], points_tr_0[0, 0], :]
    src_tr_al[1, :] = xyz0[landmarks0[102, 1], landmarks0[102, 0], :]
    src_tr_al[2, :] = xyz2[points_tr_2[0, 1], points_tr_2[0, 0], :]
    src_tr_al[3, :] = xyz2[landmarks2[331, 1], landmarks2[331, 0], :]

    matR_tr, vecT_tr = Face_ICP(src_tr_al, dst_tr_al)

    # 对关键点进行最后一轮的旋转
    src_tr_al = np.matmul(matR_tr, src_tr_al.T).T

    # print('src_tr_al:', src_tr_al)
    # print('dst_tr_al:', dst_tr_al)
    # print('matR_tr', rotationMatrixToEulerAngles(matR_tr), rotationMatrixToEulerAngles(matR_tr) * 180 / np.pi)
    angle = rotationMatrixToEulerAngles(matR_tr)
    angles.append(angle)
    print(angles)
    lines = [[0, 1], [1, 3], [3, 2], [2, 0]]  # 连接的顺序，封闭链接
    color = [[1, 0, 0] for i in range(len(lines))]  # 红色

    lines_pcd = open3d.geometry.LineSet()
    lines_pcd.lines = open3d.utility.Vector2iVector(lines)
    lines_pcd.colors = open3d.utility.Vector3dVector(color)  # 线条颜色
    lines_pcd.points = open3d.utility.Vector3dVector(src_tr_al)

    xyz_0 = np.matmul(Homo_0[0:3, 0:3], xyz_0.T).T + Homo_0[0:3, 3]
    xyz_2 = np.matmul(Homo_2[0:3, 0:3], xyz_2.T).T + Homo_2[0:3, 3]

    # 把多个视角的点云进行合并
    colors = np.vstack((colors_0, colors_1, colors_2))
    points = np.vstack((xyz_0, xyz_1, xyz_2))

# 对xyz_0, xyz_1和xyz_2进行最后一轮旋转
    points = np.matmul(matR_tr, points.T).T

    # point_cloud = open3d.geometry.PointCloud()
    # point_cloud.points = open3d.utility.Vector3dVector(points)
    # point_cloud.colors = open3d.utility.Vector3dVector(colors / 255)
    # # point_cloud.paint_uniform_color([0,1,0])
    # axis_nose = open3d.geometry.TriangleMesh.create_coordinate_frame(size=6, origin=src_tr_al[3, :])
    # axis_nose_2 = open3d.geometry.TriangleMesh.create_coordinate_frame(size=6, origin=src_tr_al[1, :])
    # axis_tr = open3d.geometry.TriangleMesh.create_coordinate_frame(size=6, origin=src_tr_al[2, :])
    # axis_tr_2 = open3d.geometry.TriangleMesh.create_coordinate_frame(size=6, origin=src_tr_al[0, :])
    #
    # # 坐标轴size单位为厘米，红、绿、蓝分别代表x,y,z。
    # axis_1 = open3d.geometry.TriangleMesh.create_coordinate_frame(size=6, origin=[0, 0, 0])
    # open3d.visualization.draw_geometries([point_cloud, axis_1, lines_pcd, axis_nose, axis_nose_2, axis_tr, axis_tr_2], window_name='Open3D', width=1536, height=864, left=50, top=50)
    #
    # # 侧视图
    # open3d.visualization.draw_geometries([point_cloud, axis_1, lines_pcd], window_name='Open3D',
    #                                      zoom=0.8,
    #                                      front=[-1, 0, 0],
    #                                      # lookat=[0, 0, 6],
    #                                      lookat=[0, 0, 0],
    #                                      up=[0, 1, 0]
    #                                      )

    return Homo_0, Homo_2

