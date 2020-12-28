# voxel Grid Downsampling
import numpy as np
import os
from pyntcloud import PyntCloud
import random
from pandas import DataFrame
import open3d as o3d


def voxelfilter(point_clound, leaf_size, filter_mode):
    print(point_clound.shape)
    filter_points = []
    # step 1 计算边界值
    x_max, y_max, z_max = np.amax(point_clound, axis=0)
    x_min, y_min, z_min = np.amin(point_clound, axis=0)
    # step 2 确定体素的尺寸
    size_r = leaf_size
    # step 3 计算每个voxel的维度
    Dx = (x_max - x_min)/size_r+1
    Dy = (y_max - y_min)/size_r+1
    Dz = (z_max - z_min)/size_r+1
    print("Dx x Dy x Dz is {} x {} x {}".format(Dx, Dy, Dz))
    # step 4 计算每个点在voxel grid内每一个维度的值
    h = list()
    for i in range(len(point_clound)):
        hx = (point_clound[i][0] - x_min)/size_r
        hy = (point_clound[i][1] - y_min)/size_r
        hz = (point_clound[i][2] - z_min)/size_r
        h.append(hx+hy*Dx+hz*Dx*Dy)
    h = np.array(h)
    # 筛选点
    # step 5 对h值进行排序
    h_indice = np.argsort(h)  # # 返回h里面的元素按从小到大排序的索引
    h_sorted = h[h_indice]
    count = 0  # 用于维度的统计
    for i in range(len(h_sorted)-1):
        if h_sorted[i] == h_sorted[i+1]:    # #当前的点与后面的相同，放在同一个volex grid中
            continue
        else:
            if (filter_mode == 'centroid'):
                point_idx = h_indice[count:i+1]
                filter_points.append(np.mean(point_clound[point_idx], axis=0))  # #取同一个grid的均值
                count = i
            elif (filter_mode == 'random'):
                point_idx = h_indice[count:i+1]
                random_points = random.choice(point_clound[point_idx])
                filter_points.append(random_points)
                count = i
    # 把点云格式改成array，并对外返回
    filter_points = np.array(filter_points, dtype=np.float64)
    print(len(filter_points))
    return filter_points


def main():
    path = 'D:\dataset\modelnet40_normal_resampled'
    shape_name_list = np.loadtxt(os.path.join(path, 'modelnet40_shape_names.txt') if os.path.isdir(path) else None,
                                 dtype=str)
    pc_list = []
    for item in shape_name_list:
        filename = os.path.join(path, item, item + '_0001.txt')
        pointcloud = np.loadtxt(filename, delimiter=',')[:, 0:3]
        """point_cloud_raw = DataFrame(pointcloud)  # 为 xyz的 N*3矩阵
        point_cloud_raw.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
        point_cloud_pynt = PyntCloud(point_cloud_raw)  # 将points的数据 存到结构体中
        point_cloud_o3d_orign = point_cloud_pynt.to_instance("open3d", mesh=False)  # to_instance实例化
        point_cloud_o3d = o3d.geometry.PointCloud()  # 实例化
        云
        pointcloud_o3d = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pointcloud))
        # o3d.visualization.draw_geometries([pointcloud_o3d])  # 显示原始点
        # 调用voxel 滤波函数
        points = np.asarray(pointcloud_o3d.points)
        filtered_cloud = voxelfilter(points, 0.05, 'centroid')
        point_cloud_o3d_filter = o3d.geometry.PointCloud()  # 实例化
        point_cloud_o3d_filter.points = o3d.utility.Vector3dVector(filtered_cloud)
        # o3d.visualization.draw_geometries([point_cloud_o3d_filter])
        """
        pointcloud_o3d = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pointcloud))
        filtered_cloud = voxelfilter(pointcloud, 0.05, 'random') + 1*np.array([0, 1, 0])
        filteredcloud_o3d = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(filtered_cloud))
        o3d.visualization.draw_geometries([pointcloud_o3d, filteredcloud_o3d])


if __name__ == '__main__':
    main()


