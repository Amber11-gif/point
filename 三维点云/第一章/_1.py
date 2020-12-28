# 对点云数据进行PCA分析并进行投影可视化
import numpy as np
import os
import open3d as o3d
from pyntcloud import PyntCloud
from pandas import DataFrame


def PCA(data, correlation=False, sort=True):
    """
    # 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
    """
    mean_vec = np.mean(data, axis=0)  # 取均值
    normal_vec = data-mean_vec        # 去中心化
    H_vec = np.dot(normal_vec.T, normal_vec)  # 求解协方差矩阵 H
    eigencectors, eigenvalues, _ = np.linalg.svd(H_vec) # # SVD求解特征值、特征向量

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigencectors = eigencectors[:, sort]

    return eigenvalues, eigencectors


def main():

    # 指定点云路径

    pointcloud = np.genfromtxt(r"airplane_0006.txt", delimiter=",")
    pointcloud = DataFrame(pointcloud[:, 0:3])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    pointcloud.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
    point_cloud_pynt = PyntCloud(pointcloud)  # 将points的数据 存到结构体中
    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)  # 实例化
    o3d.visualization.draw_geometries([point_cloud_o3d])  # 显示原始点云
    print(point_cloud_o3d)  # 打印点数

    # PCA分析点云主方向
    w, v = PCA(pointcloud)
    # print(w,v)
    point_cloud_vector1 = v[:,0]  # 最大特征值对应的特征向量，即第一主成分
    point_cloud_vector2 = v[:, 1]  # 点云主方向对应的向量，第二主成分
    point_cloud_vector = v[:, 0:2]  # 点云主方向与次方向
    print('the main orientation of this pointcloud is: ', point_cloud_vector1)
    print('the main orientation of this pointcloud is: ', point_cloud_vector2)
    # 在原点云中画图
    point = [[0, 0, 0], point_cloud_vector1, point_cloud_vector2]  # 画点：原点、第一主成分、第二主成分
    lines = [[0, 1], [0, 2]]  # 画出三点之间两两连线
    colors = [[1, 0, 0], [0, 0, 0]]
    # 构造open3d中的LineSet对象，用于主成分显示
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(point),
                                    lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud_o3d, line_set])  # 显示原始点云和PCA后的连线
    # axis = o3d.geometry.TriangleMesh.create_coordinate_frame().rotate(v, center=(0, 0, 0))
    # create_from_triangle_mesh函数能够从三角网格中生成体素网格。
    # 它返回一个体素网格，其中所有与三角形相交的网格被设置为1，其余的设置为0。其中voxel_zie参数是用来设置网格分辨率。
    # pr_data = np.dot(pointcloud, -v[:, 0:2])
    # pr_data = np.insert(pr_data,1,values=-1*np.ones((1,pr_data.shape[0])),axis=1)
    # pr_data2 = pointcloud - np.dot(pointcloud, v[:, 2][:, np.newaxis]) * v[:,2]
    # pr_data2 = 1*v[:, 2] + pr_data2
    # print(pr_data2.shape)
    principle_axis = np.concatenate((np.array([[0., 0., 0.]]), v.T))
    # print(principle_axis)
    # pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pointcloud))
    # colors = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    # pc_view.colors = o3d.utility.Vector3dVector(colors)
    # pr_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pr_data))
    # pr_view2 = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pr_data2))
    # o3d.visualization.draw_geometries([pc_view, axis, pr_view2])
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)
    normals = []
    # geometry::PointCloud with 10000 points.
    # print(pointcloud.shape[0])  # 10000
    for i in range(pointcloud.shape[0]):
        # search_knn_vector_3d函数 ， 输入值[每一点，x]
        # 返回值 [int, open3d.utility.IntVector, open3d.utility.DoubleVector]
        [_, idx, _] = pcd_tree.search_knn_vector_3d(point_cloud_o3d.points[i], 10)  # 10 个临近点
        # asarray和array 一样 但是array会copy出一个副本，asarray不会，节省内存
        k_nearest_point = np.asarray(point_cloud_o3d.points)[idx, :]
        # 找出每一点的10个临近点，类似于拟合成曲面，然后进行PCA找到特征向量最小的值，作为法向量
        w, v = PCA(k_nearest_point)
        normals.append(v[:, 2])
    normals = np.array(normals, dtype=np.float64)
    print(normals.shape)

    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([point_cloud_o3d])


if __name__ == '__main__':
    main()




