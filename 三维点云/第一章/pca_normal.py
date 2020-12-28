# 实现PCA分析，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
from pyntcloud import PyntCloud
from pandas import DataFrame

# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量


def PCA(data, correlation=False, sort=True):
    mean_vec = np.mean(data,axis=0)
    normal_vec = data - mean_vec
    H_vec = np.dot(normal_vec.T , normal_vec)

    eigenvectors,eigenvalues,_ = np.linalg.svd(H_vec)

    # 屏蔽结束

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def main():
    # 指定点云路径
    path = 'D:\dataset\modelnet40_normal_resampled'

    shape_name_list = np.loadtxt(os.path.join(path,'modelnet40_shape_names.txt') if os.path.isdir(path) else None,dtype=str)
    pc_list = []

    for item in shape_name_list:
        filename = os.path.join(path,item,item+'_0001.txt')
        pointcloud = np.loadtxt(filename,delimiter=',')[:,0:3]
        print('total points number is:', pointcloud.shape[0])
        pointcloud_row = DataFrame(pointcloud)  # 选取每一列 的 第0个元素到第二个元素   [0,3)
        pointcloud_row.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
        point_cloud_pynt = PyntCloud(pointcloud_row)  # 将points的数据 存到结构体中
        point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)  # 实例化
        o3d.visualization.draw_geometries([point_cloud_o3d])  # 显示原始点云

        w,v = PCA(pointcloud)

        # PCA分析点云主方向
        pointcloud_vector = v[:,0]
        print('the main orientation of this pointcloud is: ', pointcloud_vector)

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame().rotate(v,center=(0,0,0))
        # pr_data = np.dot(pointcloud,-v[:,0:2])
        # pr_data = np.insert(pr_data,1,values=-1*np.ones((1,pr_data.shape[0])),axis=1)

        pr_data2 = pointcloud - np.dot(pointcloud,v[:,2][:,np.newaxis])*v[:,2]
        pr_data2 = 1*v[:,2]+pr_data2 


        principle_axis = np.concatenate((np.array([[0.,0.,0.]]),v.T))
        print(principle_axis)
        # colors = [[1,0,0],[0,1,0],[0,0,1]]
        # lines = [[0,1],[0,2],[0,3]]
        # line_set = o3d.geometry.LineSet(
        #     points=o3d.utility.Vector3dVector(principle_axis),
        #     lines = o3d.utility.Vector2iVector(lines),
        # )
        # line_set.colors = o3d.utility.Vector3dVector(colors)
        pc_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pointcloud))
        pc_view.colors = o3d.utility.Vector3dVector([[0,0,0]])
        # pr_view = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pr_data))
        pr_view2 = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pr_data2))
        o3d.visualization.draw_geometries([pc_view,axis,pr_view2])

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
