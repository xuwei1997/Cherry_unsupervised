# 聚类

import os
from PIL import Image
import numpy as np
# from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from sklearn.cluster import KMeans, DBSCAN

import datetime
from functools import partial
import os
import shutil
from scipy import stats

# from pathos.multiprocessing import ProcessingPool as Pool


def create_assist_date(datestart=None, dateend=None):
    # 创建日期辅助表

    if datestart is None:
        datestart = '2016-01-01'
    if dateend is None:
        dateend = datetime.datetime.now().strftime('%Y-%m-%d')

    # 转为日期格式
    datestart = datetime.datetime.strptime(datestart, '%Y-%m-%d')
    dateend = datetime.datetime.strptime(dateend, '%Y-%m-%d')
    date_list = []
    date_list.append(datestart.strftime('%Y-%m-%d'))
    while datestart < dateend:
        # 日期叠加一天
        datestart += datetime.timedelta(days=+1)
        # 日期转字符串存入列表
        date_list.append(datestart.strftime('%Y-%m-%d'))
    # print(date_list)
    return date_list


def rename_img(dates, points, times):  # 更改文件名
    s = str(points) + '_' + dates + '_' + str(times) + '.jpg'
    return s


def get_dir(path, kind=True):  # 获取文件夹下路径
    filenames = os.listdir(path)
    filenames = sorted(filenames)
    if kind:
        pathnames = [os.path.join(path, filename) for filename in filenames]
        return pathnames
    else:
        return filenames


def img_list(name, path):  # 复制。name是文件名，opath tpath是目录
    path1 = os.path.join(path, name)
    return path1

def copy_one_img(opath,inx,point,time,dir_name):  # 复制。opath tpath是绝对路径
    # point = 1
    # time = 12
    # dir_name='F03210481'
    tpath=os.path.join('/mnt/hdd/data/out', dir_name+'p'+str(point)+'t'+str(time)+'i'+str(inx)+'d'+opath[-17:-4]+'.jpg')
    shutil.copyfile(opath, tpath)

#############################################################################

def preprocess_input_one(image):  # 将输入映射到[-1,1] 之间
    image = image / 127.5 - 1
    return image


def image_out_one(path):  # 用PIL读取RGB,单张图片
    jpg = Image.open(path)
    # jpg.show()
    jpg = np.array(jpg)

    # 将输入映射到[-1,1] 之间
    # jpg = np.array(jpg, np.float64)
    # jpg = preprocess_input_one(jpg) # 将输入映射到[-1,1] 之间

    # print(jpg)

    return jpg


def ave_rgb_one(img):  # 计算单张图片平均值
    # 整型为二维
    # print(img.shape)
    sh = img.shape
    re_img = img.reshape(sh[0] * sh[1], sh[2])
    # print(re_img.shape)
    ave = np.mean(re_img, axis=0)
    # print(ave.shape)
    # print(ave)

    return ave[0], ave[1], ave[2]
    # return ave


def dir_clu(clusters_kind,datestart,datesend,point,time,dir_name):  # 针对某个目录获取聚类结果，返回装有聚类结果的df
    # rgb_list = get_dir(path)  # 获取文件夹下路径

    # datestart = '2021-01-08'
    # datesend = '2021-05-25'
    # point = 2
    # time = '12'
    # dir_name='E88569964'

    path=os.path.join('/mnt/hdd/data/',dir_name)

    # 获取文件名列表
    # 辅助时间序列
    data_list = create_assist_date(datestart, datesend)
    # 文件名序列
    rename_img_p = partial(rename_img, points=point, times=time)
    name_list = map(rename_img_p, data_list)
    # 绝对路径序列
    img_list_p = partial(img_list, path=path)
    name_list_abs = list(map(img_list_p, name_list))
    print(name_list_abs)

    # 查看路径是否存在
    name_list_abs = [x for x in name_list_abs if os.path.exists(x) == True]

    # image_out_one(list(name_list_abs)[1])

    # 获取每张图片平均rgb
    p = Pool()
    data_img = p.map(image_out_one, name_list_abs)  # 用PIL读取RGB
    avg = p.map(ave_rgb_one, data_img)  # RGB
    print(len(avg))

    # 聚类
    estimator = KMeans(n_clusters=clusters_kind, random_state=42)  # 构造聚类器,随机数种子确定
    # # estimator = DBSCAN()  # 构造聚类器
    estimator.fit(avg)
    l = estimator.predict(avg)  # 聚类,返回标签
    # print(l)
    centroids = estimator.cluster_centers_  # 获取聚类中心
    # print(centroids)
    t = estimator.transform(avg)  # 到聚类中心距离
    t = np.array(t)
    # print(t)
    label_pred = estimator.labels_  # 获取聚类标签
    print(label_pred)
    # s = estimator.score(avg)
    # print(s)
    #
    # print(np.array(name_list_abs)[label_pred==3]) #输出标签对应的路径

    # 找出离聚类中心最近的点，按时间排序（按序号排序）

    in_min=np.argmin(t,axis = 0)
    print(in_min)
    a=np.argsort(in_min)
    print(a)
    l=np.array(range(clusters_kind))

    in_list=[]
    # 排序，替换
    for i in range(clusters_kind):
        # print(in_min[a == i][0])
        k=l[a == i][0]
        label_pred[label_pred==i]=k+10
        in_list.append(k+10)
    print(in_list)

    print(label_pred)

    # 每类都获取最接近的a个点
    a = 6
    ind = np.argpartition(t, a, axis=0)[:a, :]
    print(ind)
    # ind = np.argsort(t[:,1])
    # print(ind)
    x=[]
    y=[]
    #取出最接近的点，但不足a个，则取最少的数量
    for i in range(clusters_kind):
        k=ind[:, i]
        print(k)
        label=in_list[i] #标签
        xo=np.array(name_list_abs)[k]
        yo=np.array(label_pred)[k]
        #过滤小于a的点
        print(yo == label)
        yo1 = yo[yo == label]
        xo1 = xo[yo == label]
        # yo=[in_list[i]]*a
        print(xo1)  # 输出路径
        print(yo1)  # 输出标签
        x.extend(xo1)
        y.extend(yo1)


    print(x)
    print(y)

    #opath,inx,point,time,dir_name
    copy_one_img_p = partial(copy_one_img, point=point,time=time,dir_name=dir_name)
    p.map(copy_one_img_p,x,y)





if __name__ == '__main__':
    # dir_clu('E:\\charry\\data\\E88569964p11t09')
    clusters_kind = 4
    datestart = '2021-01-04'
    datesend = '2021-04-30'
    point_liat = [1,2,5,6,7,8,9]
    time_list = ['09','12','15']
    dir_name = 'F03210481'
    for point in point_liat:
        for time in time_list:
            dir_clu(clusters_kind,datestart,datesend,point,time,dir_name)
