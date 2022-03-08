#制作数据集
import os
import itertools
import numpy as np
import sklearn

def get_dir(path, kind=True):  # 获取文件夹下路径
    filenames = os.listdir(path)
    filenames = sorted(filenames)
    if kind:
        pathnames = [os.path.join(path, filename) for filename in filenames]
        return pathnames
    else:
        return filenames


def get_label(X): #生成标签
    a = X[0]
    b = X[1]
    # 摄像头号与预置点号
    npa = a[:11]
    npb = b[:11]
    # 标签号
    ia = a[15]
    ib = b[15]

    if npa == npb:
        if ia == ib:
            label = 0
        else:
            label = 1
    else:
        if ia == ib:
            label = 2
        else:
            label = 3

    # print(npa, npb, ia, ib)
    # print(label)

    return label


if __name__ == '__main__':
    path = '/mnt/hdd/cherry2021/out'
    filenames = get_dir(path, kind=False)
    # get_label((filenames[0],filenames[12]))
    comb = list(itertools.combinations(filenames, 2))
    label = map(get_label, comb)

    # 重采样前
    y_np = np.array(list(label))
    label = [0, 1, 2, 3]
    re_num = 56245 #重采样数

    X=[]
    Y=[]

    for i in label:
        # 获取这类的索引
        w = np.where(y_np == i)[0]
        # print(w)

        # 采样
        h = len(w)
        # print(h)
        k = np.random.randint(0, h, size=re_num) #获取随机的re_num个数
        o=w[k] # 重采样后的索引
        # print(o)

        # 获取一类的数据
        xt=np.array(comb)[o]
        yt=y_np[o]
        # print(xt)
        # print(yt)
        # print(np.sum(yt == i))

        X.extend(xt)
        Y.extend(yt)

    #打乱
    X, Y = sklearn.utils.shuffle(X, Y)
    # print(X)


