# 复制指定图片到指定文件夹
from date_pro import get_date_list
import os
import shutil
from functools import partial
from pathos.multiprocessing import ProcessingPool as Pool


def copy_one_img(opath, inx, point, time, dir_name,tpath0):  # 复制。opath tpath是绝对路径
    # point = 1
    # time = 12
    # dir_name='F03210481'

    tpath = os.path.join(tpath0,
                         dir_name + 'p' + str(point) + 't' + str(time) + 'i' + str(inx) + 'd' + opath[-17:-4] + '.jpg')
    # print(opath, tpath)
    shutil.copyfile(opath, tpath)


if __name__ == '__main__':
    buddate = '2021-01-08'  # 出芽时间,设为时间原点
    # point = 1
    # time = '12'
    dir_name = 'E88570046'
    point_liat = [1, 2,  3,4,5]
    time_list = ['09', '12', '15']

    tpath0 = '/mnt/hdd/cherry2021/out'
    if not os.path.exists(tpath0):
        os.mkdir(tpath0)
    path = os.path.join('/mnt/hdd/cherry2021/', dir_name)


    for point in point_liat:
        for time in time_list:
            date_list = get_date_list(buddate, point=point, time=time, path=path)
            print(date_list)

            x = []
            y = []

            # print(len(date_list))

            for i in range(len(date_list)):
                name_list = date_list[i]

                # 查看路径是否存在
                x0 = [k for k in name_list if os.path.exists(k) == True]
                # print(x0)
                y0 = [i] * len(x0)
                # print(y0)
                x.extend(x0)
                y.extend(y0)

            print(x)
            print(y)

            # opath,inx,point,time,dir_name
            p = Pool()
            copy_one_img_p = partial(copy_one_img, point=point, time=time, dir_name=dir_name, tpath0=tpath0)
            p.map(copy_one_img_p, x, y)
