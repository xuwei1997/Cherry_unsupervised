# 创建日期辅助表

import datetime
from functools import partial
import os


# import shutil


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


def img_list(name, path):
    path1 = os.path.join(path, name)
    return path1


def get_date_list(buddate, point, time, path):
    datestart = '2021-01-01'
    datesend = '2021-07-01'
    # buddate = '2021-01-18'  # 出芽时间,设为时间原点
    days = 12  # 每个时间段选取多少张图片

    # 相对时间原点的偏移量
    # p0 = -10  # 休眠期
    # p1 = 3  # 出芽
    # p2 = 15  # 开花
    # p3 = 45  # 幼果出现
    # p4 = 90  # 变色变红

    p0 = -12  # 休眠期
    p1 = 10  # 绿叶与开花
    p2 = 45  # 幼果出现
    p3 = 84  # 变色变红

    data_list = create_assist_date(datestart, datesend)  # 辅助时间序列
    budinx = data_list.index(buddate)  # 时间原点
    # print(data_list)
    rename_img_p = partial(rename_img, points=point, times=time)  # 更名
    name_list = list(map(rename_img_p, data_list))

    # 绝对路径序列
    img_list_p = partial(img_list, path=path)
    data_list = list(map(img_list_p, name_list))

    # budinx = data_list.index(buddate)
    d0 = data_list[p0 + budinx:p0 + budinx + days]
    d1 = data_list[p1 + budinx:p1 + budinx + days]
    d2 = data_list[p2 + budinx:p2 + budinx + days]
    d3 = data_list[p3 + budinx:p3 + budinx + days]
    # d4 = data_list[p4 + budinx:p4 + budinx + days]
    # return [d0, d1, d2, d3, d4]

    ## 查看路径是否存在,取最后的days点作为已采摘
    data_list_a = [k for k in data_list if os.path.exists(k) == True]
    d4 = data_list_a[-days:]

    return [d0, d1, d2, d3, d4]


if __name__ == '__main__':
    buddate = '2021-01-08'  # 出芽时间,设为时间原点
    point = 1
    time = '09'
    dir_name = 'E88570046'
    path = os.path.join('/mnt/hdd/cherry2021/', dir_name)

    date_list = get_date_list(buddate, point=point, time=time, path=path)
    print(date_list)
