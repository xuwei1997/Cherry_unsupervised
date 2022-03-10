import numpy as np
from preprocess_img import double_img


def minibatches(inputs=None, targets=None, batch_size=None, p=None):
    while 1:  # 要无限循环
        assert len(inputs) == len(targets)  # 判断输入数据长度和label长度是否相同
        loop_num = int(len(targets) / batch_size)  # 循环次数
        for start_idx in range(loop_num):
            # print(start_idx)
            excerpt = slice(start_idx * batch_size, start_idx * batch_size + batch_size)
            # print(excerpt)
            # print(inputs[excerpt], targets[excerpt])
            x = p.map(double_img, inputs[excerpt])
            x = np.array(x)
            # print(x.shape)
            y = targets[excerpt]
            y = np.array(y)
            # yield x, y  # 每次产生batchsize个数据
            yield ({'self_input1': x[:, 0, :, :, :], 'self_input2': x[:, 1, :, :, :]}, {'output': y})


if __name__ == '__main__':
    inputs = range(0, 100)
    targets = range(100, 200)
    minibatches(inputs, targets, 10)
