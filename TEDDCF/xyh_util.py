import numpy as np
import os
import scipy.sparse as sp
import torch
import sys

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        #xs/ys:[len(x_train.shape[0]\val\test),12,N,3]
        # print(xs.shape)train进来时(10699, 12, 170, 3)
        # sys.exit()

        self.batch_size = batch_size#64
        self.current_ind = 0
        if pad_with_last_sample:#表示需要使用最后一个样本进行填充以使数据集的总样本数能够被批量大小（batch_size）整除。
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)#函数复制最后一个样本作为填充数据，使之能被整除
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():#用的时候配合迭代器，一次只取一个batch数据
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind:end_ind, ...]
                y_i = self.ys[start_ind:end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1
                # print(self.current_ind)

        return _wrapper()

class XYHScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        pass

    def normal(self, data):
        return (data - self.mean) / self.std

    def inverse_normal(self, data):
        return data * self.std + self.mean

def seq2instance(data, P, Q):
    num_step, nodes, dims = data.shape#10714
    num_sample = num_step - P - Q + 1#10691
    x = np.zeros(shape = (num_sample, P, nodes, dims))#[10691,12,170,1]
    y = np.zeros(shape = (num_sample, Q, nodes, dims))#[10691,12,170,1]
    for i in range(num_sample):#切片12的时间步
        x[i] = data[i : i + P]#当前12步
        y[i] = data[i + P : i + P + Q]#未来12步
    return x, y

def load_data(url,batch_size):
    data={}#这里面要存放train、val、test已经std mean还有一份scare类
    #在我写的工具文件中，尽量避免使用已经预处理过的数据集
    print(url)
    origin_data = np.load(url)['data'][...,:1]#[17856,170,1]
    TE = np.zeros([origin_data.shape[0], 2])  # ndarray[17856,2]
    TE[:, 0] = np.array([i%288/288 for i in range(origin_data.shape[0])])  # day 保留6位小数
    TE[:, 1] = np.array([int(i // 288) % 7 for i in range(origin_data.shape[0])])  # week
    # 将TE拓展为[17856, 170, 2]的形状
    TE_expanded = np.repeat(np.expand_dims(TE, axis=1), repeats=origin_data.shape[1], axis=1)
    merged_data = np.concatenate((origin_data, TE_expanded), axis=2)#合并为[17856,170,3]

    train_num = round(0.6 * origin_data.shape[0])  # 10714 round四舍五入
    test_num = round(0.2 * origin_data.shape[0])  # 3571
    val_num = origin_data.shape[0] - train_num - test_num  # 3571

    trainData,valData,testData = merged_data[:train_num,:,:], merged_data[train_num:train_num+val_num,:,:],merged_data[-test_num:,:,:]

    #数据好像比原版少了几个，但我检查，其他都一样
    trainX, trainY = seq2instance(trainData, 12, 12)  # 返回窗口化后的数据x,y[106911,12,170,3]
    valX, valY = seq2instance(valData, 12, 12)  # [3571,12,170,3]
    testX, testY = seq2instance(testData, 12, 12)  # [3548,12,170,3]

    data['x_train'] = trainX
    data['y_train'] = trainY
    data['x_val'] = valX
    data['y_val'] = valY
    data['x_test'] = testX
    data['y_test'] = testY

    scaler = XYHScaler(mean=data['x_train'][...,0].mean(), std=data['x_train'][...,0].std())
    for category in ["train", "val", "test"]:#训练值都要归一化，归一化的std和mean都是train数据集中的
        data["x_" + category][..., 0] = scaler.normal(data["x_" + category][..., 0])
        pass

    print("scaler:mean:{},std:{}".format(scaler.mean, scaler.std))
    print('x_train.shape:{},x_val.shape:{},x_test.shape:{}'.format(trainX.shape, valX.shape, testX.shape))
    print('y_train.shape:{},y_val.shape:{},y_test.shape:{}'.format(trainY.shape, valY.shape, testY.shape))

    # 对顺序出现的数据全局随机打乱
    print("Perform shuffle on the dataset")
    random_train = torch.arange(int(data["x_train"].shape[0]))  # 0到len的顺序索引
    random_train = torch.randperm(random_train.size(0))  # 打乱索引
    data["x_train"] = data["x_train"][random_train, ...]  # 按照索引打乱x和标签值
    data["y_train"] = data["y_train"][random_train, ...]

    # 打乱val
    random_val = torch.arange(int(data["x_val"].shape[0]))
    random_val = torch.randperm(random_val.size(0))
    data["x_val"] = data["x_val"][random_val, ...]
    data["y_val"] = data["y_val"][random_val, ...]
    #别打乱test，不然怎么可视化

    data['scaler'] = scaler
    data["train_loader"] = DataLoader(data["x_train"], data["y_train"], batch_size)  # 只是在字典内定义了一个方法
    data["val_loader"] = DataLoader(data["x_val"], data["y_val"], batch_size)
    data["test_loader"] = DataLoader(data["x_test"], data["y_test"], batch_size)

    print()




if __name__ == '__main__':
    load_data(url='data/PEMS08/PEMS08.npz',batch_size=64)
    print('hello')

