import util
import argparse
import torch
from model import TEDDCF
import numpy as np
import pywt
import pandas as pd
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0", help="")
parser.add_argument("--data", type=str, default="PEMS08", help="data path")
parser.add_argument("--input_dim", type=int, default=3, help="number of input_dim")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.3, help="dropout rate")
parser.add_argument(
    "--weight_decay", type=float, default=0.0001, help="weight decay rate"
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="'/Ablation/epoch100_20240326_fulltrees_delete_Conv3_and_Conv4'PEMS08/best_model.pth",
    help="",
)
args = parser.parse_args()

def disentangle(x, w, j):
    #x为各批次数据[64,1,170,12],w是一个名称，j为lawer=2
    x = x.transpose(0,3,2,1) # [S,D,N,T][64,1,170,12]
    #小波分解
    coef = pywt.wavedec(x, w, level=j)#使用PyWavelets库中的wavedec函数对转置后的数据进行多级小波分解，w是所选择的小波基，level=j指定了分解的层数。是一个list
    #分解后得到的coef列表包含了各级小波系数，包括逼近系数（低频部分）和细节系数（高频部分）。初始化两个列表coefl和coefh：
    coefl = [coef[0]]
    for i in range(len(coef)-1):
        coefl.append(None)#两个list第一个为[64,1,170,8],第二个是None
    coefh = [None]
    for i in range(len(coef)-1):
        coefh.append(coef[i+1])#两个list第二个为[64,1,170,8],第一个个是None

    #重构小波,低频和高频重构
    xl = pywt.waverec(coefl, w).transpose(0,3,2,1)
    xh = pywt.waverec(coefh, w).transpose(0,3,2,1)

    return xl, xh#都是nadarray [64,12,170,1]

wave = 'coif1'
level = 1


def main():

    if args.data == "PEMS08":
        args.data = "data//" + args.data
        num_nodes = 170
        granularity = 288
        channels = 96

    elif args.data == "PEMS03":
        args.data = "data//" + args.data
        num_nodes = 358
        args.epochs = 300
        args.es_patience = 100
        granularity = 288
        channels = 32

    elif args.data == "PEMS04":
        args.data = "data//" + args.data
        num_nodes = 307
        granularity = 288
        channels = 64


    elif args.data == "PEMS07":
        args.data = "data//" + args.data
        num_nodes = 883
        granularity = 288
        channels = 128


    elif args.data == "bike_drop":
        args.data = "data//" + args.data
        num_nodes = 250
        granularity = 48
        channels = 32


    elif args.data == "bike_pick":
        args.data = "data//" + args.data
        num_nodes = 250
        granularity = 48
        channels = 32


    elif args.data == "taxi_drop":
        args.data = "data//" + args.data
        num_nodes = 266
        granularity = 48
        channels = 96

    elif args.data == "taxi_pick":
        args.data = "data//" + args.data
        num_nodes = 266
        granularity = 48
        channels = 96

    device = torch.device(args.device)

    model = TEDDCF(
        device, args.input_dim, num_nodes, channels, granularity, args.dropout
    )
    model.to(device)
    model.load_state_dict(torch.load(args.checkpoint))
    model.eval()
    print("model load successfully")

    dataloader = util.load_dataset(
        args.data, args.batch_size, args.batch_size, args.batch_size
    )
    scaler = dataloader["scaler"]
    outputs = []
    realy = torch.Tensor(dataloader["y_test"]).to(device)
    realy = realy.transpose(1, 3)[:, 0, :, :]

    for iter, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
        #############################
        x_flow = x[:, :, :, 0]  # x_flow进来的时候已经编码和归一化了
        x_flow = x_flow[:, :, :, np.newaxis]  # [64,12,170,1]
        x_flow = scaler.inverse_transform(x_flow)
        y_flow = y[:, :, :, 0]
        y_flow = y_flow[:, :, :, np.newaxis]
        TE = x[:, :, :, 1:3]  # [64,12,170,2]

        XL, XH = disentangle(x_flow, wave, level)  # DWT不需要归一化和one-hot编码
        YL, _ = disentangle(y_flow, wave, level)
        XL, XH = scaler.transform(XL), scaler.transform(XH)

        testxl, testxh, TE = torch.Tensor(XL).to(device), torch.Tensor(XH).to(device), torch.Tensor(TE).to(device)
        testxl = torch.cat((testxl, TE), dim=3)
        testxh = torch.cat((testxh, TE), dim=3)
        ##########################

        testxl = testxl.transpose(1, 3)
        testxh = testxh.transpose(1, 3)
        with torch.no_grad():
            preds = model(testxl,testxh).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[: realy.size(0), ...]

    amae = []
    amape = []
    awmape = []
    armse = []

    for i in range(12):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        log = "Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test WMAPE: {:.4f}"
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2], metrics[3]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        awmape.append(metrics[3])

    log = "On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}, Test WMAPE: {:.4f}"
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse), np.mean(awmape)))


if __name__ == "__main__":
    main()
