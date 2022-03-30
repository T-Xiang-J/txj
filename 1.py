import numpy as np
import torch
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

def load_data(path):
    # 原始数据集为xls文件,将导入的xls文件转换为矩阵
    import xlrd
    data = xlrd.open_workbook(path)
    table = data.sheets()[0]
    nrows = table.nrows  # 行数
    x = []
    for i in range(nrows):
        x.append(table.row_values(i))
    datamatrix = np.mat(x)
    return datamatrix
mat = load_data(r".\data数据2.xls")
m = mat.shape[0]  # 行数
n = mat.shape[1]  # 列数

total_relative_error=[]
total_mean_abs_error=[]
b=4
q1=np.repeat(np.linspace(0,n-1,n),b)


for j in range(len(q1)):
    print('第'+str(j+1)+'次运行')
    q=int(q1[j])
    import random
# q = np.random.randint(1, n)
    print('预测特征:'+str(q))
    y = mat[:, q]
    x = np.delete(mat, q, axis=1)
    p = []
    for i in range(round(0.01 * m)):
        p.append(random.randint(0, m - 1))
    new_x = mat[p, :]
    x = np.delete(mat, p, axis=0)
    origin_y = y[p, :]
    y = np.delete(y, p, axis=0)

    input_size = x.shape[1]
    hidden_size = 10
    output_size = 1
    batch_size = 50000


    class Txj(nn.Module):
        def __init__(self):  # 初始化
            super(Txj, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, output_size),
            )

        def forward(self, x):
            x = self.model(x)
            return x

    cost = torch.nn.MSELoss(reduction='mean')

    net = Txj()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    # if torch.cuda.is_available():
    #     net = net.cuda()
    #     cost = cost.cuda()
    net.to(device)
    cost.to(device)

    epoch = 1000
    for i in range(epoch):
        batch_loss = []
        # MINI-Batch方法来进行训练
        for start in range(0, len(x), batch_size):
            end = start + batch_size if start + batch_size < len(x) else len(x)
            batch_x = torch.tensor(x[start:end], dtype=torch.float, requires_grad=True)
            batch_y = torch.tensor(y[start:end], dtype=torch.float, requires_grad=True)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # if torch.cuda.is_available():
            #     batch_x = batch_x.cuda()
            #     batch_y = batch_y.cuda()

           # print(batch_x.is_cuda)

            prediction = net(batch_x)
            loss = cost(prediction, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.data.cpu().numpy())
        # 打印损失
        losses = []
        # if i % 100 == 0:
        #     losses.append(np.mean(batch_loss))
            #print(i, round(np.mean(batch_loss), 4))
        if i==epoch-1:
            print("MSE为"+str(round(np.mean(batch_loss), 4)))
        if np.mean(batch_loss) < 0.1:
            print("MSE为" + str(round(np.mean(batch_loss), 4)))
            print("最大迭代次数为"+str(i+1))
            break

    new_x = torch.tensor(new_x, dtype=torch.float)
    if torch.cuda.is_available():
        new_x = new_x.cuda()

    predict = net(new_x).data.cpu().numpy()
    comparison = np.hstack((origin_y, predict))
    mean_abs_error = np.mean(abs(origin_y - predict))
    relative_error = round(np.mean(abs(origin_y - predict) / abs(origin_y)), 4)
    print("平均绝对误差为"+"%.2f" % mean_abs_error)
    print("平均相对误差为"+"%.2f" % (relative_error * 100)+"%")

    total_mean_abs_error.append(mean_abs_error)
    total_relative_error.append(relative_error)

total_mean_abs_error=np.reshape(np.array(total_mean_abs_error),(n,b))
total_relative_error=np.reshape(np.array(total_relative_error),(n,b))

tt_mean_abs_error=np.round(np.mean(total_mean_abs_error.T,axis=0),4)
tt_mean_rea_error=np.round(np.mean(total_relative_error.T,axis=0),4)

a_total_stack = np.vstack((tt_mean_abs_error, tt_mean_rea_error))
print("结束了，请看结果！")