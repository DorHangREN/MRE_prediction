import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import sys
from datetime import datetime#建立excel保存精度和损失
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

from CNN_BiLSTM_Attention_model import CNNBiLSTMAttention

df = pd.DataFrame(columns=['time', 'step', 'train Loss'])  # 列名#建立excel保存精度和损失

df.to_csv("D:/Performance prediction of magnetorheological elastomers based on CNN-BiLSTM Attention/prediction_code/CNN_BiLSTM_Attention/loss_train_Fe.csv",
              index=False)  ##建立excel保存精度和损失路径可以根据需要更改
df = pd.DataFrame(columns=['time', 'step', 'test Loss'])  # 列名#建立excel保存精度和损失
df.to_csv("D:/Performance prediction of magnetorheological elastomers based on CNN-BiLSTM Attention/prediction_code/CNN_BiLSTM_Attention/loss_test_Fe.csv",
              index=False)  ##建立excel保存精度和损失路径可以根据需要更改


# PATH = "mdoel_weight_18_5_75.pth"
# 创建CNN-BiLSTM-Attention模型实例
model = CNNBiLSTMAttention().to(device)
# model.load_state_dict(torch.load(PATH))#第一次跑注释掉

# 自定义数据集类
class MyDataset(Dataset):
    def __init__(self, feature_file, label_file):
        # 读取数据集
        data_features = np.genfromtxt(feature_file, delimiter=',')
        data_labels = np.genfromtxt(label_file, delimiter=',')
        # 转置数据集
        data_1 = data_features.T
        data_2 = data_labels.T
        self.features = data_1
        self.labels = data_2

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # for col in df.columns:
        #     data = df[col]
        # feature = torch.tensor(self.features[idx][:200], dtype=torch.float32)
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return feature, label


# 加载数据集
feature_file = 'train_test_feature_400_Tem.csv'
label_file = 'train_test_stroage_mudule_Tem.csv'
dataset = MyDataset(feature_file, label_file)

# 划分训练集和测试集
train_size = 0.9
train_dataset, test_dataset = train_test_split(dataset, train_size=train_size, shuffle=True)


# 创建数据加载器
batch_size = 6
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
losses = []
num_epochs = 100
for epoch in range(num_epochs):
    train_bar = tqdm(train_dataloader, file=sys.stdout)
    # 训练阶段
    model.train()
    for i, (features, labels) in enumerate(train_dataloader):
        # 前向传播
        features = features.to(device)
        labels = labels.to(device)

        outputs = model(features)
        print(labels)
        print('Predicted Labels:', outputs)
        # 计算损失
        # labels=torch.squeeze(labels, dim=1)   #为标签消去一个维度
        loss = criterion(outputs, labels)  #outputs:torch.size[2,101]   label:troch.size[2,101]

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        time = "%s" % datetime.now()  # 获取当前时间
        list = [time, i, loss.item()]  # 建立excel保存精度和损失
        data = pd.DataFrame([list])  # 建立excel保存精度和损失
        data.to_csv('D:/Performance prediction of magnetorheological elastomers based on CNN-BiLSTM Attention/prediction_code/CNN_BiLSTM_Attention/loss_train_Fe.csv', mode='a',
                    header=False, index=False)  # mode设为a,就可以向csv文件追加数据了
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, 100, loss)

        losses.append(loss.item())  # 存储训练误差

        # 打印训练损失
        print('Epoch [{}/{}], Train Loss: {:.6f}'.format(epoch + 1, num_epochs, loss.item()))
    PATH = "mdoel.pth"
    torch.save(model.state_dict(), PATH)
    print("save success")

    # 测试阶段
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for features, labels in test_dataloader:
            features = features.to(device)
            labels = labels.to(device)
            # 前向传播
            # labels = torch.squeeze(labels, dim=1)  #为标签消去一个维度
            outputs = model(features)
            print(labels)
            print('Predicted Labels:', outputs)

            # 计算损失
            loss = criterion(outputs, labels)
            time = "%s" % datetime.now()  # 获取当前时间
            list = [time, i, loss.item()]  # 建立excel保存精度和损失
            data = pd.DataFrame([list])  # 建立excel保存精度和损失
            data.to_csv(
                'D:/Performance prediction of magnetorheological elastomers based on CNN-BiLSTM Attention/prediction_code/CNN_BiLSTM_Attention/loss_test_Fe.csv',mode='a',
                header=False, index=False)  # mode设为a,就可以向csv文件追加数据了

            total_loss += loss.item()

        # 打印测试损失
        avg_loss = total_loss / len(test_dataloader)
        print('Epoch [{}/{}], Test Loss: {:.6f}'.format(epoch + 1, num_epochs, avg_loss))





import matplotlib.pyplot as plt
# 使用训练好的模型预测剩余一个样本
model.eval()
with torch.no_grad():
    for features, labels in test_dataloader:
        features = features.to(device)
        labels = labels.to(device)
        predicted_labels = model(features)
        print(labels)
        print('Predicted Labels:', predicted_labels)
        # 绘图
        labels = torch.squeeze(labels, dim=0)  # 为标签消去一个维度
        predicted_labels = torch.squeeze(predicted_labels, dim=0)  # 为标签消去一个维度
        labels = labels.cpu().tolist()
        predicted_labels = predicted_labels.cpu().tolist()
        plt.plot(labels, label='True')
        plt.plot(predicted_labels, 'red', label='Predict')
        plt.legend()
        plt.show()


