import os
import datetime
import torch
from torch import nn, optim
from torchvision import  transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from my_dataset import MyDataSet
from utils_datasets import read_split_data
import json
from ResNet50_based_on_OD import ResNet50_basedOD

# 参数设置
info_len = 7
batch_size = 100
epochs = 50
num_class = 4
train_p = 0.85
split_seed = 0
sampling_seed = 1

# 文件配置
# -------------------------------------------------- #
# 数据集文件夹位置
filepath = './datasets'
# 权重保存文件夹路径
savepath = './resnet_logs'
 
# 获取GPU设备
if torch.cuda.is_available(): 
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print("using {} device.".format(device))

# 构造数据集
data_transform = {
    "train": transforms.Compose([transforms.Resize((224,224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

# 数据集打包
train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(filepath, split_seed, 1-train_p)
train_size = len(train_images_label)
val_size = len(val_images_label)

json_dict = open(filepath+'/'+'OD_info.json', 'r')
OD_info_dict = json.load(json_dict)
train_dataset = MyDataSet(  images_path=train_images_path,
                            images_class=train_images_label,
                            OD_info_dict=OD_info_dict,
                            transform=data_transform["train"])

val_dataset = MyDataSet(  images_path=val_images_path,
                            images_class=val_images_label,
                            OD_info_dict=OD_info_dict,
                            transform=data_transform["val"])

# 数据平衡
# train_labels = [ train_dataset.dataset.targets[train_dataset.indices[i]] for i in range(len(train_dataset.indices))]
# class_list = np.unique(train_labels)
# class_count = np.zeros(len(class_list))
# for i in range(len(class_list)):
#     class_count[i] = train_labels.count(class_list[i])
# total_train_samples_num = np.sum(class_count)
# samples_p = np.zeros(len(train_labels))
# for i in range(len(train_labels)):
#     samples_p[i] = class_count[train_labels[i]]/total_train_samples_num
# samples_weight = np.array(1./samples_p)
# print(train_labels)
# print(samples_weight)
# weight_sampler = WeightedRandomSampler(torch.DoubleTensor(samples_weight),num_samples=len(train_labels),replacement=True)

# 构造训练集
torch.manual_seed(sampling_seed)
train_loader = DataLoader(dataset=train_dataset,  # 接收训练集
                          batch_size=batch_size,  # 训练时每个step处理32张图
                          shuffle=True,
                          #sampler=weight_sampler, # WeightedRandomSampler        
                          num_workers=0,          # 加载数据时的线程数量，windows环境下只能=0
                          collate_fn=train_dataset.collate_fn)
 
# 构造验证集
val_loader = DataLoader(dataset=val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=0,
                        collate_fn=val_dataset.collate_fn)

# num0 = 0
# num1 = 0
# num2 = 0
# num3 = 0
# for step, data in enumerate(train_loader):
#     images, labels = data
#     print(labels)
#     num0 += list(labels).count(0)
#     num1 += list(labels).count(1)
#     num2 += list(labels).count(2)
#     num3 += list(labels).count(3)
# print(num0)
# print(num1)
# print(num2)
# print(num3)

net = ResNet50_basedOD(num_class=num_class,info_len=info_len)
net.load_weight('./pretrained_weights/resnet50-0676ba61.pth')
for name, param in net.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False

net.to(device)
# 定义交叉熵损失
loss_function = nn.CrossEntropyLoss()
# 定义优化器
optimizer = optim.Adam(net.parameters())
# 保存准确率最高的一次迭代
best_acc = 0.0

# 模型训练
time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
log_dir         = os.path.join(savepath, "loss_" + str(time_str))
os.makedirs(log_dir)

for epoch in range(epochs):
    print('-'*30, '\n', 'epoch:', epoch)
    # 训练
    net.train()
    # 计算训练一个epoch的总损失
    running_loss = 0.0
    # 每个step训练一个batch
    for step, data in enumerate(train_loader):
        # data中包含图像及其对应的标签
        images, infos, labels = data
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        outputs = net(images.to(device),infos.to(device))
        # 计算预测值和真实值的交叉熵损失
        loss = loss_function(outputs, labels.to(device))
        # 梯度计算
        loss.backward()
        # 权重更新
        optimizer.step()
        # 累加每个step的损失
        running_loss += loss.item()
        # 打印每个step的损失
        print(f'step:{step} loss:{loss}')
    
# 网络验证
    net.eval()  # 验证模型
    acc = 0.0   # 验证集准确率
    train_acc = 0.0
    val_loss = 0.0
    with torch.no_grad(): 
        # 每次验证一个batch
        for data_train in train_loader:
            train_images, train_infos, train_labels = data_train
            outputs = net(train_images.to(device),train_infos.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            train_acc += (predict_y == train_labels.to(device)).sum().item()
        train_acc = train_acc/train_size
        # 写入
        with open(os.path.join(log_dir, 'train_acc.txt'),'a+') as f:
            f.write(str(train_acc)+'\n')

        for data_test in val_loader:
            # 获取验证集的图片和标签
            test_images, test_infos, test_labels = data_test
            # 前向传播
            outputs = net(test_images.to(device),test_infos.to(device))
            loss = loss_function(outputs, test_labels.to(device))
            val_loss += loss.item()
            # 预测分数的最大值
            predict_y = torch.max(outputs, dim=1)[1]
            # 累加每个step的准确率
            acc += (predict_y == test_labels.to(device)).sum().item()
        # 计算所有图片的平均准确率
        acc_test = acc / val_size
        with open(os.path.join(log_dir, 'val_acc.txt'),'a+') as f:
            f.write(str(acc_test)+'\n')
        with open(os.path.join(log_dir, 'train_loss.txt'),'a+') as f:
            f.write(str(running_loss/step)+'\n')
        with open(os.path.join(log_dir, 'val_loss.txt'),'a+') as f:
            f.write(str(val_loss/step)+'\n')
        
        # 打印每个epoch的训练损失和验证准确率
        print(f'total_train_loss:{running_loss/step}, total_val_loss:{val_loss/step},total_train_acc:{train_acc},total_test_acc:{acc_test}')
        
# 权重保存
        # 保存最好的准确率的权重
        if acc_test > best_acc:
            # 更新最佳的准确率
            best_acc = acc_test
            # 保存的权重名称
            savename = os.path.join(savepath, 'best_resnet50_OD.pth')
            # 保存当前权重
            torch.save(net.state_dict(), savename)
        # 保存的权重名称
        savename = os.path.join(savepath, 'epoch_'+str(epoch)+'_resnet50_OD.pth')
        # 保存当前权重e
        torch.save(net.state_dict(), savename)
        
