import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
# train_data=pd.read_csv(r'C:\Users\蒋思清\Desktop\d pd sta\data\train_trts0.7.csv').values
# test_data=pd.read_csv(r'C:\Users\蒋思清\Desktop\d pd sta\data\test_trts0.7.csv').values

#train_data=pd.read_csv(r'/data/sjd/d/p_d/stomach/data/train_trts0.7.csv').values
train_data=pd.read_csv(r'/home/jiangsiqing/ml/train_clean.csv').values
test_data=pd.read_csv(r'/data/sjd/d/p_d/stomach/data/test_trts0.7.csv').values
# print(train_data,train_data.shape)
X_train=train_data[:,:-1]
xlist=[]
for i in X_train:
    i=torch.from_numpy(i)
    xlist.append(i)
x_train_torchlist=xlist
# print(xlist)
y_train=train_data[:,-1]
list=[]
for i in y_train:
    i=int(i)-1
    a=torch.zeros(4)
    a[i]=1
    list.append(a)
y_train_torchlist=list
# print(list)
#
# for i,data in enumerate(X_train):
#     list.append(data)

# print(list[0])
class stomach_dataset(nn.Module):
    def __init__(self):
        super(stomach_dataset, self).__init__()
        self.xtrain=x_train_torchlist
        self.ytrain=y_train_torchlist

    def __getitem__(self, index):
        xtrain=self.xtrain[index]
        ytrain=self.ytrain[index]
        return xtrain,ytrain

    def __len__(self):
        return len(self.xtrain)

def calc_f1(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return f1_score(y_true, y_pre)

def get_acc(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return accuracy_score(y_true, y_pre)



def get_recall(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return recall_score(y_true, y_pre)



def get_pre(y_true, y_pre, threshold=0.5):
    y_true = y_true.view(-1).cpu().detach().numpy().astype(np.int)
    y_pre = y_pre.view(-1).cpu().detach().numpy() > threshold
    return precision_score(y_true, y_pre)
# for x,y in a:
#     print(x.shape,y.shape)
if __name__ =='__main__':
    import models
    import tqdm
    import torch.optim as optim
    a = stomach_dataset()
    train_dataloader=DataLoader(dataset=a,batch_size=5,shuffle=True,num_workers=2)
    for x,y in train_dataloader:
        print(x,y)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=getattr(models,'resnet34')()
    model=model.to(device)
    for epoch in range(1,500):
        step=0
        tq=tqdm.tqdm(total=len(train_dataloader))
        tq.set_description('epoch{}'.format(epoch))
        F1,Acc,Pre,Recall=0,0,0,0
        for i,(input,targets) in enumerate(train_dataloader):
            tq.update(1)
            input=input.view(-1,81,1)
            input = input.type(torch.FloatTensor)
            targets = targets.type(torch.FloatTensor)
            critertion=nn.BCEWithLogitsLoss()
            lr=0.0001
            opt = optim.Adam(model.parameters(), lr)
            opt.zero_grad()
            input=input.to(device)
            targets=targets.to(device)
            output=model(input)
            loss=critertion(output,targets)
            loss.backward()
            step+=1
            F1+=calc_f1(targets,output)
            f1 = F1 / step
            Acc += get_acc(targets, output)
            acc = Acc / step
            Recall += get_recall(targets, output)
            recall = Recall / step
            Pre += get_pre(targets, output)
            pre = Pre / step
            if i != 0 and i % 10 == 0:
                print('train_loss(epoch=%i,step=%i):' % (epoch, i), str(loss).split('(')[1].split(',')[0], '||',
                      'loss_mean_to step=%i:' % i,'f1:', f1, '||', 'acc:', acc, '||', 'recall:', recall, '||', 'pre:', pre)
        tq.close()
