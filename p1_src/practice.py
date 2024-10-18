from glob import glob
import os
import pandas as pd
import timm
import torch
from torchvision import models

resnet = models.resnet50(weights=None)
torch.save(resnet.state_dict(), './resnet_weight_none_backbone.pt')

# print(glob('../hw1_data/p1_data/office/train/*'))

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
print(parent_dir)

data_dir = os.path.join(parent_dir, 'hw1_data', 'p1_data', 'office', 'train.csv')
print(data_dir)


df = pd.read_csv(data_dir)

# for i in df['filename']:
#     print(i)
# # print(df['filename'])
# print(df.filename)

data = list(zip(df.filename, df.label))
data_tuple = tuple(data)
print(f"\n数据总数: {len(data)}")
print(f"data_tuple 的类型: {type(data_tuple)}")
print(f"data_tuple 中第一个元素的类型: {type(data_tuple[0])}")
print(f"data_tuple 中第一个元素的第一项: {data_tuple[0][0]}")



# print(timm.list_models('resnet50*'))
print(models.resnet50(weights=None))

# for (i,j) in zip(df.filename, df.label):
#     print(i,j)
#     image.append(i)
#     label.append(j)

# print(image)
# print(label)


