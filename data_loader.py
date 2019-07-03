import numpy as np

train_target = np.load('csv_train_target.npy')
valid_target = np.load('csv_valid_target.npy')

train_img = np.load('train_img.npy')
valid_img = np.load('valid_img.npy')
test_img = np.load('test_img.npy')

train_data_img = []
valid_data_img = []
train_data_target = []
valid_data_target = []


#for i in range(len(train_target.items())):
    #filename = train_target.item().keys()[0]


for key in train_target.item().keys():
    if key in train_img.item().keys():
        train_data_img.append(train_img.item()[key])
        train_data_target.append(np.float(train_target.item()[key]))
        #train_data[key] = [train_img.item()[key], train_target.item()[key]]
        #print(train_data[key])
        #print(train_data[key][0].shape)



for key in valid_target.item().keys():
    if key in valid_img.item().keys():
           valid_data_img.append(valid_img.item()[key])
           valid_data_target.append(np.float(valid_target.item()[key]))
        #valid_data[key] = [valid_img.item()[key], valid_target.item()[key]]
        #print(valid_data[key])
        #print(valid_data[key][0].shape)

'''
for i in range(len(train_data_target)):
    train_data_img[i] = np.array(train_data_img[i].dataobj)
    train_data_target[i] = np.float(train_data_target[i])

for i in range(len(valid_data_target)):
    valid_data_img[i] = np.array(valid_data_img[i].dataobj)
    valid_data_target[i] = np.float(valid_data_target[i])
'''



np.save('train_data_img.npy', train_data_img_4d)
np.save('valid_data_img.npy', valid_data_img_4d)
np.save('train_data_target.npy', train_data_target_4d)
np.save('valid_data_target.npy', valid_data_target_4d)




