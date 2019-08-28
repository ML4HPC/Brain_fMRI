import csv
import numpy as np

path = "/global/cscratch1/sd/yanzhang/data_brain/image03/"

train = "training_fluid_intelligenceV1.csv"
valid = "validation_fluid_intelligenceV1.csv"



def readcsv(path, filename):
    with open(path+filename) as csvfile:
        csv_dict = {}
        csv_data = csv.reader(csvfile, delimiter=',')
        next(csv_data)
        for row in csv_data:
            csv_dict[row[0]] = row[1]
    return csv_dict



csv_train = readcsv(path, train)
csv_valid = readcsv(path, valid)


for key, value in csv_train.items():
    print(key, value)

for key, value in csv_valid.items():
    print(key, value)



print('saving train dict!')
np.save('csv_train_target.npy', csv_train)
print('saving valid dict!')
np.save('csv_valid_target.npy', csv_valid)
