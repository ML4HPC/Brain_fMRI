import numpy as np

base_path = '/hpcgpfs01/scratch/seungwook/processed_norm/'
save_path = '/hpcgpfs01/scratch/seungwook/processed_norm_siemens/'

siemens_subjects = np.load(base_path + 'siemens_subjects.npy')
train_subjects = np.load(base_path + 'train_subjects.npy')
valid_subjects = np.load(base_path + 'valid_subjects.npy')
test_subjects = np.load(base_path + 'test_subjects.npy')

train_idx = []
valid_idx = []
test_idx = []

for i in range(len(train_subjects)):
    if train_subjects[i] in siemens_subjects:
        train_idx.append(i)

for i in range(len(valid_subjects)):
    if valid_subjects[i] in siemens_subjects:
        valid_idx.append(i)

for i in range(len(test_subjects)):
    if test_subjects[i] in siemens_subjects:
        test_idx.append(i)

print(len(train_idx))
print(len(valid_idx))
print(len(test_idx))

train_t1 = np.load(base_path + 'train_data_img_T1.npy')
train_t2 = np.load(base_path + 'train_data_img_T2.npy')
train_fa = np.load(base_path + 'train_data_img_FA.npy')
train_md = np.load(base_path + 'train_data_img_MD.npy')
train_rd = np.load(base_path + 'train_data_img_RD.npy')
train_ad = np.load(base_path + 'train_data_img_AD.npy')

valid_t1 = np.load(base_path + 'valid_data_img_T1.npy')
valid_t2 = np.load(base_path + 'valid_data_img_T2.npy')
valid_fa = np.load(base_path + 'valid_data_img_FA.npy')
valid_md = np.load(base_path + 'valid_data_img_MD.npy')
valid_rd = np.load(base_path + 'valid_data_img_RD.npy')
valid_ad = np.load(base_path + 'valid_data_img_AD.npy')

test_t1 = np.load(base_path + 'test_data_img_T1.npy')
test_t2 = np.load(base_path + 'test_data_img_T2.npy')
test_fa = np.load(base_path + 'test_data_img_FA.npy')
test_md = np.load(base_path + 'test_data_img_MD.npy')
test_rd = np.load(base_path + 'test_data_img_RD.npy')
test_ad = np.load(base_path + 'test_data_img_AD.npy')

train_siemens_t1 = train_t1[train_idx]
train_siemens_t2 = train_t2[train_idx]
train_siemens_fa = train_fa[train_idx]
train_siemens_md = train_md[train_idx]
train_siemens_rd = train_rd[train_idx]
train_siemens_ad = train_ad[train_idx]

valid_siemens_t1 = valid_t1[valid_idx]
valid_siemens_t2 = valid_t2[valid_idx]
valid_siemens_fa = valid_fa[valid_idx]
valid_siemens_md = valid_md[valid_idx]
valid_siemens_rd = valid_rd[valid_idx]
valid_siemens_ad = valid_ad[valid_idx]

test_siemens_t1 = test_t1[test_idx]
test_siemens_t2 = test_t2[test_idx]
test_siemens_fa = test_fa[test_idx]
test_siemens_md = test_md[test_idx]
test_siemens_rd = test_rd[test_idx]
test_siemens_ad = test_ad[test_idx]

train_target = np.load(base_path + 'train_data_target.npy')
valid_target = np.load(base_path + 'valid_data_target.npy')
test_target = np.load(base_path + 'test_data_target.npy')

train_target_siemens = train_target[train_idx]
valid_target_siemens = valid_target[valid_idx]
test_target_siemens = test_target[test_idx]

np.save(save_path + 'train_data_img_T1.npy', train_siemens_t1)
np.save(save_path + 'train_data_img_T2.npy', train_siemens_t2)
np.save(save_path + 'train_data_img_FA.npy', train_siemens_fa)
np.save(save_path + 'train_data_img_MD.npy', train_siemens_md)
np.save(save_path + 'train_data_img_RD.npy', train_siemens_rd)
np.save(save_path + 'train_data_img_AD.npy', train_siemens_ad)

np.save(save_path + 'valid_data_img_T1.npy', valid_siemens_t1)
np.save(save_path + 'valid_data_img_T2.npy', valid_siemens_t2)
np.save(save_path + 'valid_data_img_FA.npy', valid_siemens_fa)
np.save(save_path + 'valid_data_img_MD.npy', valid_siemens_md)
np.save(save_path + 'valid_data_img_RD.npy', valid_siemens_rd)
np.save(save_path + 'valid_data_img_AD.npy', valid_siemens_ad)

np.save(save_path + 'test_data_img_T1.npy', test_siemens_t1)
np.save(save_path + 'test_data_img_T2.npy', test_siemens_t2)
np.save(save_path + 'test_data_img_FA.npy', test_siemens_fa)
np.save(save_path + 'test_data_img_MD.npy', test_siemens_md)
np.save(save_path + 'test_data_img_RD.npy', test_siemens_rd)
np.save(save_path + 'test_data_img_AD.npy', test_siemens_ad)

np.save(save_path + 'train_data_target.npy', train_target_siemens)
np.save(save_path + 'valid_data_target.npy', valid_target_siemens)
np.save(save_path + 'test_data_target.npy', test_target_siemens)

