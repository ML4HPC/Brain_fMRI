import os
import csv
import numpy as np
import pandas as pd
import argparse
import IPython

# Global variables
train_csv = "training_fluid_intelligenceV1.csv"
valid_csv = "validation_fluid_intelligenceV1.csv"
acspsw_txt = "acspsw03.txt"
abcd_txt = "abcd_lt01.txt"
pdemo_txt = "pdem02.txt"

def get_covariates(key, acspsw_data, abcd_data, pdemo_data):
    abcd_row = abcd_data[(abcd_data['subjectkey']==key) & (abcd_data['eventname']=='baseline_year_1_arm_1')]
    acspsw_row = acspsw_data[(acspsw_data['subjectkey']==key) & (acspsw_data['eventname']=='baseline_year_1_arm_1')]
    pdemo_row = pdemo_data[(pdemo_data['subjectkey']==key) & (pdemo_data['eventname']=='baseline_year_1_arm_1')]
    
    age = int(acspsw_row['interview_age'])
    gender = 0 if (acspsw_row['gender'] == 'M').bool() else 1
    race_ethnicity = int(acspsw_row['race_ethnicity']) - 1

    # Renormalizing highest education range to [0, 22] (with 22 as unknown or null)
    high_edu = 0
    high_edu_prnt1 = int(pdemo_row['demo_prnt_ed_v2']) if not pdemo_row['demo_prnt_ed_v2'].isnull().bool() else 777
    high_edu_prnt2 = int(pdemo_row['demo_prtnr_ed_v2']) if not pdemo_row['demo_prtnr_ed_v2'].isnull().bool() else 777
    
    if (high_edu_prnt1 == 777 or high_edu_prnt1 == 999) and (high_edu_prnt2 == 777 or high_edu_prnt2 == 999):
        high_edu = 22
    elif high_edu_prnt1 == 777 or high_edu_prnt1 == 999:
        high_edu = high_edu_prnt2
    elif high_edu_prnt2 == 777 or high_edu_prnt2 == 999:
        high_edu = high_edu_prnt1
    else:
        high_edu = max(high_edu_prnt1, high_edu_prnt2)
    
    married = int(pdemo_row['demo_prnt_marital_v2'])
    
    if married == 777:
        married = 7
    
    married -= 1
    
    # Normalizing to range [0, 20] from [1, 21]
    site = int(abcd_row['site_id_l'].str.strip('site')) - 1

    assert (gender == 0 or gender == 1)
    assert (married >= 0 or married <= 6)
    assert (race_ethnicity >= 0 and race_ethnicity <= 4)
    assert (high_edu >= 0 and high_edu <= 22)
    assert (site >= 0 and site <= 21)

    return [age, gender, race_ethnicity, high_edu, married, site] 

def readcsv(path, filename):
    with open(path+filename) as csvfile:
        csv_dict = {}
        csv_data = csv.reader(csvfile, delimiter=',')
        next(csv_data)
        for row in csv_data:
            csv_dict[row[0]] = row[1]
    return csv_dict


def readtxt(path, csv_train, csv_valid):
    abcd_data = pd.read_csv(os.path.join(path, abcd_txt), sep='\t', header=0)
    acspsw_data = pd.read_csv(os.path.join(path, acspsw_txt), sep='\t', header=0)
    pdemo_data = pd.read_csv(os.path.join(path, pdemo_txt), sep='\t', header=0)
    
    acspsw_data = acspsw_data.drop(0)
    abcd_data = abcd_data.drop(0)
    pdemo_data = pdemo_data.drop(0)

    if 'NDAR_INVLDGEWALX' in csv_train:
        del csv_train['NDAR_INVLDGEWALX']
    if 'NDAR_INVLDGEWALX' in csv_valid:
        del csv_valid['NDAR_INVLDGEWALX']

    for key in csv_train.keys():
        covar = get_covariates(key, acspsw_data, abcd_data, pdemo_data)
        fluid_intel = float(csv_train[key])
        covar.insert(0, fluid_intel)
        csv_train[key] = covar
    
    for key in csv_valid.keys():
        covar = get_covariates(key, acspsw_data, abcd_data, pdemo_data)
        fluid_intel = float(csv_valid[key])
        covar.insert(0, fluid_intel)
        csv_valid[key] = covar
    
    return csv_train, csv_valid

# for key, value in csv_train.items():
#     print(key, value)

# for key, value in csv_valid.items():
#     print(key, value)

if __name__ == "__main__":
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Read csv + txt data')
    parser.add_argument('--path', help='Path to data directory')
    args = parser.parse_args()

    csv_train = readcsv(args.path, train_csv)
    csv_valid = readcsv(args.path, valid_csv)

    train_w_covar, valid_w_covar = readtxt(args.path, csv_train, csv_valid)

    print('Saving train dataset')
    np.save('csv_train_target.npy', train_w_covar)    
    print('Saving valid dataset')
    np.save('csv_valid_target.npy', valid_w_covar)    

