import csv
from sklearn.preprocessing import MultiLabelBinarizer

total_num_sc = 0
max_id = 0
num_exp = 0

# 1. get the unique id list
print('Get the unique id list')
id_list = []
with open('data/LOG.csv', encoding='utf-8') as f:
    f_csv = csv.reader(f)
    
    for row in f_csv:
        for element in row:
            if element not in id_list and element != '' and element != '\ufeff326':
                id_list.append(int(element))
id_list = sorted(id_list)
print('The length of id list is {}'.format(len(id_list)))
print('The Min and Max of id list are {} and {}.'.format(min(id_list), max(id_list)))
exit()

# Multi-Label Binarizer
mlb = MultiLabelBinarizer()
with open('data/LOG.csv', encoding='utf-8') as f:
    f_csv = csv.reader(f)
    mlb.fit_transform(f_csv)

print(mlb.classes_)
exit()


# Statistics
with open('data/LOG.csv', encoding='utf-8') as f:
    f_csv = csv.reader(f)
    
    for row in f_csv:
        total_num_sc += len(row)
        num_exp += 1
        if int(max(row)) > max_id:
            max_id = max(row)

percentage_actived_sc = total_num_sc / num_exp / max_id

print("Maximun Id of Scan Chains: {}".format(max_id))
print("Percentage of Activated Scan Chains: {}".format(percentage_actived_sc))
print("Total number of example: {}".format(num_exp))

# Conversion
# I should convert each row into a vector



# Encoding
