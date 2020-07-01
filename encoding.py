import csv

total_num_sc = 0
max_id = 0
num_exp = 0


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
