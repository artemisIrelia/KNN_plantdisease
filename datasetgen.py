import main_model
import pandas as pd

files = ['./dataset/c1.jpg', './dataset/s1.jpg', './dataset/s2.jpg', './dataset/c2.jpg', './dataset/s6.jpg', './dataset/c3.jpg', './dataset/c4.jpg', './dataset/s5.jpg', './dataset/c5.jpg', './dataset/c6.jpg', './dataset/s3.jpg', './dataset/s4.jpg', './dataset/c7.jpg', './dataset/c8.jpg','./dataset/c9.jpg','./dataset/s7.jpg', './dataset/s8.jpg']


# 0 is circle 1 is square
output = [0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1]
dataset_final = []
count = 0
for f in files:
    result = main_model.disease(f)
    result += [output[count]]
    print result
    dataset_final += [result]
    count += 1


print dataset_final

dataset_final_pd = pd.DataFrame(dataset_final)
print dataset_final_pd
dataset_final_pd.to_csv("disease_dataset.csv", header=['filename', 'total_cnt', 'circ_avg', 'sqr_avg', 'disease'], index=False)
