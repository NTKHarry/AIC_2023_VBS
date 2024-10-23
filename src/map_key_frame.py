import pandas as pd
import numpy as np

def extract(img_name : str, path) -> int:
    df = pd.read_csv(path)
    number_str = img_name.split('.')[0]  # Extracts '001' by splitting at the dot
    number = int(number_str)  # Converts '001' to integer 1
    matching_row = df[df.iloc[:, 0] == number]
    return int(matching_row['frame_idx'].values[0])

# def extract_2(img_name : str, path) -> int:
#     df = pd.read_csv(path)
#     number_str = img_name.split('.')[0]  # Extracts '001' by splitting at the dot
#     number = int(number_str)  # Converts '001' to integer 1
#     matching_row = df[df.iloc[:, 0] == number]
#     return round(int(matching_row['pts_time'].values[0]))

def extract_2(img_name : str, path) -> int:
    df = pd.read_csv(path)
    number_str = img_name.split('.')[0]  # Extracts '001' by splitting at the dot
    number = int((int(number_str)-1)*50 // df.iloc[0,2])  # Converts '001' to integer 1
    #matching_row = df[df.iloc[:, 0] == number]
    #return round(int(matching_row['pts_time'].values[0]))
    return number

def extract_3(start, end, path):
    df = pd.read_csv(path)
    matching_row = df[(df.iloc[:,1] >= start) & (df.iloc[:,1] <= end)]
    selected_columns = matching_row[['n', 'frame_idx']].to_numpy()
    return selected_columns

def extract_3(start, end, path):
    df = pd.read_csv(path)
    matching_row = df[(df.iloc[:,1] >= start) & (df.iloc[:,1] <= end)]
    selected_columns = matching_row[['n', 'frame_idx']].to_numpy()
    return selected_columns

# path = r'D:\tfolder\codingFile\AIlearning\AIchallenge\AIC_EITA\datasets\map-keyframes\L01_V001.csv'
# img_name = '005.jpg'
# df = pd.read_csv(path)
# res = extract(img_name,df)

# print(res, type(res))

