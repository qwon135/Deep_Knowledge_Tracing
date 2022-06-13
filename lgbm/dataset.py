import pandas as pd

def custom_split(df, ratio=0.9, split=True):
    # users = list(zip(df['userID'].value_counts().index, df['userID'].value_counts()))
    # random.seed(42)
    # random.shuffle(users)
    
#     max_train_data_len = ratio*len(df)
#     sum_of_train_data = 0
#     user_ids =[]

#     for user_id, count in users:
#         sum_of_train_data += count
#         if max_train_data_len < sum_of_train_data:
#             break
#         user_ids.append(user_id)
    user_ids = pd.read_csv('/opt/ml/input/data/cv_train_data.csv').userID.unique()


    train = df[df['userID'].isin(user_ids)]
    test = df[df['userID'].isin(user_ids) == False]

    # test데이터셋은 각 유저의 마지막 interaction만 추출
    test = test[test['userID'] != test['userID'].shift(-1)]
    return train, test