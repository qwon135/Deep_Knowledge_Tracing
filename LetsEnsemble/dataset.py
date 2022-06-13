import os 

import pandas as pd
import random
import numpy as np
dtype = {
    'userID': 'int16',
    'answerCode': 'int8',
    'KnowledgeTag': 'int16'
}

def load_dataframe(basepath, filename):
    return pd.read_csv(os.path.join(basepath, filename), dtype=dtype, parse_dates=['Timestamp'])

def add_head_term(df):
    import math
    # result = []
    # from collections import defaultdict
    # for u in df.userID.unique():
    #     last_head = {}
    #     prev_t = 0
    #     prev_h = 0
    #     for h,t in zip(df.i_head[df.userID==u], df.Timestamp[df.userID==u].view('int64')//1e9):
    #         t = int(t)

    #         if h != prev_h: # head가 변경되면
    #             last_head[prev_h] = prev_t # 기록 해줌
            
    #         if h in last_head: # 이전에 풀어본 head면
    #             result.append(t - last_head[h])
    #         else: # 처음 풀어보는 head 면
    #             result.append(np.NaN)
    #         prev_t = t
    #         prev_h = h
            
    # df['head_term'] = result
    gdf = df[['userID','testId','i_tail','i_head','Timestamp']].sort_values(by=['userID','i_head','Timestamp'])
    gdf['b_userID'] = gdf['userID'] != gdf['userID'].shift(1)
    gdf['b_i_head'] = gdf['i_head'] != gdf['i_head'].shift(1)
    gdf['first'] = gdf[['b_userID','b_i_head']].any(axis=1).apply(lambda x : 1- int(x))
    gdf['head_term'] = gdf['Timestamp'].diff().fillna(pd.Timedelta(seconds=0)) 
    gdf['head_term'] = gdf['head_term'].apply(lambda x: x.total_seconds()) * gdf['first']
    df['head_term'] = gdf['head_term'].apply(lambda x : math.log(x+1))

    return df

def ELO_function (df) :
    def get_new_theta(is_good_answer, beta, left_asymptote, theta, nb_previous_answers):
        return theta + learning_rate_theta(nb_previous_answers) * (
            is_good_answer - probability_of_good_answer(theta, beta, left_asymptote)
        )

    def get_new_beta(is_good_answer, beta, left_asymptote, theta, nb_previous_answers):
        return beta - learning_rate_beta(nb_previous_answers) * (
            is_good_answer - probability_of_good_answer(theta, beta, left_asymptote)
        )

    def learning_rate_theta(nb_answers):
        return max(0.3 / (1 + 0.01 * nb_answers), 0.04)

    def learning_rate_beta(nb_answers):
        return 1 / (1 + 0.05 * nb_answers)

    def probability_of_good_answer(theta, beta, left_asymptote):
        return left_asymptote + (1 - left_asymptote) * sigmoid(theta - beta)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def estimate_parameters(answers_df, granularity_feature_name='assessmentItemID'):
        item_parameters = {
            granularity_feature_value: {"beta": 0, "nb_answers": 0}
            for granularity_feature_value in np.unique(answers_df[granularity_feature_name])
        }
        student_parameters = {
            student_id: {"theta": 0, "nb_answers": 0}
            for student_id in np.unique(answers_df.userID)
        }

        # print("Parameter estimation is starting...")

        for student_id, item_id, left_asymptote, answered_correctly in (
            zip(answers_df.userID.values, answers_df[granularity_feature_name].values, answers_df.left_asymptote.values, answers_df.answerCode.values)
        ):
            theta = student_parameters[student_id]["theta"]
            beta = item_parameters[item_id]["beta"]

            item_parameters[item_id]["beta"] = get_new_beta(
                answered_correctly, beta, left_asymptote, theta, item_parameters[item_id]["nb_answers"],
            )
            student_parameters[student_id]["theta"] = get_new_theta(
                answered_correctly, beta, left_asymptote, theta, student_parameters[student_id]["nb_answers"],
            )

            item_parameters[item_id]["nb_answers"] += 1
            student_parameters[student_id]["nb_answers"] += 1

        # print(f"Theta & beta estimations on {granularity_feature_name} are completed.")
        return student_parameters, item_parameters
    
    def gou_func (theta, beta) :
        return 1 / (1 + np.exp(-(theta - beta)))
    
    
    df['left_asymptote'] = 0

    # print(f"Dataset of shape {df.shape}")
    # print(f"Columns are {list(df.columns)}")

    student_parameters, item_parameters = estimate_parameters(df)
    
    prob = [gou_func(student_parameters[student]['theta'], item_parameters[item]['beta']) for student, item in zip(df.userID.values, df.assessmentItemID.values)]
    
    df['elo_prob'] = prob
    
    return df

def add_col(df):
    pre = df["testId"][0]
    count = df["answerCode"][0]
    c = 1
    new = []

    for idx, answer in zip(df["testId"],df["answerCode"]):
        if pre != idx :
            pre = idx
            new.append(0)
            c = 1
            count = answer
        else :
            new.append(count/c)
            c += 1
            count += answer
    df['cum_correct'] = new
    return df

def add_last_problem(df):
    new = []
    pre = df['testId'][0]
    for idx in df['testId']:
        if pre != idx :
            new[-1]=-1
            pre = idx
        new.append(0)
    df['last_problem'] = new
    return df


def feature_engineering(df):
    
    #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    df.sort_values(by=['userID','Timestamp'], inplace=True)
    
    #유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
    df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
    df['user_acc'] = df['user_correct_answer']/df['user_total_answer']
    
    df['hour'] = df['Timestamp'].dt.hour
    df['dow'] = df['Timestamp'].dt.dayofweek

    # diff()를 이용하여 시간 차이를 구해줍니다
    diff = df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))
    
    # 만약 0초만에 풀었으면 0으로 치환
    diff = diff.fillna(pd.Timedelta(seconds=0))
    
    # 시간을 전부 초단위로 변경합니다.
    diff = diff['Timestamp'].apply(lambda x: x.total_seconds())

    # df에 elapsed(문제 풀이 시간)을 추가해줍니다.
    df['t_elapsed'] = diff
    
    # 문제 풀이 시간이 650초 이상은 이상치로 판단하고 제거합니다.
    df['t_elapsed'] = df['t_elapsed'].apply(lambda x : x if x <650 else None)
    
    # 대분류(앞 세자리)
    df['i_head']=df['testId'].apply(lambda x : int(x[1:4])//10)

    # 중분류(중간 세자리)
    df['i_mid'] = df['testId'].apply(lambda x : int(x[-3:]))

    # 문제 번호(분류를 제외한)
    df['i_tail'] = df['assessmentItemID'].apply(lambda x : int(x[-3:]))

    df = add_col(df)
    df = add_last_problem(df)
    df = add_head_term(df)
    df = ELO_function(df)
    # 유저 피쳐    
    user_feature = df.groupby(['userID','i_head']).agg({
    'answerCode':['mean', 'count'],
    't_elapsed':['mean']
    })
    user_feature.reset_index(inplace=True)
    user_feature.columns = ["userID","i_head","u_head_mean","u_head_count", "u_head_elapsed"]
    
    # 시험지 피쳐
    len_seq = lambda x : len(set(x))

    testId_feature = df.groupby(['testId']).agg({
        't_elapsed': 'mean',
        'answerCode':['mean', 'sum'],
        'i_tail':'max',
        'KnowledgeTag':len_seq
    })
    testId_feature.reset_index(inplace=True)
    testId_feature['i_head']=testId_feature['testId'].apply(lambda x : int(x[1:4])//10)
    testId_feature['i_mid']=testId_feature['testId'].apply(lambda x : int(x[-3:]))
    testId_feature.columns = ['testId','i_mid_elapsed','i_mid_mean','i_mid_sum' ,'i_mid_count', 'i_mid_tag_count', 'i_head', 'i_mid']
    testId_feature = testId_feature[['testId','i_mid_elapsed','i_mid_mean','i_mid_sum' ,'i_mid_count', 'i_mid_tag_count']]

    df['pkt'] = df.groupby(['userID','KnowledgeTag']).cumcount()

    # 태그 피쳐
    knowLedgedTag_acc = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum'])
    knowLedgedTag_acc.columns = ["tag_mean", 'tag_sum']
    
    df = pd.merge(df, user_feature, on=['userID', 'i_head'], how="left")
    df = pd.merge(df, testId_feature, on=['testId'], how="left")
    df = pd.merge(df, knowLedgedTag_acc, on=['KnowledgeTag'], how="left")
    
    return df


def custom_train_test_split(df, ratio=0.9, split=True):
#     users = list(zip(df['userID'].value_counts().index, df['userID'].value_counts()))
#     random.seed(42)
#     random.shuffle(users)
    
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


def make_dataset(train, test):
    # 사용할 Feature 설정

    # X, y 값 분리
    y_train = train['answerCode']
    train = train.drop(['answerCode'], axis=1)

    y_test = test['answerCode']
    test = test.drop(['answerCode'], axis=1)

    return y_train, train, y_test, test