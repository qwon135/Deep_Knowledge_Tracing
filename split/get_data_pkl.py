from copyreg import pickle
import os

import pandas as pd
import random
import lightgbm as lgb
import numpy as np
dtype = {
    'userID': 'int16',
    # 'answerCode': 'int',
    'KnowledgeTag': 'int16'
}

'''

데이터 로드 속도를 줄이기 위해 pickle파일로 변환해서 저장합니다.

피쳐가 추가될 때 마다 피클 파일을 지우고 새로 실행해주세요!

'''

def feature_engineering(csv_data_path):    
    # 1. train/valid/test pkl 파일이 저장 될 경로
    pickle_path = '/opt/ml/level2-dkt-level2-recsys-08/data_pkl'
    
    if not os.path.isdir(os.path.join(pickle_path)):            
        os.mkdir(os.path.join(pickle_path))
        print('/opt/ml/level2-dkt-level2-recsys-08/data_pkl 위치에 폴더 생성!')    
    
    def add_head_term(df):
        import math
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
    #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    df = pd.read_csv(csv_data_path , dtype=dtype, parse_dates=['Timestamp'])
    df.sort_values(by=['userID','Timestamp'], inplace=True)

    print('피쳐 추가 시작')
    # 2. 피쳐 추가
    df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
    df['user_acc'] = df['user_correct_answer']/df['user_total_answer']
    

    df['hour'] = df['Timestamp'].dt.hour
    df['dow'] = df['Timestamp'].dt.dayofweek
    
    diff = df.loc[:, ['userID', 'Timestamp']].groupby('userID').diff().fillna(pd.Timedelta(seconds=0))

    diff = diff.fillna(pd.Timedelta(seconds=0))            
    diff = diff['Timestamp'].apply(lambda x: x.total_seconds())

    df['t_elapsed'] = diff            
    df['t_elapsed'] = df['t_elapsed'].apply(lambda x : x if x <650 else None)            
    df['i_head']=df['testId'].apply(lambda x : int(x[1:4])//10)            
    df['i_mid'] = df['testId'].apply(lambda x : int(x[-3:]))            
    df['i_tail'] = df['assessmentItemID'].apply(lambda x : int(x[-3:]))

    df = add_col(df)
    df = add_last_problem(df)
    df = add_head_term(df)
    df = ELO_function(df)
    
    # assementItemID
    assessmentID_feature = df.groupby(['assessmentItemID'])['answerCode'].agg(['mean', 'sum', 'std'])
    assessmentID_feature.columns = ["assessment_mean", 'assessment_sum', 'assessment_std']

    # head 피쳐    
    user_feature = df.groupby(['userID','i_head']).agg({
        'answerCode':[
                'mean',
                'count',
                'std'],
    't_elapsed':['mean']
    })
    user_feature.reset_index(inplace=True)
    user_feature.columns = ["userID","i_head","u_head_mean","u_head_count", "u_head_std","u_head_elapsed"]

    # testId 피쳐
    len_seq = lambda x : len(set(x))

    testId_feature = df.groupby(['testId']).agg({'t_elapsed': 'mean','answerCode':['mean','std','sum'], 'i_tail':'max', 'KnowledgeTag' : len_seq})
    
    testId_feature.reset_index(inplace=True)
    testId_feature['i_head']=testId_feature['testId'].apply(lambda x : int(x[1:4])//10)
    testId_feature['i_mid']=testId_feature['testId'].apply(lambda x : int(x[-3:]))
    testId_feature.columns = ['testId','i_mid_elapsed','i_mid_mean', 'i_mid_std','i_mid_sum', 'i_mid_count', 'i_mid_tag_count', 'i_head', 'i_mid']
    testId_feature = testId_feature[['testId','i_mid_elapsed','i_mid_mean','i_mid_std', 'i_mid_sum' ,'i_mid_count', 'i_mid_tag_count']]
    
    # Tag
    KnowledgeTag_feature = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum', 'std'])
    KnowledgeTag_feature.columns = ["tag_mean", 'tag_sum', 'tag_std']
    df['pkt'] = df.groupby(['userID','KnowledgeTag']).cumcount()

    # tail
    tail_feature = df.groupby(['i_tail'])['answerCode'].agg(['mean', 'sum', 'std'])
    tail_feature.columns = ["tail_mean", 'tail_sum', 'tail_std']

    # hour
    hour_feature = df.groupby(['hour'])['answerCode'].agg(['mean', 'sum', 'std'])
    hour_feature.columns = ["hour_mean", 'hour_sum', 'hour_std']
    
    # dow
    dow_feature = df.groupby(['dow'])['answerCode'].agg(['mean', 'sum', 'std'])
    dow_feature.columns = ["dow_mean", 'dow_sum', 'dow_std']


    df = pd.merge(df, user_feature, on=['userID', 'i_head'], how="left")
    df = pd.merge(df, testId_feature, on=['testId'], how="left")
    df = pd.merge(df, assessmentID_feature, on=['assessmentItemID'], how="left")
    df = pd.merge(df, KnowledgeTag_feature, on=['KnowledgeTag'], how="left")
    df = pd.merge(df, tail_feature, on=['i_tail'], how="left")
    df = pd.merge(df, hour_feature, on=['hour'], how="left")
    df = pd.merge(df, dow_feature, on=['dow'], how="left")

    O = df[df['answerCode']==1]
    X = df[df['answerCode']==0]
    
    elp_k = df.groupby(['KnowledgeTag'])['t_elapsed'].agg('mean').reset_index()
    elp_k.columns = ['KnowledgeTag',"tag_elapsed"]
    elp_k_o = O.groupby(['KnowledgeTag'])['t_elapsed'].agg('mean').reset_index()
    elp_k_o.columns = ['KnowledgeTag', "tag_elapsed_o"]
    elp_k_x = X.groupby(['KnowledgeTag'])['t_elapsed'].agg('mean').reset_index()
    elp_k_x.columns = ['KnowledgeTag', "tag_elapsed_x"]
    
    df = pd.merge(df, elp_k, on=['KnowledgeTag'], how="left")
    df = pd.merge(df, elp_k_o, on=['KnowledgeTag'], how="left")
    df = pd.merge(df, elp_k_x, on=['KnowledgeTag'], how="left")

    ass_k = df.groupby(['assessmentItemID'])['t_elapsed'].agg('mean').reset_index()
    ass_k.columns = ['assessmentItemID',"assessment_elapsed"]
    ass_k_o = O.groupby(['assessmentItemID'])['t_elapsed'].agg('mean').reset_index()
    ass_k_o.columns = ['assessmentItemID',"assessment_elapsed_o"]
    ass_k_x = X.groupby(['assessmentItemID'])['t_elapsed'].agg('mean').reset_index()
    ass_k_x.columns = ['assessmentItemID',"assessment_elapsed_x"]

    df = pd.merge(df, ass_k, on=['assessmentItemID'], how="left")
    df = pd.merge(df, ass_k_o, on=['assessmentItemID'], how="left")
    df = pd.merge(df, ass_k_x, on=['assessmentItemID'], how="left")

    prb_k = df.groupby(['i_tail'])['t_elapsed'].agg('mean').reset_index()
    prb_k.columns = ['i_tail',"tail_elapsed"]
    prb_k_o = O.groupby(['i_tail'])['t_elapsed'].agg('mean').reset_index()
    prb_k_o.columns = ['i_tail',"tail_elapsed_o"]
    prb_k_x = X.groupby(['i_tail'])['t_elapsed'].agg('mean').reset_index()
    prb_k_x.columns = ['i_tail',"tail_elapsed_x"]

    df = pd.merge(df, prb_k, on=['i_tail'], how="left")
    df = pd.merge(df, prb_k_o, on=['i_tail'], how="left")
    df = pd.merge(df, prb_k_x, on=['i_tail'], how="left")


    df = df.fillna(0)

    file_name = csv_data_path.split('/')[-1].replace('.csv','.pkl')
    
    save_path = os.path.join(pickle_path, file_name)
    
    df.to_pickle(save_path)
    if os.path.exists(save_path):
        print(f'{save_path} 저장 완료!')

if __name__ == "__main__":
    # # all data
    # feature_engineering(csv_data_path='/opt/ml/input/data/all.csv')

    # # train data
    # feature_engineering(csv_data_path='/opt/ml/input/data/train_data.csv')
    
    # # valid data
    # feature_engineering(csv_data_path='/opt/ml/input/data/cv_valid_data.csv')

    # # test data 이상함?
    # feature_engineering(csv_data_path='/opt/ml/input/data/test_data.csv')

    # test data -1 마지막 -1 값만
    feature_engineering(csv_data_path='/opt/ml/input/data/all-1.csv')
    