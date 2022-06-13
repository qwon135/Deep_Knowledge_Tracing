import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.preprocessing import LabelEncoder

import os 

import pandas as pd
import random
import numpy as np



class Preprocess:

    def __init__(self, args):
        self.dtype = {
            'userID': 'int16',
            'answerCode': 'int8',
            'KnowledgeTag': 'int16'
        }
        self.args = args
        self.train_data = None
        self.test_data = None
                
        self.file_name = args.file_name
        self.test_file_name = args.test_file_name
    def get_train_data(self): # train 데이터 리턴해줌
        return self.train_data

    def get_test_data(self): # test 데이터 리턴해줌
        return self.test_data

    def split_data(self, data, ratio=0.7, shuffle=True, seed=0): # split은 랜덤으로 7대3
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size] #쪼개기
        data_2 = data[size:]

        # data_1 = data[data['userID'].isin(pd.read_csv('/opt/ml/input/data/cv_train_data.csv').userID.unique())]
        # data_2 = data[data['userID'].isin(pd.read_csv('/opt/ml/input/data/cv_valid_data.csv').userID.unique())]

        return data_1, data_2 # 쪼개진 데이터 리턴

    def __save_labels(self, encoder, name): # 라벨을 저장(뭔진 잘몰겟음)
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy") 
        np.save(le_path, encoder.classes_)    
        
        
        
    def __preprocessing(self, df, is_train=True): # 전처리 파트 <- 여기 넣기..? 
        cate_cols = self.args.cate_col
        # cate_cols = ["assessmentItemID", "testId", "KnowledgeTag"]

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        for col in cate_cols:

            le = LabelEncoder()
            if is_train:
                # train에 안 나올 수 있는 class를 위해 unknown 클래스 추가
                a = df[col].unique().tolist() + ["unknown"]  # ["A010001001",...,"unknown"]
                le.fit(a)  # fit() : 실제로 label encoding 진행 (mapping)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(lambda x: x if str(x) in le.classes_ else "unknown")

            # 얘는 str이 아니면 무조건 걸러버림
            # 모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])  # encoded label을 return
            df[col] = test

        # def convert_time(s):  
        #     timestamp = time.mktime(
        #         datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
        #     )
        #     return int(timestamp)

        # df["Timestamp"] = df["Timestamp"].apply(convert_time)

        return df

    def __feature_engineering(self, df, is_train = True):
        
        pickle_path = '/opt/ml/level2-dkt-level2-recsys-08/data_pkl'
        if not os.path.isdir(os.path.join(pickle_path)):            
            os.mkdir(os.path.join(pickle_path))
        
        if is_train and os.path.exists(os.path.join(pickle_path, self.file_name[:-3]+'pkl')):
            print(f'{self.file_name[:-3]}pkl을 불러옵니다!')
            return pd.read_pickle(os.path.join(pickle_path, self.file_name[:-3]+'pkl'))
        
        if not is_train and os.path.exists(os.path.join(pickle_path, self.test_file_name[:-3]+'pkl')):
            print(f'{self.test_file_name[:-3]}pkl을 불러옵니다!')
            return pd.read_pickle(os.path.join(pickle_path, self.test_file_name[:-3]+'pkl'))

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

        testId_feature = df.groupby(['testId']).agg({
            't_elapsed': 'mean',
            'answerCode':[
                        'mean',
                        'std',
                        'sum'],
            'i_tail':'max',
            'KnowledgeTag':len_seq
        })
        
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

        df = df.fillna(0)

        if is_train:
            print(f'{self.file_name[:-3]}pkl을 저장합니다!')
            df.to_pickle(os.path.join(pickle_path, self.file_name[:-3]+'pkl'))
        else:
            print(f'{self.test_file_name[:-3]}pkl을 저장합니다!')
            df.to_pickle(os.path.join(pickle_path, self.test_file_name[:-3]+'pkl'))        

        return df

    def load_data_from_file(self, file_name, is_train=True):        
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path, dtype=self.dtype, parse_dates=['Timestamp'])

        df = self.__feature_engineering(df, is_train)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할 때 사용
        self.args.n_questions = len(np.load(os.path.join(self.args.asset_dir, "assessmentItemID_classes.npy")))
        self.args.n_test = len(np.load(os.path.join(self.args.asset_dir, "testId_classes.npy")))
        self.args.n_tag = len(np.load(os.path.join(self.args.asset_dir, "KnowledgeTag_classes.npy")))
        self.args.n_head = len(np.load(os.path.join(self.args.asset_dir, "i_head_classes.npy")))  ### 추가한 부분 


        self.args.n_head= len(
            np.load(os.path.join(self.args.asset_dir, "i_head_classes.npy"))
        )


        # self.args.n_mid= len(
        #     np.load(os.path.join(self.args.asset_dir, "i_mid_classes.npy"))
        # )

        self.args.n_tail = len(
            np.load(os.path.join(self.args.asset_dir, "i_tail_classes.npy"))
        )

        self.args.n_hour= len(
            np.load(os.path.join(self.args.asset_dir, "hour_classes.npy"))
        )

        self.args.n_dow= len(
            np.load(os.path.join(self.args.asset_dir, "dow_classes.npy"))
        )

        self.args.n_cont = len(self.args.cont_col)


        df = df.sort_values(by=['userID', 'Timestamp'], axis=0)

        columns = self.args.cate_col + self.args.cont_col + ['userID','answerCode']
        
        if is_train:
            group = df[columns].groupby(['userID']).apply(lambda row : 
                tuple(row[c].values for c in row.columns if not c == 'userID')
                )
        else:
            group = df[columns].groupby(['userID']).apply(lambda row : 
                tuple(row[c].values for c in row.columns if not c == 'userID')
                )

        return group.values

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train=False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]  # 특정 학생에 대한 풀이 내역 데이터

        # 각 data의 sequence length (학생의 문제 풀이 내역 개수)
        seq_len = len(row[0])
        # TODO 6 : 정해지지 않은 변수로 받을 수 있게 하기 
        
        cate_cols = [row[i] for i in range(0,len(row))]
        # cate_cols = [test, question, tag, correct]
        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1  # 뒤에 seq_len 개만큼의 데이터는 실제 기록이므로

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)  

        # np.array → torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)

        return cate_cols

    def __len__(self):
        return len(self.data)

from torch.nn.utils.rnn import pad_sequence


def collate(batch):
    col_n = len(batch[0])  # 여러 명의 학생 데이터
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])
    
    # batch의 값들을 각 column끼리 그룹화
    for row in batch:  # 각 학생에 대해 padding 처리
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)            
            pre_padded[-len(col):] = col
            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])
    
    return tuple(col_list)


def get_loaders(args, train, valid):

    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )

    return train_loader, valid_loader

def post_process(df, args):  
    df = df.sort_values(by=['userID'], axis=0)
    columns = args.cate_col + args.cont_col + ['userID','answerCode']
    group = df[columns].groupby(['userID','i_head']).apply(lambda row :
            tuple(row[c].values for c in row.columns if not c == 'userID')
            )

    return group.values

def add_features(args):
    cate_dict = {
        'base' : ['assessmentItemID', 'testId', 'KnowledgeTag', 'i_head','hour', 'dow'],
        'cont' : ['assessmentItemID', 'testId', 'KnowledgeTag', 'i_head','hour', 'dow']
    }
    cont_dict = {
        'base' : [],
        'cont' : [  # 원하는 연속형 피쳐 선택
                    'user_correct_answer',
                    'user_total_answer',
                    'user_acc',            
                    't_elapsed',            
                    'cum_correct',
                    # 'last_problem',
                    'head_term',
                    # 'left_asymptote',
                    'elo_prob',
                    'pkt',
                    'u_head_mean',
                    'u_head_count',
                    'u_head_std',
                    'u_head_elapsed',
                    'i_mid_elapsed',
                    'i_mid_mean',
                    'i_mid_std',
                    'i_mid_sum',
                    'i_mid_count',
                    'i_mid_tag_count',
                    'assessment_mean',
                    'assessment_sum',
                    # 'assessment_std',
                    'tag_mean',
                    'tag_sum',
                    # 'tag_std',
                    'tail_mean',
                    'tail_sum',
                    # 'tail_std',
                    'hour_mean',
                    'hour_sum',
                    # 'hour_std',
                    'dow_mean',
                    'dow_sum',
                    # 'dow_std',
                    'tag_elapsed',
                    'tag_elapsed_o',
                    'tag_elapsed_x',
                    'assessment_elapsed',
                    'assessment_elapsed_o',
                    'assessment_elapsed_x',
                    'tail_elapsed',
                    'tail_elapsed_o',
                    'tail_elapsed_x']           
    }
    args.cate_col = cate_dict[args.feature_type]
    args.cont_col = cont_dict[args.feature_type]
    print("cate_columns : ", args.cate_col)
    print("cont_columns : ", args.cont_col)
    return args

