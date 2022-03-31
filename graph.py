"""  5-1. GET BASE DATA  """
import joblib
import os
base = os.path.dirname(os.path.abspath(__file__))
# stock = joblib.load(os.path.join(base,'target','stock'))
stock = 'TSLA' # 코드 전체 작성 후 자동화 완료시 삭제
df = joblib.load(os.path.join(base,'data',stock))
#df = df[10:] # 이동평균선으로 생기는 결측치 제거
#df = df.reset_index(drop=True) # 인덱스 초기화

pipe_1 = joblib.load(os.path.join(base,'model','pipe_1'))
pipe_2 = joblib.load(os.path.join(base,'model','pipe_2'))
pipe_3 = joblib.load(os.path.join(base,'model','pipe_3'))
pipe_4 = joblib.load(os.path.join(base,'model','pipe_4'))
pipe_5 = joblib.load(os.path.join(base,'model','pipe_5'))
pipe_6 = joblib.load(os.path.join(base,'model','pipe_6'))
pipe_7 = joblib.load(os.path.join(base,'model','pipe_7'))
pipe_8 = joblib.load(os.path.join(base,'model','pipe_8'))
pipe_9 = joblib.load(os.path.join(base,'model','pipe_9'))
pipe_10 = joblib.load(os.path.join(base,'model','pipe_10'))

""" 5-2. TRAIN-TEST SPLIT """
train = df[0:int(len(df)*0.75)]
test = df[int(len(df)*0.75):]
features = ['Search','Price','Open','High','Low','Vol','Change','avg_one','avg_thr','avg_fiv']

X_train = train[features]

""" 5-2. VISUALIZING """
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
def graphy(pipe,num):
    ftr_importances_values = pipe.feature_importances_
    ftr_importances = pd.Series(ftr_importances_values, index = X_train.columns)
    ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
    plt.figure(figsize=(8,6))
    title = 'Top Feature Importances' + str(num)
    plt.title(title)
    sns.barplot(x=ftr_top20, y=ftr_top20.index)


    return ftr_top20

graphy(pipe_1,1)
graphy(pipe_2,2)
graphy(pipe_3,3)
graphy(pipe_4,4)
graphy(pipe_5,5)
graphy(pipe_6,6)
graphy(pipe_7,7)
graphy(pipe_8,8)
graphy(pipe_9,9)
graphy(pipe_10,10)

plt.show()