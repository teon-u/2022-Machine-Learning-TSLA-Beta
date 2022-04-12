"""  4-1. GET BASE DATA  """
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

"""  4-2. PREDICTION """
import numpy as np
features = ['Search', 'Price', 'Open', 'High', 'Low', 'Vol', 'Change', 'avg_one', 'avg_thr', 'avg_fiv']#, 'avg_ten']
temp_df = df[features]
df['p_one'] = pipe_1.predict(temp_df)
df['p_two'] = pipe_2.predict(temp_df)
df['p_thr'] = pipe_3.predict(temp_df)
df['p_for'] = pipe_4.predict(temp_df)
df['p_fiv'] = pipe_5.predict(temp_df)
df['p_six'] = pipe_6.predict(temp_df)
df['p_sev'] = pipe_7.predict(temp_df)
df['p_eig'] = pipe_8.predict(temp_df)
df['p_nin'] = pipe_9.predict(temp_df)
df['p_ten'] = pipe_10.predict(temp_df)

condition = (df['p_one'] + df['p_two'] + df['p_thr'] + df['p_for'] + df['p_fiv'] + df['p_six'] + \
            df['p_sev'] + df['p_eig'] + df['p_nin'] + df['p_ten']) > 6

target_feature = df.Price
much = 3
Income_list = []
Income_rate = []

for i in range(len(target_feature)):
  try:
    income = target_feature[i+much] - target_feature[i]
    Income_rate.append(round((income/target_feature[i]*100),2)) # 수익률
    Income_list.append(round((income),2)) # 수익

  except:
    Income_list.append(np.nan)
    Income_rate.append(np.nan)

df['Income'] = Income_list
df['Incrat'] = Income_rate

dfk = df[condition]
print("DATA----------------------------------------------------------")
print("전체 데이터: ",df.shape)
print("추출 데이터: ", dfk.shape)
print("추출 비율 :", dfk.shape[0]/df.shape[0]*100 ,"%")
print("PRED----------------------------------------------------------")


temp_list = []
df['sum_p'] = df['p_one'] + df['p_two'] + df['p_thr'] + df['p_for'] + df['p_fiv'] + df['p_six'] + df['p_sev'] + df['p_eig'] + df['p_nin'] + df['p_ten']
df['sum_r'] = df['one'] + df['two'] + df['thr'] + df['for'] + df['fiv'] + df['six'] + df['sev'] + df['eig'] + df['nin'] + df['ten']
dfp = df[['Date','Search','Price','sum_r','sum_p','Income','Incrat']]

print(dfp[condition].tail(10))
print("BASE----------------------------------------------------------")
print(dfp.tail(10))

print("Money Pred----------------------------------------------------")
# 예측된 모든 지표가 1일 경우 ( 모델 예측 시행 )
print('예측 투자 평균 수익률 (전체) :',round(np.mean(dfk.Incrat),3),'%')
# 최근 5주 예측 수익률
print('평균 예측 수익률 (최근 05주) :',round(np.mean(dfk.Incrat[-8:-3]),3),'%')
# 최근 10주 예측 수익률
print('평균 예측 수익률 (최근 10주) :',round(np.mean(dfk.Incrat[-14:-3]),3),'%')
# 최근 25주 예측 수익률
print('평균 예측 수익률 (최근 25주) :',round(np.mean(dfk.Incrat[-29:-3]),3),'%')
print("Money Base----------------------------------------------------")
# 무지성 투자했을경우의 수익률 (Base Model)
print('기본 투자 평균 수익률 (전체) :',round(np.mean(df.Incrat),3),'%')
# 최근 5주 기본 수익률
print('평균 기본 수익률 (최근 05주) :',round(np.mean(df.Incrat[-8:-3]),3),'%')
# 최근 10주 기본 수익률
print('평균 기본 수익률 (최근 10주) :',round(np.mean(df.Incrat[-14:-3]),3),'%')
# 최근 25주 기본 수익률
print('평균 기본 수익률 (최근 25주) :',round(np.mean(df.Incrat[-29:-3]),3),'%')
print("--------------------------------------------------------------")