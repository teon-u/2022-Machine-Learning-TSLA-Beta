""" 1-1. GET BASE DATA """
import joblib
import os
base = os.path.dirname(os.path.abspath(__file__)) #1
#stock = joblib.load(os.path.join(base,'target','stock')) #2
stock = 'TSLA' #stock

# 1. 현재폴더 경로 -> base 변수
# 2. target/stock 데이터 -> stock 변수 (TSLA 등 주식 Ticker)


""" 1-2. SET DATETIME """
from datetime import datetime, timedelta
start_date = '2017-05-01' #1
end_date = str(datetime.today().strftime("%Y-%m-%d")) #2
time_range = start_date + " " + end_date

# 1. start_date 값이 5년 이전으로 가면, pytrend 크롤링 과정에서 오류발생(주단위 -> 월단위 데이터로 자동 크롤링)
# 2. end_date 값은 오늘(최신)으로 지정


""" 1-3. COLLECT STOCK DATA """
import yfinance as yf
df_yf = yf.download(stock,start = start_date)
df_yf = df_yf.resample(rule = 'w').last() #1
df_yf = df_yf.reset_index(drop = True) #2

# 1. 받아온 일단위 데이터를 주단위로 변경, 마지막 값이 대표하도록
# 2. 인덱스 초기화 (날짜정보 제거됨)


""" 1-4. COLLECT SEARCH DATA """
from pytrends.request import TrendReq
pytrends = TrendReq(hl='en-US') #1
pytrends.build_payload([stock], cat=0, timeframe = time_range, geo = '', gprop = '') #2
df_pt = pytrends.interest_over_time() #3
df_pt = df_pt.reset_index(drop = False) #4
df_pt = df_pt[['date', stock]] #5
df_pt.columns = ['Date', stock] #6

# 1. hl = specifies host language for accessing Google Trends.
# 2. stock 를 지정, Time_range 기간의 데이터 Payload 생성
# 3. 시간별 검색량 변화 데이터 크롤링 -> df_pt
# 4. 인덱스 초기화 (날짜정보 남김)
# 5. 불필요 열 제거 (isPartial 열)
# 6. 컬럼명 재설정


""" 1-5. COMBINE & EDA DATA """
import pandas as pd
df = pd.concat([df_pt,df_yf], axis=1, ignore_index=True) #1
df.iloc[-1:,0:1] = end_date #2
df = df.fillna(method = 'ffill') #3
df = df.drop([df.columns[6]], axis=1) #4
df.columns = ['Date','Search','Price','Open','High','Low','Vol'] #5

# 1. 주식, 검색 데이터프레임 합치기 (Concat)
# 2. 검색 데이터 마지막 줄이 결측치로 처리됨(주 별 데이터로 받아오기 때문), 이중 Date 열을 현재날짜로 채워넣기
# 3. 검색 데이터 - 검색량 결측치를 ffill (이전값으로 채우기) 방식으로 처리
# 4. 모델에서 사용하지 않는 열 제거 (증자, 배당 등 반영한 주가라는데.. 잘 모르니 제거)
# 5. 컬럼명 보기쉽게 재설정
# 데이터 Cut-Off 코드는 반영 보류 (연산 후 성능하락시 재반영)


""" 1-6. FEATURE ENGINEERING """
import numpy as np

def change(target_feature): #1
    target_list = [np.nan] #2
    for i in range(len(target_feature)):
        try:
            target_list.append(round(target_feature[i+1] / target_feature[i] * 100 - 100,2)) #3
        except:
            target_list.append(np.nan) #4
    target_list = target_list[:-1] #2
    return target_list

# 1. 변동량 계산 함수 - 전날 값 대비 오늘 값을 %로 계산
# 2. 하루씩 밀려서 계산되게 하려고 초기값으로 결측치 하나 넣어둠 -> 마지막에 뒤에서 한줄 삭제해서 Row size 맞춤
# 3. 다음날 주가 / 오늘 주가 * 100 - 100 한 뒤 소수점 두번째 자리에서 잘라 리스트에 추가
# 4. i+1의 값이 없을때가 나오기 때문에 넣어둔 오류처리


def moving_avg(much, target_feature): #1
    target_list = []
    for i in range(len(target_feature)):
        feature_list = []
        try:
            for l in range(much+1): #2
                feature_list.append(target_feature[i-l])
            target_list.append(round(np.mean(feature_list),2)) #3
        except:
            target_list.append(np.nan) #4
    return target_list

# 1. 이동평균선 계산 함수 - much 만큼 전의 데이터들을 긁어와 평균을 계산
# 2. much 값 만큼 반복문 실행, 반복문 실행하면서 이전 값들을 Feature 리스트에 추가
# 3. Feature 리스트의 값을 평균내서 Target 리스트에 추가
# 4. much 주 전의 값이 없어 연산이 안되는 경우도 있기 때문에 오류처리 (데이터 초반)


def price_go(much, target_feature): #1
    week = []
    for i in range(len(target_feature)):
        try:
            if target_feature[i] < target_feature[i+much]: #2
                week.append(1)
            elif target_feature[i] >= target_feature[i+much]: #3
                week.append(0)
        except:
            week.append(np.nan) #4
    return week

# 1. 가격변화 계산 함수 - much 만큼 후의 가격이 올랐으면 1, 내렸으면 0을 반환
# 2. 올랐으면 week 리스트에 1 추가
# 3. 내렸으면 week 리스트에 0 추가
# 4. much 주 후의 값이 없어 연산이 안되는 경우도 있기 때문에 오류처리 (데이터 후반)

df['Change'] = change(df['Price']) #1
df['avg_one'] = moving_avg(1, df.Price) #2
df['avg_thr'] = moving_avg(3, df.Price)
df['avg_fiv'] = moving_avg(5, df.Price)
df['one'] = price_go(1,df.Price) #3
df['two'] = price_go(2,df.Price)
df['thr'] = price_go(3,df.Price)
df['for'] = price_go(4,df.Price)
df['fiv'] = price_go(5,df.Price)
df['six'] = price_go(6,df.Price)
df['sev'] = price_go(7,df.Price)
df['eig'] = price_go(8,df.Price)
df['nin'] = price_go(9,df.Price)
df['ten'] = price_go(10,df.Price)
df = df[5:] #4

# 1. 변동량 함수 적용 : 특성 'Change'
# 2. 이평선 함수 적용 : 특성 'avg_  '
# 3. 가격변화 함수 적용 : 타겟 '1~10'
# 4. 이평선 함수로 인해 생성된 결측치를 제거해 줌 (데이터 앞쪽 5줄)
# 가격변화 함수가 타겟인 이유는, 머신러닝으로 예측, 투자에 활용할 값이기 때문임
# 일반적인 머신러닝에서는 타겟이 하나임


""" 1-7. SEND & SAVE DATA """
joblib.dump(df, os.path.join(base,'data',stock)) #1
# 1. picke화 해서 저장 / 위치 : 현재경로 아래 'data' 폴더 / 이름 : stock 변수에 들어있는 문자열