
# %%

import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
import os
import numpy as np

#print(os.getcwd())   

#file_path = 'C:\Users\LG\Desktop\AI_training\250711_포항 2소결 SCR NOx 측정값_250612-250619.xlsx'

xls = pd.ExcelFile('250711_포항 2소결 SCR NOx 측정값_250612-250619.xlsx', engine='openpyxl')
#print(xls.sheet_names)

# SCR 전단 nox 읽기 및 nan 처리 등
pre_scr = pd.read_excel(xls, sheet_name='01. 전단')
pre_scr_5m = pre_scr[['시간, 5분단위','NOx 표준산소 보정.1']]
pre_scr_5m.isna().sum() # nan 개수 sum
pre_scr_5m_clean = pre_scr_5m.dropna()
# dropna 이렇게 하면 행에 nan 하나만 있어도 다 날아감
pre_scr_5m_clean.columns = ['time', 'scr_pre_nox'] # column 이름 변경

# TMS nox 읽기 및 nan 처리 등
tms = pd.read_excel(xls, sheet_name='04. TMS')
tms1 = tms[['TMS실적조회','Unnamed: 3']]
tms2 = tms1.iloc[8:-4, [0, 1]].copy() # useless header 자르고 copy
tms2.columns = ['time', 'nox'] # column 이름 변경
tms_clean = tms2.dropna()

# datetime으로 변환 & 같은 시간 일치 extract, (다른 방법도 있음)
pre_scr_5m_clean['time'] = pd.to_datetime(pre_scr_5m_clean['time'])
tms2['time'] = pd.to_datetime(tms2['time'])
scr_tms_com = pd.merge(pre_scr_5m_clean, tms2, on='time', how='inner')
#print(scr_tms_com)

# pre SCR variables 처리 
#file_path1 = '/content/drive/MyDrive/Classroom/250709_2소결_연원료_소결bed_612_619.xlsx'
xls1 = pd.ExcelFile('250709_2소결_연원료_소결bed_612_619.xlsx', engine='openpyxl')
print(xls1.sheet_names)

raw = pd.read_excel(xls1, sheet_name='원료 종류 및 사용량, 생산량',skiprows=4)
raw1 = raw.iloc[:-1, [0] + list(range(13, raw.shape[1], 4))].copy() # 앞에 date 붙이고, 4간격으로 합계만 읽기
raw1.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

raw1['Date'] = pd.to_datetime(raw1['Date'], format='%y-%m-%d') # merge 를 위해  datetime으로 지정 

# day 비교를 위해 오른쪽에 day column 추가 
pre_scr_5m_clean['day'] = pre_scr_5m_clean['time'].dt.date
raw1['day'] = raw1['Date'].dt.date

# 5분 할당된 raw material 넣을 column 
new_cols = [f'new{i}' for i in range(1, 23)]
for col in new_cols:
    pre_scr_5m_clean[col] = 0

# 5분 할당된 raw material date 같으면 넣기 
for i in range(pre_scr_5m_clean.shape[0]):
    for j in range(raw1.shape[0]):
     for k in range(1,23):
      if pre_scr_5m_clean['day'].iloc[i] == raw1['day'].iloc[j]:
         pre_scr_5m_clean.iloc[i,k+2] = raw1.iloc[j,k]/288 #일일 5분 개수

## 소결 bed variables 
bed = pd.read_excel(xls1, sheet_name='DB')

# 새로 date_time 설정하고 5분 평균
bed['time'] = pd.to_datetime(bed['DATE_TIME'], format='%Y%m%d%H%M')
bed = bed.set_index('time')
bed_5m = bed.resample('5T').mean().reset_index()
del bed_5m['DATE_TIME']

# bed, 원료, SCR 전단 nox 자료 통합
merged = pd.merge(pre_scr_5m_clean, bed_5m, on='time', how='inner')
del merged['day']






merged.to_excel('test.xlsx', sheet_name='Sheet1', index=False)


#%reset -f


#del merged['day']




# %%
# Figures 
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt



left_col = 'scr_pre_nox'
# 오른쪽에 반복해서 그릴 new1~new22 칼럼 자동 추출
new_cols = [c for c in merged.columns if c.startswith('TIC')]

for i, right_col in enumerate(new_cols, start=1):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    # 왼쪽 축: scr_pre_nox
    ax1.plot(merged.index, merged[left_col], label=left_col)
    ax1.set_ylabel(left_col)

    # 오른쪽 축: newX
    ax2.plot(merged.index, merged[right_col], label=right_col, color='tab:red')
    ax2.set_ylabel(right_col)

    # 공통 설정
    ax1.set_xlabel('Time')
    fig.suptitle(f"{left_col} vs {right_col}")

    # 범례 합치기
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # JPG로 저장
    filename = f"plot_{i}_{left_col}_vs_{right_col}.jpg"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {filename}")


# plotly.express 사용 예
fig = px.line(
    merged,
    x='time',
    y=cols[0],
    labels={'time':'Time', cols[0]: cols[0]},
    title=f"{cols[0]} over Time"
)
fig.show()


# fig 생성
fig = go.Figure()

# 첫 번째 축 (왼쪽 y-axis)
fig.add_trace(
    go.Scatter(
        x=merged['time'], 
        y=merged['scr_pre_nox'],
        mode='lines'
    )
)

# 두 번째 축 (오른쪽 y-axis)
fig.add_trace(
    go.Scatter(
        x=merged['time'],
        y=merged['new4'],
        mode='lines',
        yaxis='y2'
    )
)

# 레이아웃 업데이트
fig.update_layout(
    title="Multi-Axis Plot Example",
    xaxis=dict(title='Time'),
    yaxis=dict(title=merged.columns[1]),          
    yaxis2=dict(
        title=merged.columns[2],                   
        overlaying='y',
        side='right'
    ),
    legend=dict(x=0.1, y=1.1)
)

# 그래프 출력
fig.show()

