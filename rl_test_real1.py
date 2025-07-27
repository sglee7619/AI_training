# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import warnings
warnings.filterwarnings('ignore')

# 데이터 전처리 함수
def load_and_preprocess_data(file_path='250711_포항 2소결 SCR NOx 측정값_250612-250619.xlsx'):
 """SCR NH3 데이터 로드 및 전처리"""
 print("데이터 로딩 중...")
 
 try:
 # Excel 파일 읽기
 xls = pd.ExcelFile(file_path, engine='openpyxl')
 print(f"✓ Excel 파일 로드 성공: {file_path}")
 print(f"사용 가능한 시트: {xls.sheet_names}")
 
 except FileNotFoundError:
 print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
 print("현재 디렉토리의 파일들을 확인해보세요.")
 import os
 print(f"현재 디렉토리: {os.getcwd()}")
 excel_files = [f for f in os.listdir('.') if f.endswith('.xlsx')]
 print(f"Excel 파일들: {excel_files}")
 raise
 except Exception as e:
 print(f"❌ 파일 로드 중 오류: {e}")
 raise
 
 try:
 # SCR 전단 NOx 읽기
 pre_scr = pd.read_excel(xls, sheet_name='01. 전단')
 pre_scr_5m = pre_scr[['시간, 5분단위','NOx 표준산소 보정.1']]
 pre_scr_5m_clean = pre_scr_5m.dropna()
 pre_scr_5m_clean.columns = ['time', 'scr_pre_nox']
 print(f"✓ SCR 전단 데이터: {len(pre_scr_5m_clean)}개 행")
 
 # TMS NOx 읽기
 tms = pd.read_excel(xls, sheet_name='04. TMS')
 tms1 = tms[['TMS실적조회','Unnamed: 3']]
 tms2 = tms1.iloc[8:-4, [0, 1]].copy()
 tms2.columns = ['time', 'tms_nox']
 tms_clean = tms2.dropna()
 print(f"✓ TMS 데이터: {len(tms_clean)}개 행")
 
 # SCR 조업 변수 읽기
 scr = pd.read_excel(xls, sheet_name='05. 조업 변수')
 scr_clean = scr.dropna()
 print(f"✓ SCR 조업 변수: {len(scr_clean)}개 행, {len(scr_clean.columns)}개 컬럼")
 
 # Object 타입 열을 숫자로 변환
 obj_cols = scr_clean.select_dtypes(include='object').columns
 print(f"변환할 Object 컬럼: {len(obj_cols)}개")
 
 scr_clean[obj_cols] = scr_clean[obj_cols].apply(
 lambda col: pd.to_numeric(col, errors='coerce'))
 
 # 시간 컬럼 생성 및 5분 평균
 scr_clean['time'] = pd.to_datetime(scr_clean['DATE_TIME.1'], format='%Y%m%d%H%M')
 scr_clean = scr_clean.set_index('time')
 scr_5m = scr_clean.resample('5T').mean().reset_index()
 print(f"✓ 5분 평균 후: {len(scr_5m)}개 행")
 
 except KeyError as e:
 print(f"❌ 시트명 또는 컬럼명 오류: {e}")
 print("Excel 파일의 시트명과 컬럼명을 확인해주세요.")
 raise
 except Exception as e:
 print(f"❌ 데이터 처리 중 오류: {e}")
 raise
 
 try:
 # 중요 센서 변수들 선택
 sensor_list = [
 "time","FIC2607_PV", "TI2607_PV", "FIC2632_PV", "PI2632_PV", "FIC2631_PV",
 "TI2631_PV", "FIC2642_PV", "PI2642_PV", "FIC2641_PV", "TI2641_PV",
 "PI2652_PV", "FIC2651_PV", "TI2651_PV", "FI2662_PV", "PI2662_PV",
 "FIC2661_PV", "PI2661_PV", "TI2661_PV", "FI2608_PV", "TI2601_PV",
 "TI2602_PV", "TI2604_PV", "PI2605_PV", "PDI2603_PV", "PI2606_PV",
 "PI2601_PV", "PDI2601_PV", "PI2602_PV", "AI2601_PV", "NOXI2601_PV",
 "O2I2601_PV", "PI2603_PV", "FI2602A_PV", "FI2602B_PV", "PDI2633_PV",
 "NH3I2602_PV", "O2I2602_PV", "TI2603A_PV", "TI2603_PV", "TI2608_PV",
 "FIC2652_PV"
 ]
 
 # 실제 존재하는 컬럼만 선택
 available_sensors = [col for col in sensor_list if col in scr_5m.columns]
 missing_sensors = [col for col in sensor_list if col not in scr_5m.columns]
 
 print(f"사용 가능한 센서: {len(available_sensors)}개")
 if missing_sensors:
 print(f"⚠ 누락된 센서: {missing_sensors}")
 
 scr_5m1 = scr_5m[available_sensors]
 
 # 시간 정렬 및 병합
 pre_scr_5m_clean['time'] = pd.to_datetime(pre_scr_5m_clean['time'])
 tms2['time'] = pd.to_datetime(tms2['time'])
 
 scr_tms_com = pd.merge(pre_scr_5m_clean, tms2, on='time', how='inner')
 scr_tms_com1 = pd.merge(scr_tms_com, scr_5m1, on='time', how='inner')
 scr_tms_com2 = scr_tms_com1.dropna()
 
 # 안정화된 데이터 사용 (처음 500개 제외)
 if len(scr_tms_com2) > 500:
 scr_tms_com3 = scr_tms_com2.iloc[500:]
 else:
 print("⚠ 데이터가 부족합니다. 전체 데이터를 사용합니다.")
 scr_tms_com3 = scr_tms_com2
 
 # 컬럼명 소문자로 통일
 scr_tms_com3.columns = [col.lower() for col in scr_tms_com3.columns]
 
 print(f"✓ 전처리 완료: {len(scr_tms_com3)}개 데이터 포인트")
 print(f"최종 컬럼 수: {len(scr_tms_com3.columns)}")
 print(f"컬럼명: {list(scr_tms_com3.columns)}")
 
 return scr_tms_com3
 
 except Exception as e:
 print(f"❌ 데이터 병합 중 오류: {e}")
 raise


# 개선된 SCR NH3 환경 클래스
class SCRNH3Env(gym.Env):
 """SCR NH3 제어 강화학습 환경"""
 
 def __init__(self, df, nox_target=80, alpha=0.1, beta=0.01, nh3_penalty_threshold=50):
 super().__init__()
 
 self.df = df.reset_index(drop=True)
 self.original_df = self.df.copy() # 원본 데이터 보존
 
 # 컬럼명 정의
 self.nox_meas_col = 'tms_nox'
 self.nh3_flow_col = 'fic2607_pv' # NH3 주입량
 self.nox_pre_col = 'scr_pre_nox' # SCR 전단 NOx
 
 # 상태 변수 (시간 제외)
 self.state_cols = [col for col in df.columns if col != 'time']
 
 # 환경 파라미터
 self.nox_target = nox_target
 self.alpha = alpha # NOx 목표 달성 가중치
 self.beta = beta # NH3 사용량 페널티 가중치
 self.nh3_penalty_threshold = nh3_penalty_threshold
 
 # 스텝 관리
 self.max_step = len(df) - 2
 self.current_step = 0
 
 # 정규화를 위한 통계값 계산
 numeric_df = df.select_dtypes(include=[np.number])
 self.state_mean = numeric_df.mean()
 self.state_std = numeric_df.std() + 1e-8
 
 # 관찰 공간 (정규화된 상태)
 self.observation_space = spaces.Box(
 low=-5.0, high=5.0, 
 shape=(len(self.state_cols),), 
 dtype=np.float32
 )
 
 # 액션 공간 (NH3 주입량: 0-100 kg/h)
 self.action_space = spaces.Box(
 low=0.0, high=100.0, 
 shape=(1,), 
 dtype=np.float32
 )
 
 # 성능 추적 변수
 self.episode_rewards = []
 self.episode_nox_errors = []
 self.episode_nh3_usage = []
 
 def reset(self, seed=None, options=None):
 """환경 초기화"""
 super().reset(seed=seed)
 
 # 랜덤한 시작점 선택 (시계열 특성 고려)
 self.current_step = np.random.randint(0, min(100, self.max_step))
 
 # 원본 데이터로 리셋
 self.df = self.original_df.copy()
 
 # 에피소드 추적 변수 초기화
 self.episode_reward = 0
 self.episode_steps = 0
 self.nox_errors = []
 self.nh3_actions = []
 
 obs = self._get_obs()
 return obs, {}
 
 def _get_obs(self):
 """정규화된 관찰값 반환"""
 if self.current_step >= len(self.df):
 self.current_step = len(self.df) - 1
 
 row = self.df.iloc[self.current_step]
 
 # 상태 정규화
 obs = []
 for col in self.state_cols:
 if col in self.state_mean.index:
 normalized_val = (row[col] - self.state_mean[col]) / self.state_std[col]
 obs.append(np.clip(normalized_val, -5.0, 5.0))
 else:
 obs.append(0.0)
 
 return np.array(obs, dtype=np.float32)
 
 def step(self, action):
 """액션 실행 및 보상 계산"""
 if self.current_step >= self.max_step:
 return self._get_obs(), 0, True, False, {}
 
 # 액션 클리핑 및 적용
 nh3_flow = float(np.clip(action[0], 0.0, 100.0))
 
 # DataFrame에 NH3 주입량 업데이트
 self.df.at[self.current_step + 1, self.nh3_flow_col] = nh3_flow
 
 # 다음 스텝으로 이동
 self.current_step += 1
 
 # 다음 상태의 NOx 측정값 얻기
 next_row = self.df.iloc[self.current_step]
 nox_measured = next_row[self.nox_meas_col]
 
 # 보상 계산
 reward = self._calculate_reward(nox_measured, nh3_flow)
 
 # 종료 조건
 done = self.current_step >= self.max_step
 
 # 성능 추적
 self.episode_reward += reward
 self.episode_steps += 1
 self.nox_errors.append(abs(nox_measured - self.nox_target))
 self.nh3_actions.append(nh3_flow)
 
 # 다음 관찰값
 next_obs = self._get_obs()
 
 # 정보 수집
 info = {
 'nox_measured': nox_measured,
 'nox_target': self.nox_target,
 'nox_error': abs(nox_measured - self.nox_target),
 'nh3_flow': nh3_flow,
 'episode_reward': self.episode_reward,
 'episode_steps': self.episode_steps
 }
 
 return next_obs, reward, done, False, info
 
 def _calculate_reward(self, nox_measured, nh3_flow):
 """개선된 보상 함수"""
 # 1. NOx 목표 달성 보상/패널티
 nox_error = abs(nox_measured - self.nox_target)
 
 if nox_error <= 5: # 목표 ±5 범위 내
 nox_reward = 10.0 - nox_error # 더 가까울수록 높은 보상
 elif nox_error <= 15: # 허용 범위
 nox_reward = -0.5 * nox_error
 else: # 허용 범위 초과
 nox_reward = -2.0 * nox_error
 
 # Critical zone 패널티
 if nox_measured > 120: # 환경 규제 초과
 nox_reward -= 50.0
 
 # 2. NH3 사용량 효율성 보상
 if nh3_flow <= self.nh3_penalty_threshold:
 nh3_reward = -self.beta * (nh3_flow ** 1.5) # 제곱근 패널티로 완화
 else:
 nh3_reward = -self.beta * (nh3_flow ** 2) # 과다 사용 시 강한 패널티
 
 # 3. 안정성 보너스 (변화량이 적을 때)
 stability_reward = 0
 if self.current_step > 0:
 prev_nh3 = self.df.iloc[self.current_step - 1][self.nh3_flow_col]
 nh3_change = abs(nh3_flow - prev_nh3)
 if nh3_change < 5: # 급격한 변화 방지
 stability_reward = 2.0
 elif nh3_change > 20:
 stability_reward = -5.0
 
 total_reward = nox_reward + nh3_reward + stability_reward
 return total_reward

# 커스텀 콜백 클래스
class SCRTrainingCallback(BaseCallback):
 """SCR 훈련 진행 상황 모니터링 콜백"""
 
 def __init__(self, verbose=0):
 super().__init__(verbose)
 self.episode_rewards = []
 self.episode_nox_errors = []
 self.episode_nh3_usage = []
 
 def _on_step(self) -> bool:
 # 에피소드 종료 시 통계 수집
 if self.locals.get('dones', [False])[0]:
 if 'infos' in self.locals and len(self.locals['infos']) > 0:
 info = self.locals['infos'][0]
 self.episode_rewards.append(info.get('episode_reward', 0))
 
 return True
 
 def _on_training_end(self) -> None:
 print(f"훈련 완료! 총 {len(self.episode_rewards)}개 에피소드")

def train_scr_ppo_with_data(df_numeric):
 """전처리된 데이터로 SCR NH3 제어 PPO 훈련"""
 
 print(f"훈련 데이터 형태: {df_numeric.shape}")
 print(f"사용할 변수들: {list(df_numeric.columns)}")
 
 # 1. 환경 생성 및 검증
 print("\n환경 생성 및 검증 중...")
 single_env = SCRNH3Env(df_numeric, nox_target=80, alpha=0.1, beta=0.01)
 
 try:
 check_env(single_env)
 print("✓ 환경 검증 완료")
 except Exception as e:
 print(f"⚠ 환경 검증 경고: {e}")
 
 # 2. 벡터 환경 생성
 env = DummyVecEnv([lambda: Monitor(SCRNH3Env(df_numeric, nox_target=80, alpha=0.1, beta=0.01))])
 
 # 3. 평가용 환경 (다른 구간 데이터 사용)
 eval_data = df_numeric.iloc[len(df_numeric)//2:].copy() # 후반부 데이터로 평가
 eval_env = DummyVecEnv([lambda: Monitor(SCRNH3Env(eval_data, nox_target=80, alpha=0.1, beta=0.01))])
 
 # 4. PPO 모델 설정
 policy_kwargs = dict(
 net_arch=[dict(pi=[256, 128, 64], vf=[256, 128, 64])], # 더 큰 네트워크
 activation_fn=torch.nn.ReLU
 )
 
 print("\nPPO 모델 생성 중...")
 model = PPO(
 "MlpPolicy",
 env,
 verbose=1,
 learning_rate=3e-4, # 더 높은 학습률
 n_steps=512, # 더 많은 스텝 수집
 batch_size=64,
 n_epochs=10, # 더 많은 업데이트 에포크
 gamma=0.7, # 장기 보상 중시
 gae_lambda=0.95, # GAE 파라미터
 clip_range=0.2, # PPO 클리핑
 clip_range_vf=None,
 ent_coef=0.01, # 탐험 장려
 vf_coef=0.5, # 가치함수 가중치
 max_grad_norm=0.5, # 그래디언트 클리핑
 policy_kwargs=policy_kwargs,
 tensorboard_log="./scr_ppo_tensorboard/"
 )
 
 # 5. 콜백 설정
 eval_callback = EvalCallback(
 eval_env,
 best_model_save_path='./scr_best_model/',
 log_path='./scr_eval_logs/',
 eval_freq=2000,
 deterministic=True,
 render=False,
 verbose=1
 )
 
 checkpoint_callback = CheckpointCallback(
 save_freq=5000,
 save_path='./scr_checkpoints/',
 name_prefix='scr_ppo'
 )
 
 training_callback = SCRTrainingCallback(verbose=1)
 
 # 6. 모델 훈련
 print("\n=== SCR NH3 제어 PPO 훈련 시작 ===")
 total_timesteps = 50000 # 실제 산업 데이터이므로 적절한 양으로 조정
 
 model.learn(
 total_timesteps=total_timesteps,
 callback=[eval_callback, checkpoint_callback, training_callback],
 tb_log_name="SCR_NH3_PPO",
 progress_bar=True
 )
 
 # 7. 모델 저장
 model.save("scr_nh3_ppo_final")
 print("\n✓ 모델 저장 완료: scr_nh3_ppo_final.zip")
 
 return model, env, df_numeric

def train_scr_ppo():
 """SCR NH3 제어 PPO 훈련 메인 함수 (구버전 - 사용 안함)"""
 
 # 1. 데이터 로드 및 전처리
 df_all = load_and_preprocess_data()
 
 # 시간 컬럼 제거
 df_numeric = df_all.drop(columns=['time'])
 
 return train_scr_ppo_with_data(df_numeric)

def evaluate_scr_model_with_comparison(model, env, df_data, n_episodes=5):
 """훈련된 SCR 모델 평가 및 실제 사용량과 비교"""
 print(f"\n=== 모델 평가 및 NH3 사용량 비교 ({n_episodes} 에피소드) ===")
 
 results = {
 'episode_rewards': [],
 'nox_errors': [],
 'nh3_usage': [],
 'actual_nh3_usage': [], # 실제 NH3 사용량
 'nox_values': [],
 'actual_nox_values': [], # 실제 NOx 농도 (원본 데이터)
 'target_achievement': [],
 'nh3_savings': [] # NH3 절약량
 }
 
 # 실제 NH3 사용량 (원본 데이터에서)
 nh3_flow_col = 'fic2607_pv'
 nox_meas_col = 'tms_nox' # 실제 NOx 측정값
 
 for episode in range(n_episodes):
 obs = env.reset()
 episode_reward = 0
 episode_nox_errors = []
 episode_nh3_usage = []
 episode_actual_nh3 = []
 episode_nox_values = []
 episode_actual_nox = [] # 실제 NOx 값들
 done = False
 step_count = 0
 
 # 각 에피소드에서 사용할 실제 데이터 구간 선택
 start_idx = np.random.randint(0, max(1, len(df_data) - 200))
 end_idx = min(start_idx + 200, len(df_data))
 
 while not done and step_count < (end_idx - start_idx - 1):
 action, _ = model.predict(obs, deterministic=True)
 obs, reward, done, info = env.step(action)
 
 episode_reward += reward[0]
 
 # 실제 데이터에서 해당 시점의 NH3 사용량과 NOx 농도 가져오기
 actual_idx = start_idx + step_count
 if actual_idx < len(df_data):
 if nh3_flow_col in df_data.columns:
 actual_nh3 = df_data.iloc[actual_idx][nh3_flow_col]
 episode_actual_nh3.append(actual_nh3 if not np.isnan(actual_nh3) else 0)
 else:
 episode_actual_nh3.append(0)
 
 if nox_meas_col in df_data.columns:
 actual_nox = df_data.iloc[actual_idx][nox_meas_col]
 episode_actual_nox.append(actual_nox if not np.isnan(actual_nox) else 80)
 else:
 episode_actual_nox.append(80)
 else:
 episode_actual_nh3.append(0)
 episode_actual_nox.append(80)
 
 if len(info) > 0 and info[0]:
 episode_nox_errors.append(info[0].get('nox_error', 0))
 episode_nh3_usage.append(info[0].get('nh3_flow', 0))
 episode_nox_values.append(info[0].get('nox_measured', 0))
 
 step_count += 1
 
 # 에피소드 결과 저장
 results['episode_rewards'].append(episode_reward)
 results['nox_errors'].append(np.mean(episode_nox_errors) if episode_nox_errors else 0)
 results['nh3_usage'].append(episode_nh3_usage)
 results['actual_nh3_usage'].append(episode_actual_nh3)
 results['nox_values'].append(episode_nox_values)
 results['actual_nox_values'].append(episode_actual_nox) # 실제 NOx 값 저장
 
 # 목표 달성률 계산
 target_achievement = sum(1 for err in episode_nox_errors if err <= 10) / len(episode_nox_errors) if episode_nox_errors else 0
 results['target_achievement'].append(target_achievement * 100)
 
 # NH3 절약량 계산
 if episode_nh3_usage and episode_actual_nh3:
 avg_optimized = np.mean(episode_nh3_usage)
 avg_actual = np.mean(episode_actual_nh3)
 avg_nox_optimized = np.mean(episode_nox_values) if episode_nox_values else 80
 avg_nox_actual = np.mean(episode_actual_nox) if episode_actual_nox else 80
 savings_percent = ((avg_actual - avg_optimized) / avg_actual * 100) if avg_actual > 0 else 0
 results['nh3_savings'].append(savings_percent)
 
 print(f"Episode {episode+1}: Reward={episode_reward:.2f}, "
 f"평균 NOx 오차={np.mean(episode_nox_errors):.2f}, "
 f"실제 NH3={avg_actual:.2f} kg/h → NOx={avg_nox_actual:.1f} ppm, "
 f"최적화 NH3={avg_optimized:.2f} kg/h → NOx={avg_nox_optimized:.1f} ppm, "
 f"절약률={savings_percent:.1f}%")
 else:
 results['nh3_savings'].append(0)
 print(f"Episode {episode+1}: Reward={episode_reward:.2f}, "
 f"평균 NOx 오차={np.mean(episode_nox_errors):.2f}")
 
 # 전체 평가 결과 출력
 print(f"\n=== 전체 평가 결과 ===")
 print(f"평균 에피소드 보상: {np.mean(results['episode_rewards']):.2f}")
 
 # NOx 성능 비교
 all_actual_nox = [val for episode in results['actual_nox_values'] for val in episode if val > 0]
 all_optimized_nox = [val for episode in results['nox_values'] for val in episode if val > 0]
 
 if all_actual_nox and all_optimized_nox:
 print(f"실제 평균 NOx 농도: {np.mean(all_actual_nox):.2f} ppm")
 print(f"최적화 평균 NOx 농도: {np.mean(all_optimized_nox):.2f} ppm")
 nox_diff = np.mean(all_optimized_nox) - np.mean(all_actual_nox)
 print(f"NOx 농도 차이: {nox_diff:+.2f} ppm ({'상승' if nox_diff > 0 else '감소'})")
 
 print(f"평균 NOx 오차: {np.mean(results['nox_errors']):.2f} ppm")
 
 if results['nh3_savings']:
 avg_savings = np.mean([s for s in results['nh3_savings'] if s != 0])
 print(f"평균 NH3 절약률: {avg_savings:.1f}%")
 
 # 실제 vs 최적화 사용량 비교
 all_actual = [val for episode in results['actual_nh3_usage'] for val in episode if val > 0]
 all_optimized = [val for episode in results['nh3_usage'] for val in episode if val > 0]
 
 if all_actual and all_optimized:
 print(f"실제 평균 NH3 사용량: {np.mean(all_actual):.2f} kg/h")
 print(f"최적화 평균 NH3 사용량: {np.mean(all_optimized):.2f} kg/h")
 print(f"절대 절약량: {np.mean(all_actual) - np.mean(all_optimized):.2f} kg/h")
 
 print(f"평균 목표 달성률: {np.mean(results['target_achievement']):.1f}%")
 
 return results

def plot_scr_results_with_comparison(results, df_data):
 """SCR 결과 시각화 (NH3 사용량 비교 및 NOx 농도 비교 포함)"""
 fig, axes = plt.subplots(3, 3, figsize=(20, 15))
 fig.suptitle('SCR NH3 제어 강화학습 결과 및 사용량/NOx 비교', fontsize=16)
 
 # 1. 에피소드별 보상
 axes[0,0].plot(results['episode_rewards'], 'b-o')
 axes[0,0].set_title('에피소드별 보상')
 axes[0,0].set_xlabel('에피소드')
 axes[0,0].set_ylabel('보상')
 axes[0,0].grid(True)
 
 # 2. NOx 오차 분포
 axes[0,1].boxplot(results['nox_errors'])
 axes[0,1].axhline(y=10, color='r', linestyle='--', label='목표 오차 (10ppm)')
 axes[0,1].set_title('NOx 오차 분포')
 axes[0,1].set_ylabel('NOx 오차 (ppm)')
 axes[0,1].legend()
 axes[0,1].grid(True)
 
 # 3. NOx 농도 비교 (실제 vs 최적화)
 if results['nox_values'] and results['actual_nox_values']:
 # 각 에피소드의 평균 NOx 농도 계산
 avg_actual_nox = [np.mean(episode) for episode in results['actual_nox_values'] if episode]
 avg_optimized_nox = [np.mean(episode) for episode in results['nox_values'] if episode]
 
 episodes = range(1, len(avg_actual_nox) + 1)
 width = 0.35
 x = np.arange(len(episodes))
 
 bars1 = axes[0,2].bar(x - width/2, avg_actual_nox, width, label='실제 NOx', alpha=0.8, color='orange')
 bars2 = axes[0,2].bar(x + width/2, avg_optimized_nox, width, label='최적화 NOx', alpha=0.8, color='purple')
 
 # NOx 목표선 추가
 axes[0,2].axhline(y=80, color='red', linestyle='--', linewidth=2, label='NOx 목표 (80ppm)')
 
 axes[0,2].set_title('NOx 농도 비교 (실제 vs 최적화)')
 axes[0,2].set_xlabel('에피소드')
 axes[0,2].set_ylabel('NOx 농도 (ppm)')
 axes[0,2].set_xticks(x)
 axes[0,2].set_xticklabels(episodes)
 axes[0,2].legend()
 axes[0,2].grid(True, alpha=0.3)
 
 # NOx 차이 표시
 for i, (actual, optimized) in enumerate(zip(avg_actual_nox, avg_optimized_nox)):
 diff = optimized - actual
 color = 'red' if diff > 0 else 'green'
 axes[0,2].annotate(f'{diff:+.1f}', 
 xy=(i, max(actual, optimized) + 2), 
 ha='center', va='bottom', fontsize=8, color=color)
 
 # 4. NH3 사용량 비교 (실제 vs 최적화)
 if results['nh3_usage'] and results['actual_nh3_usage']:
 # 각 에피소드의 평균 사용량 계산
 avg_actual = [np.mean(episode) for episode in results['actual_nh3_usage'] if episode]
 avg_optimized = [np.mean(episode) for episode in results['nh3_usage'] if episode]
 
 episodes = range(1, len(avg_actual) + 1)
 width = 0.35
 x = np.arange(len(episodes))
 
 bars1 = axes[1,0].bar(x - width/2, avg_actual, width, label='실제 사용량', alpha=0.8, color='red')
 bars2 = axes[1,0].bar(x + width/2, avg_optimized, width, label='최적화 사용량', alpha=0.8, color='blue')
 
 axes[1,0].set_title('NH3 사용량 비교 (실제 vs 최적화)')
 axes[1,0].set_xlabel('에피소드')
 axes[1,0].set_ylabel('NH3 사용량 (kg/h)')
 axes[1,0].set_xticks(x)
 axes[1,0].set_xticklabels(episodes)
 axes[1,0].legend()
 axes[1,0].grid(True, alpha=0.3)
 
 # 절약량 표시
 for i, (actual, optimized) in enumerate(zip(avg_actual, avg_optimized)):
 if actual > optimized:
 axes[1,0].annotate(f'-{actual-optimized:.1f}', 
 xy=(i, max(actual, optimized) + 1), 
 ha='center', va='bottom', fontsize=8, color='green')
 
 # 5. NH3 절약률
 if results['nh3_savings']:
 savings_data = [s for s in results['nh3_savings'] if s != 0]
 if savings_data:
 axes[1,1].bar(range(len(savings_data)), savings_data, color='green', alpha=0.7)
 axes[1,1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
 axes[1,1].set_title('NH3 절약률')
 axes[1,1].set_xlabel('에피소드')
 axes[1,1].set_ylabel('절약률 (%)')
 axes[1,1].grid(True, alpha=0.3)
 
 # 평균 절약률 표시
 avg_savings = np.mean(savings_data)
 axes[1,1].axhline(y=avg_savings, color='red', linestyle='--', 
 label=f'평균 절약률: {avg_savings:.1f}%')
 axes[1,1].legend()
 
 # 6. 시간별 NOx 농도 패턴 (첫 번째 에피소드)
 if results['nox_values'] and results['actual_nox_values']:
 if results['nox_values'][0] and results['actual_nox_values'][0]:
 time_steps = range(min(len(results['nox_values'][0]), len(results['actual_nox_values'][0])))
 actual_nox_pattern = results['actual_nox_values'][0][:len(time_steps)]
 optimized_nox_pattern = results['nox_values'][0][:len(time_steps)]
 
 axes[1,2].plot(time_steps, actual_nox_pattern, 'o-', color='orange', 
 label='실제 NOx', linewidth=2, alpha=0.8, markersize=4)
 axes[1,2].plot(time_steps, optimized_nox_pattern, 's-', color='purple', 
 label='최적화 NOx', linewidth=2, alpha=0.8, markersize=4)
 axes[1,2].axhline(y=80, color='red', linestyle='--', linewidth=2, 
 label='NOx 목표 (80ppm)', alpha=0.7)
 axes[1,2].fill_between(time_steps, 70, 90, color='green', alpha=0.1, label='허용 범위')
 
 axes[1,2].set_title('시간별 NOx 농도 패턴 (Episode 1)')
 axes[1,2].set_xlabel('시간 스텝')
 axes[1,2].set_ylabel('NOx 농도 (ppm)')
 axes[1,2].legend()
 axes[1,2].grid(True, alpha=0.3)
 
 # 7. 목표 달성률
 axes[2,0].bar(range(len(results['target_achievement'])), results['target_achievement'])
 axes[2,0].axhline(y=80, color='r', linestyle='--', label='목표 달성률 (80%)')
 axes[2,0].set_title('목표 달성률')
 axes[2,0].set_xlabel('에피소드')
 axes[2,0].set_ylabel('달성률 (%)')
 axes[2,0].legend()
 axes[2,0].grid(True)
 
 # 8. 시간별 NH3 사용량 패턴 (첫 번째 에피소드)
 if results['nh3_usage'] and results['actual_nh3_usage']:
 if results['nh3_usage'][0] and results['actual_nh3_usage'][0]:
 time_steps = range(min(len(results['nh3_usage'][0]), len(results['actual_nh3_usage'][0])))
 actual_pattern = results['actual_nh3_usage'][0][:len(time_steps)]
 optimized_pattern = results['nh3_usage'][0][:len(time_steps)]
 
 axes[2,1].plot(time_steps, actual_pattern, 'r-', label='실제 사용량', linewidth=2, alpha=0.8)
 axes[2,1].plot(time_steps, optimized_pattern, 'b-', label='최적화 사용량', linewidth=2, alpha=0.8)
 axes[2,1].fill_between(time_steps, actual_pattern, optimized_pattern, 
 where=np.array(actual_pattern) > np.array(optimized_pattern), 
 color='green', alpha=0.3, label='절약 구간')
 
 axes[2,1].set_title('시간별 NH3 사용량 패턴 (Episode 1)')
 axes[2,1].set_xlabel('시간 스텝')
 axes[2,1].set_ylabel('NH3 사용량 (kg/h)')
 axes[2,1].legend()
 axes[2,1].grid(True, alpha=0.3)
 
 # 9. NH3-NOx 효율성 스캐터 플롯
 if results['nh3_usage'] and results['nox_values'] and results['actual_nh3_usage'] and results['actual_nox_values']:
 # 모든 데이터 포인트 수집
 all_actual_nh3 = [val for episode in results['actual_nh3_usage'] for val in episode if val > 0]
 all_actual_nox = [val for episode in results['actual_nox_values'] for val in episode if val > 0]
 all_opt_nh3 = [val for episode in results['nh3_usage'] for val in episode if val > 0]
 all_opt_nox = [val for episode in results['nox_values'] for val in episode if val > 0]
 
 # 데이터 길이 맞추기
 min_len_actual = min(len(all_actual_nh3), len(all_actual_nox))
 min_len_opt = min(len(all_opt_nh3), len(all_opt_nox))
 
 if min_len_actual > 0 and min_len_opt > 0:
 axes[2,2].scatter(all_actual_nh3[:min_len_actual], all_actual_nox[:min_len_actual], 
 alpha=0.6, color='red', s=20, label='실제 운전')
 axes[2,2].scatter(all_opt_nh3[:min_len_opt], all_opt_nox[:min_len_opt], 
 alpha=0.6, color='blue', s=20, label='최적화 운전')
 
 # NOx 목표선
 axes[2,2].axhline(y=80, color='green', linestyle='--', linewidth=2, label='NOx 목표')
 
 axes[2,2].set_title('NH3 사용량 vs NOx 농도 효율성')
 axes[2,2].set_xlabel('NH3 사용량 (kg/h)')
 axes[2,2].set_ylabel('NOx 농도 (ppm)')
 axes[2,2].legend()
 axes[2,2].grid(True, alpha=0.3)
 
 plt.tight_layout()
 plt.show()
 
 # 1. 에피소드별 보상
 axes[0,0].plot(results['episode_rewards'], 'b-o')
 axes[0,0].set_title('에피소드별 보상')
 axes[0,0].set_xlabel('에피소드')
 axes[0,0].set_ylabel('보상')
 axes[0,0].grid(True)
 
 # 2. NOx 오차 분포
 axes[0,1].boxplot(results['nox_errors'])
 axes[0,1].axhline(y=10, color='r', linestyle='--', label='목표 오차 (10ppm)')
 axes[0,1].set_title('NOx 오차 분포')
 axes[0,1].set_ylabel('NOx 오차 (ppm)')
 axes[0,1].legend()
 axes[0,1].grid(True)
 
 # 3. NH3 사용량 비교 (실제 vs 최적화)
 if results['nh3_usage'] and results['actual_nh3_usage']:
 # 각 에피소드의 평균 사용량 계산
 avg_actual = [np.mean(episode) for episode in results['actual_nh3_usage'] if episode]
 avg_optimized = [np.mean(episode) for episode in results['nh3_usage'] if episode]
 
 episodes = range(1, len(avg_actual) + 1)
 width = 0.35
 x = np.arange(len(episodes))
 
 bars1 = axes[1,0].bar(x - width/2, avg_actual, width, label='실제 사용량', alpha=0.8, color='red')
 bars2 = axes[1,0].bar(x + width/2, avg_optimized, width, label='최적화 사용량', alpha=0.8, color='blue')
 
 axes[1,0].set_title('NH3 사용량 비교 (실제 vs 최적화)')
 axes[1,0].set_xlabel('에피소드')
 axes[1,0].set_ylabel('NH3 사용량 (kg/h)')
 axes[1,0].set_xticks(x)
 axes[1,0].set_xticklabels(episodes)
 axes[1,0].legend()
 axes[1,0].grid(True, alpha=0.3)
 
 # 절약량 표시
 for i, (actual, optimized) in enumerate(zip(avg_actual, avg_optimized)):
 if actual > optimized:
 axes[1,0].annotate(f'-{actual-optimized:.1f}', 
 xy=(i, max(actual, optimized) + 1), 
 ha='center', va='bottom', fontsize=8, color='green')
 
 # 4. NH3 절약률
 if results['nh3_savings']:
 savings_data = [s for s in results['nh3_savings'] if s != 0]
 if savings_data:
 axes[1,1].bar(range(len(savings_data)), savings_data, color='green', alpha=0.7)
 axes[1,1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
 axes[1,1].set_title('NH3 절약률')
 axes[1,1].set_xlabel('에피소드')
 axes[1,1].set_ylabel('절약률 (%)')
 axes[1,1].grid(True, alpha=0.3)
 
 # 평균 절약률 표시
 avg_savings = np.mean(savings_data)
 axes[1,1].axhline(y=avg_savings, color='red', linestyle='--', 
 label=f'평균 절약률: {avg_savings:.1f}%')
 axes[1,1].legend()
 
 # 5. 목표 달성률
 axes[2,0].bar(range(len(results['target_achievement'])), results['target_achievement'])
 axes[2,0].axhline(y=80, color='r', linestyle='--', label='목표 달성률 (80%)')
 axes[2,0].set_title('목표 달성률')
 axes[2,0].set_xlabel('에피소드')
 axes[2,0].set_ylabel('달성률 (%)')
 axes[2,0].legend()
 axes[2,0].grid(True)
 
 # 6. 시간별 NH3 사용량 패턴 (첫 번째 에피소드)
 if results['nh3_usage'] and results['actual_nh3_usage']:
 if results['nh3_usage'][0] and results['actual_nh3_usage'][0]:
 time_steps = range(min(len(results['nh3_usage'][0]), len(results['actual_nh3_usage'][0])))
 actual_pattern = results['actual_nh3_usage'][0][:len(time_steps)]
 optimized_pattern = results['nh3_usage'][0][:len(time_steps)]
 
 axes[2,1].plot(time_steps, actual_pattern, 'r-', label='실제 사용량', linewidth=2, alpha=0.8)
 axes[2,1].plot(time_steps, optimized_pattern, 'b-', label='최적화 사용량', linewidth=2, alpha=0.8)
 axes[2,1].fill_between(time_steps, actual_pattern, optimized_pattern, 
 where=np.array(actual_pattern) > np.array(optimized_pattern), 
 color='green', alpha=0.3, label='절약 구간')
 
 axes[2,1].set_title('시간별 NH3 사용량 패턴 (Episode 1)')
 axes[2,1].set_xlabel('시간 스텝')
 axes[2,1].set_ylabel('NH3 사용량 (kg/h)')
 axes[2,1].legend()
 axes[2,1].grid(True, alpha=0.3)
 
 plt.tight_layout()
 plt.show()
 
 # 추가 분석: 전체 데이터에 대한 상세 비교
 if results['nh3_usage'] and results['actual_nh3_usage']:
 plt.figure(figsize=(14, 8))
 
 # 모든 에피소드의 데이터를 하나로 합치기
 all_actual = []
 all_optimized = []
 all_time_steps = []
 
 cumulative_time = 0
 for i, (actual_ep, optimized_ep) in enumerate(zip(results['actual_nh3_usage'], results['nh3_usage'])):
 if actual_ep and optimized_ep:
 min_len = min(len(actual_ep), len(optimized_ep))
 episode_time = list(range(cumulative_time, cumulative_time + min_len))
 
 all_actual.extend(actual_ep[:min_len])
 all_optimized.extend(optimized_ep[:min_len])
 all_time_steps.extend(episode_time)
 
 cumulative_time += min_len + 10 # 에피소드 간 간격
 
 # 전체 패턴 플롯
 plt.subplot(2, 1, 1)
 plt.plot(all_time_steps, all_actual, 'r-', label='실제 NH3 사용량', linewidth=1.5, alpha=0.8)
 plt.plot(all_time_steps, all_optimized, 'b-', label='최적화 NH3 사용량', linewidth=1.5, alpha=0.8)
 plt.fill_between(all_time_steps, all_actual, all_optimized, 
 where=np.array(all_actual) > np.array(all_optimized), 
 color='green', alpha=0.3, label='NH3 절약 구간')
 plt.title('전체 NH3 사용량 패턴 비교')
 plt.xlabel('시간 스텝')
 plt.ylabel('NH3 사용량 (kg/h)')
 plt.legend()
 plt.grid(True, alpha=0.3)
 
 # 히스토그램 비교
 plt.subplot(2, 1, 2)
 plt.hist(all_actual, bins=30, alpha=0.7, color='red', label='실제 사용량 분포', density=True)
 plt.hist(all_optimized, bins=30, alpha=0.7, color='blue', label='최적화 사용량 분포', density=True)
 plt.axvline(np.mean(all_actual), color='red', linestyle='--', linewidth=2, 
 label=f'실제 평균: {np.mean(all_actual):.2f} kg/h')
 plt.axvline(np.mean(all_optimized), color='blue', linestyle='--', linewidth=2, 
 label=f'최적화 평균: {np.mean(all_optimized):.2f} kg/h')
 plt.title('NH3 사용량 분포 비교')
 plt.xlabel('NH3 사용량 (kg/h)')
 plt.ylabel('확률 밀도')
 plt.legend()
 plt.grid(True, alpha=0.3)
 
 plt.tight_layout()
 plt.show()
 
 # 성능 요약 출력
 if all_actual and all_optimized:
 total_actual = sum(all_actual)
 total_optimized = sum(all_optimized)
 total_savings = total_actual - total_optimized
 savings_percent = (total_savings / total_actual * 100) if total_actual > 0 else 0
 
 print(f"\n=== NH3 사용량 최적화 요약 ===")
 print(f"총 실제 NH3 사용량: {total_actual:.2f} kg")
 print(f"총 최적화 NH3 사용량: {total_optimized:.2f} kg")
 print(f"총 절약량: {total_savings:.2f} kg ({savings_percent:.1f}% 절약)")
 print(f"평균 실제 사용량: {np.mean(all_actual):.2f} kg/h")
 print(f"평균 최적화 사용량: {np.mean(all_optimized):.2f} kg/h")
 
 # 표준편차 비교 (안정성)
 std_actual = np.std(all_actual)
 std_optimized = np.std(all_optimized)
 print(f"실제 사용량 표준편차: {std_actual:.2f} kg/h")
 print(f"최적화 사용량 표준편차: {std_optimized:.2f} kg/h")
 print(f"안정성 개선: {((std_actual - std_optimized)/std_actual*100):.1f}% 변동성 감소")

if __name__ == "__main__":
 # 필요한 라이브러리 임포트
 import torch
 
 print("=== SCR NH3 제어 강화학습 시스템 ===")
 
 try:
 # 1. 데이터 로드 및 전처리 (먼저 실행)
 df_all = load_and_preprocess_data()
 df_numeric = df_all.drop(columns=['time'])
 
 # 2. 모델 훈련
 model, env, df_data = train_scr_ppo_with_data(df_numeric)
 
 # 3. 모델 평가 (NH3 사용량 비교 포함)
 results = evaluate_scr_model_with_comparison(model, env, df_data, n_episodes=5)
 
 # 4. 결과 시각화 (비교 분석 포함)
 print("\n결과 시각화 및 비교 분석 중...")
 plot_scr_results_with_comparison(results, df_data)
 
 print("\n=== 훈련 및 평가 완료 ===")
 print("저장된 파일:")
 print("- 최종 모델: scr_nh3_ppo_final.zip")
 print("- 최고 모델: ./scr_best_model/best_model.zip")
 print("- 체크포인트: ./scr_checkpoints/")
 print("- 텐서보드 로그: ./scr_ppo_tensorboard/")
 
 # 모델 로드 예시
 print("\n모델 로드 예시:")
 print("loaded_model = PPO.load('scr_nh3_ppo_final')")
 
 except Exception as e:
 print(f"오류 발생: {e}")
 print("데이터 파일 경로와 시트명을 확인해주세요.")

# %%
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

# 폰트 경로 설정
font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
font_name = fm.FontProperties(fname=font_path).get_name()

# matplotlib에 설정
mpl.rc('font', family=font_name)
mpl.rcParams['axes.unicode_minus'] = False # 마이너스 깨짐 방지

# 테스트
plt.plot([1, 2, 3], [1, 4, 9])
plt.title('한글 제목')
plt.xlabel('가로축')
plt.ylabel('세로축')
plt.show()
# %%