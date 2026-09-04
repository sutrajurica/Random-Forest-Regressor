import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def pace_to_seconds(pace):
    pace = str(pace)
    if pace.count(":") == 1:
        m, s = pace.split(":")
        return int(m) * 60 + int(s)
    return np.nan

def time_to_seconds(t):
    try:
        t = str(t).strip()
        if ":" not in t:
            return np.nan
        parts = t.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        return np.nan
    except:
        return np.nan

def seconds_to_pace(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"

df = pd.read_csv("activity_log.csv")
df["Time (sec)"] = df["Time"].apply(time_to_seconds)
df["Avg Pace (sec)"] = df["Avg Pace"].apply(pace_to_seconds)
df['Date'] = pd.to_datetime(df['Date'], format ='mixed')
df = df.sort_values('Date', ascending=True).reset_index(drop=True)
df = df.set_index('Date')


df = df.dropna(subset=[
    "Distance", "Avg HR", "Max HR", "Avg Run Cadence",
    "Time (sec)", "Avg Pace (sec)"
])

df['days_since_start'] = (df.index - df.index.min()).days
df['week_number'] = df['days_since_start'] // 7
df['month_number'] = df['days_since_start'] // 30

df['cumulative_distance'] = df['Distance'].cumsum()
df['cumulative_time'] = df['Time (sec)'].cumsum()
df['run_count'] = range(1, len(df) + 1)

df['distance_last_7days'] = df['Distance'].rolling(window='7D').sum()
df['distance_last_14days'] = df['Distance'].rolling(window='14D').sum()
df['distance_last_30days'] = df['Distance'].rolling(window='30D').sum()

df['runs_last_7days'] = df['Distance'].rolling(window='7D').count()
df['runs_last_30days'] = df['Distance'].rolling(window='30D').count()

df['avg_distance_last_10runs'] = df['Distance'].rolling(window=10, min_periods=1).mean()
df['avg_time_last_10runs'] = df['Time (sec)'].rolling(window=10, min_periods=1).mean()

df['days_since_last_run'] = df.index.to_series().diff().dt.days.fillna(0)
df['avg_recovery_days'] = df['days_since_last_run'].rolling(window=10, min_periods=1).mean()

df['avg_hr_last_5runs'] = df['Avg HR'].rolling(window=5, min_periods=1).mean()
df['avg_hr_last_10runs'] = df['Avg HR'].rolling(window=10, min_periods=1).mean()
df['max_hr_last_10runs'] = df['Max HR'].rolling(window=10, min_periods=1).mean()

df['hr_trend'] = df['Avg HR'].diff().rolling(window=5, min_periods=1).mean()
df['hr_variability'] = df['Avg HR'].rolling(window=10, min_periods=1).std()

df['avg_cadence_last_5runs'] = df['Avg Run Cadence'].rolling(window=5, min_periods=1).mean()
df['avg_cadence_last_10runs'] = df['Avg Run Cadence'].rolling(window=10, min_periods=1).mean()
df['cadence_trend'] = df['Avg Run Cadence'].diff().rolling(window=5, min_periods=1).mean()

df['high_hr_runs_last_10'] = (df['Max HR'] > df['Max HR'].quantile(0.75)).rolling(window=10, min_periods=1).sum()

feature_cols = [
    'Distance',
    'Avg HR',
    'Max HR',
    'Avg Run Cadence',
    
    'days_since_start',
    'run_count',
    'cumulative_distance',
    'cumulative_time',
    
    'distance_last_7days',
    'distance_last_14days',
    'distance_last_30days',
    'runs_last_7days',
    'runs_last_30days',
    
    'avg_distance_last_10runs',
    'avg_time_last_10runs',
    
    'days_since_last_run',
    'avg_recovery_days',
    
    'avg_hr_last_5runs',
    'avg_hr_last_10runs',
    'max_hr_last_10runs',
    'hr_trend',
    'hr_variability',
    
    'avg_cadence_last_5runs',
    'avg_cadence_last_10runs',
    'cadence_trend',
    
    'high_hr_runs_last_10',
]

df_model = df.dropna(subset=feature_cols + ['Avg Pace (sec)'])

X = df_model[feature_cols]
y = df_model['Avg Pace (sec)']

split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

estimators = [50, 100, 200]
depths = [2, 4, 8, 16, 32, 64, 128]

mae_50_train = []
mae_50_test = []

mae_100_train = []
mae_100_test = []

mae_200_train = []
mae_200_test = []

split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

for i in estimators:
    for j in depths:
        model = RandomForestRegressor(
            n_estimators=i,
            max_depth=j, 
            min_samples_split=10, 
            min_samples_leaf=5, 
            max_features='sqrt',
            random_state=42
        )

        model.fit(X_train, y_train)
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        print("="*70)
        print(f"\nTraining Set ({len(X_train)} runs):")
        print(f"  MAE: {mean_absolute_error(y_train, train_preds):.2f} ({i} estimators and {j} depths)  seconds/mile")
        print(f"  R²:  {r2_score(y_train, train_preds):.4f}")

        print(f"\nTest Set ({len(X_test)} runs - Future predictions):")
        print(f"  MAE: {mean_absolute_error(y_test, test_preds):.2f} ({i} estimators and {j} depths) seconds/mile")
        print(f"  R²:  {r2_score(y_test, test_preds):.4f}")

        if (i==50):
            mae_50_train.append(mean_absolute_error(y_train, train_preds))
            mae_50_test.append(mean_absolute_error(y_test, test_preds))
        elif (i == 100):
            mae_100_train.append(mean_absolute_error(y_train, train_preds))
            mae_100_test.append(mean_absolute_error(y_test, test_preds))
        elif (i == 200):
            mae_200_train.append(mean_absolute_error(y_train, train_preds))
            mae_200_test.append(mean_absolute_error(y_test, test_preds))

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n{'='*70}")
print("WHAT BIOLOGICAL FACTORS DRIVE YOUR PACE?")
print("="*70)
print("\nTop 15 Most Important Factors:")
print("-" * 70)

for idx, row in feature_importance.head(15).iterrows():
    print(f"{row['feature']:30s} {row['importance']:6.4f}")

print(f"\n{'='*70}")
print("30-DAY PACE PREDICTION")
print("="*70)

last_date = df_model.index.max()
thirty_days_ago = last_date - pd.Timedelta(days=30)
last_30_days = df_model[df_model.index >= thirty_days_ago]

recent_runs = len(last_30_days)
recent_distance_per_run = last_30_days['Distance'].mean()
recent_total_distance = last_30_days['Distance'].sum()
recent_avg_hr = last_30_days['Avg HR'].mean()
recent_avg_cadence = last_30_days['Avg Run Cadence'].mean()

current_pace = y.iloc[-1]
last_row = df_model.iloc[-1]

print(f"\nYour Current Fitness:")
print(f"  Current pace: {seconds_to_pace(current_pace)} min/mile ({current_pace:.1f} sec/mile)")
print(f"  Average HR: {recent_avg_hr:.0f} bpm")
print(f"  Average cadence: {recent_avg_cadence:.0f} spm")

print(f"\nRecent Training (last 30 days):")
print(f"  Total runs: {recent_runs}")
print(f"  Total distance: {recent_total_distance:.1f} mile")
print(f"  Avg distance per run: {recent_distance_per_run:.2f} mile")

future_row = last_row[feature_cols].copy()
future_row['days_since_start'] += 30
future_row['run_count'] += recent_runs
future_row['cumulative_distance'] += recent_total_distance
future_row['cumulative_time'] += last_30_days['Time (sec)'].sum()
future_row['distance_last_30days'] = recent_total_distance

predicted_pace_maintain = model.predict(pd.DataFrame([future_row]))[0]
improvement_maintain = current_pace - predicted_pace_maintain

print(f"\n{'='*70}")
print("PREDICTION SCENARIOS (30 days from now)")
print("="*70)

print(f"\n1. MAINTAIN CURRENT TRAINING")
print(f"   ({recent_runs} runs, {recent_total_distance:.0f}mile per month)")
print(f"   Predicted pace: {seconds_to_pace(predicted_pace_maintain)} ({predicted_pace_maintain:.1f} sec/mile)")
print(f"   Expected change: {improvement_maintain:+.1f} sec/mile ({improvement_maintain/current_pace*100:+.2f}%)")
