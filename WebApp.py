import streamlit as st
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
        if len(parts) == 1:
            return float(s)
        return np.nan
    except:
        return np.nan

def seconds_to_pace(seconds):
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"

st.set_page_config(page_title="Run Pace Predictor", layout="wide")
st.title("🏃 Run Pace Predictor & Analyzer")
st.markdown("""
This app analyzes your running data. It **fine-tunes a Random Forest Regressor** to find the optimal hyperparameters 
(Depth & Estimators), displays the tuning results, and then uses the best model to determine what drives your pace 
and predicts future performance.
""")

st.sidebar.header("Settings")
distance_unit = st.sidebar.radio("Select Distance Unit in your CSV:", ("Kilometers", "Miles"))

if distance_unit == "Kilometers":
    unit_label = "km"
    pace_label = "min/km"
    dist_full_label = "kilometers"
else:
    unit_label = "mile"
    pace_label = "min/mile"
    dist_full_label = "miles"

st.warning("⚠️ Please ensure your CSV has the following columns: **Date, Distance, Time, Avg Pace, Avg HR, Max HR, Avg Run Cadence**")


uploaded_file = st.file_uploader("Upload your activity log (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        
        df["Time (sec)"] = df["Time"].apply(time_to_seconds)
        df["Avg Pace (sec)"] = df["Avg Pace"].apply(pace_to_seconds)
        df['Date'] = pd.to_datetime(df['Date'], format='mixed')
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
            'Distance', 'Avg HR', 'Max HR', 'Avg Run Cadence',
            'days_since_start', 'run_count', 'cumulative_distance', 'cumulative_time',
            'distance_last_7days', 'distance_last_14days', 'distance_last_30days',
            'runs_last_7days', 'runs_last_30days',
            'avg_distance_last_10runs', 'avg_time_last_10runs',
            'days_since_last_run', 'avg_recovery_days',
            'avg_hr_last_5runs', 'avg_hr_last_10runs', 'max_hr_last_10runs',
            'hr_trend', 'hr_variability',
            'avg_cadence_last_5runs', 'avg_cadence_last_10runs', 'cadence_trend',
            'high_hr_runs_last_10',
        ]

        df_model = df.dropna(subset=feature_cols + ['Avg Pace (sec)'])
        X = df_model[feature_cols]
        y = df_model['Avg Pace (sec)']

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        st.divider()
        st.header("1. Fine-Tuning Model Hyperparameters")
        st.write("Training multiple models with different depths and estimator counts to find the best configuration...")

        estimators_list = [50, 100, 200]
        depths_list = [2, 4, 8, 16, 32, 64, 128]
        
        results = {est: {'train': [], 'test': []} for est in estimators_list}
        
        best_mae = float('inf')
        best_model = None
        best_params = {}

        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_iterations = len(estimators_list) * len(depths_list)
        current_iter = 0

        for n_est in estimators_list:
            for depth in depths_list:
                status_text.text(f"Training: {n_est} estimators, Depth {depth}...")
                
                model = RandomForestRegressor(
                    n_estimators=n_est,
                    max_depth=depth,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    max_features='sqrt',
                    random_state=42
                )
                
                model.fit(X_train, y_train)
                train_preds = model.predict(X_train)
                test_preds = model.predict(X_test)
                
                train_mae = mean_absolute_error(y_train, train_preds)
                test_mae = mean_absolute_error(y_test, test_preds)
                
                results[n_est]['train'].append(train_mae)
                results[n_est]['test'].append(test_mae)
                
                if test_mae < best_mae:
                    best_mae = test_mae
                    best_model = model
                    best_params = {'n_estimators': n_est, 'max_depth': depth}
                
                current_iter += 1
                progress_bar.progress(current_iter / total_iterations)

        status_text.text("Fine-tuning complete!")
        
        st.subheader("Fine-Tuning Results (MAE vs. Depth)")
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
        axes = [ax1, ax2, ax3]
        
        for idx, n_est in enumerate(estimators_list):
            ax = axes[idx]
            ax.plot(depths_list, results[n_est]['test'], 'r-o', label=f'{n_est} estimators (test)', linewidth=2, markersize=8)
            ax.plot(depths_list, results[n_est]['train'], 'b-o', label=f'{n_est} estimators (train)', linewidth=2, markersize=8)
            
            ax.set_xlabel('Max Depth', fontsize=12, fontweight='bold')
            if idx == 0:
                ax.set_ylabel(f'MAE (seconds/{unit_label})', fontsize=12, fontweight='bold')
            ax.set_title(f'{n_est} Estimators', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(depths_list)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.success(f"**Best Configuration Selected:** {best_params['n_estimators']} Estimators, Max Depth {best_params['max_depth']}")
        st.metric(f"Best Test MAE (sec/{unit_label})", f"{best_mae:.2f}")

        st.divider()
        st.header("2. What Drives Your Pace?")
        st.write("Based on the best selected model.")
        
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.bar_chart(feature_importance.set_index('feature').head(15))
        with col2:
            st.dataframe(feature_importance.head(15), hide_index=True)

        with st.expander("ℹ️ Feature Explanations"):
            st.markdown("""
            **Run Specific Metrics:**
            * `Distance`: The distance covered in a single run.
            * `Avg HR` / `Max HR`: Average and Maximum Heart Rate during the run.
            * `Avg Run Cadence`: The number of steps per minute.

            **Cumulative & History:**
            * `days_since_start`: Days elapsed since your first recorded run.
            * `run_count`: The total number of runs completed up to that point.
            * `cumulative_distance`: Total kilometers/miles run since day 1.
            * `cumulative_time`: Total time spent running since day 1.

            **Recent Training Load (Rolling Averages):**
            * `distance_last_7days` / `14days` / `30days`: Total distance covered in the respective past window.
            * `runs_last_7days` / `30days`: Number of runs in the past week or month.
            * `avg_distance_last_10runs`: Average distance per run over the last 10 sessions.
            * `avg_time_last_10runs`: Average duration per run over the last 10 sessions.
            * `days_since_last_run`: Number of rest days before the run.
            * `avg_recovery_days`: Average rest days taken between runs (last 10 runs).

            **Physiological Trends:**
            * `avg_hr_last_5runs` / `10runs`: Your average heart rate baseline over recent runs.
            * `max_hr_last_10runs`: Average peak heart rate over the last 10 runs.
            * `hr_trend`: The rate of change in your Avg HR (last 5 runs) - are you working harder?
            * `hr_variability`: Consistency of your heart rate over the last 10 runs.
            * `high_hr_runs_last_10`: Count of high-intensity runs (top 25% Max HR) in the last 10 sessions.
            * `avg_cadence_last_5runs` / `10runs`: Recent cadence baseline.
            * `cadence_trend`: Change in cadence over the last 5 runs.
            """)

        st.divider()
        st.header("3. 30-Day Pace Prediction")

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

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Current Fitness")
            st.write(f"**Current Pace:** {seconds_to_pace(current_pace)} {pace_label}")
            st.write(f"**Avg HR:** {recent_avg_hr:.0f} bpm")
            st.write(f"**Avg Cadence:** {recent_avg_cadence:.0f} spm")
        
        with col2:
            st.subheader("Recent Training (Last 30 Days)")
            st.write(f"**Total Runs:** {recent_runs}")
            st.write(f"**Total Distance:** {recent_total_distance:.1f} {dist_full_label}")
            st.write(f"**Avg Dist/Run:** {recent_distance_per_run:.2f} {dist_full_label}")

        future_row = last_row[feature_cols].copy()
        future_row['days_since_start'] += 30
        future_row['run_count'] += recent_runs
        future_row['cumulative_distance'] += recent_total_distance
        future_row['cumulative_time'] += last_30_days['Time (sec)'].sum()
        future_row['distance_last_30days'] = recent_total_distance

        predicted_pace_maintain = best_model.predict(pd.DataFrame([future_row]))[0]
        improvement_maintain = current_pace - predicted_pace_maintain

        st.subheader("Scenario: Maintain Current Training")
        st.info(f"""
        If you maintain {recent_runs} runs and {recent_total_distance:.0f} {dist_full_label} per month:
        
        **Predicted Pace:** {seconds_to_pace(predicted_pace_maintain)} {pace_label}
        
        **Expected Change:** {improvement_maintain:+.1f} sec/{unit_label} ({improvement_maintain/current_pace*100:+.2f}%)
        """)

        st.divider()
        st.header("4. Injury Risk Prediction")
        st.write("Based on your longest run in the last 30 days.")

        last_date_risk = df.index.max()
        thirty_days_ago_risk = last_date_risk - pd.Timedelta(days=30)
        recent_runs_risk = df[df.index >= thirty_days_ago_risk]

        if not recent_runs_risk.empty:
            longest_run_dist = recent_runs_risk['Distance'].max()
            st.write(f"**Longest run in the last 30 days:** {longest_run_dist:.2f} {dist_full_label}")

            planned_distance = st.number_input(
                f"Enter planned distance for next run ({unit_label}):", 
                min_value=0.0, 
                value=float(longest_run_dist), 
                step=0.1,
                format="%.2f"
            )

            if planned_distance > 0:
                ratio = planned_distance / longest_run_dist
                max_safe_dist = longest_run_dist * 1.10

                st.write(f"**Max Safe Distance (110%):** {max_safe_dist:.2f} {unit_label}")

                if ratio <= 1.10:
                    st.success("✅ **Low Risk.** This distance is within the safe range (≤ 110% of longest run).")
                elif 1.10 < ratio <= 1.30:
                    st.warning("⚠️ **64% Higher Injury Risk.** (110% - 130% of longest run).")
                elif 1.30 < ratio <= 2.00:
                    st.warning("⚠️ **52% Higher Injury Risk.** (130% - 200% of longest run).")
                else:
                    st.error("🚨 **128% Higher Injury Risk.** (> 200% of longest run).")
        else:
            st.warning("No runs found in the last 30 days to calculate risk.")

    except Exception as e:   
        st.error(f"Error processing file: {e}")
        st.write("Please ensure your CSV has the required columns: .")
else:
    st.info("Awaiting CSV file upload. Please upload 'activity_log.csv' or similar.")