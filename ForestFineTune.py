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