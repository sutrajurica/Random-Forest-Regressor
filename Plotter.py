fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))


ax1.plot(depths, mae_50_test, 'r-o', label='50 estimators (test)', linewidth=2, markersize=8)
ax2.plot(depths, mae_100_test, 'r-o', label='100 estimators (test)', linewidth=2, markersize=8)
ax3.plot(depths, mae_200_test, 'r-o', label='200 estimators (test)', linewidth=2, markersize=8)

ax1.plot(depths, mae_50_train, 'b-o', label='50 estimators (train)', linewidth=2, markersize=8)
ax2.plot(depths, mae_100_train, 'b-o', label='100 estimators (train', linewidth=2, markersize=8)
ax3.plot(depths, mae_200_train, 'b-o', label='200 estimators (train)', linewidth=2, markersize=8)

ax1.set_xlabel('Max Depth', fontsize=12, fontweight='bold')
ax1.set_ylabel('MAE (seconds/mile)', fontsize=12, fontweight='bold')
ax1.set_title('Test And Train Set Performance (Future Predictions)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(depths)

ax2.set_xlabel('Max Depth', fontsize=12, fontweight='bold')
ax2.set_ylabel('MAE (seconds/mile)', fontsize=12, fontweight='bold')
ax2.set_title('Test and Train Set Performance (Future Predictions)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(depths)

ax3.set_xlabel('Max Depth', fontsize=12, fontweight='bold')
ax3.set_ylabel('MAE (seconds/mile)', fontsize=12, fontweight='bold')
ax3.set_title('Test And TrainSet Performance (Future Predictions)', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(depths)
plt.tight_layout()
plt.savefig('rf_mae_vs_depth.png', dpi=300, bbox_inches='tight')
plt.show()
