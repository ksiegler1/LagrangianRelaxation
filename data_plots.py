import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

df = pd.read_csv('run_data_200.csv')
# df = pd.read_csv('run_data_400.csv')
# df = pd.read_csv('run_data_700.csv')

# coverage plot
plt.plot(df['p_vals49'], df['pct_vals49'], color='red', label='49')
plt.plot(df['p_vals88'], df['pct_vals88'], color='blue', label='88')
plt.plot(df['p_vals150'], df['pct_vals150'], color='green', label='150')

plt.xlabel("Number of Facilities", size=22)
plt.ylabel("% Coverage", size=22)
plt.legend(fontsize=16, loc=2)
plt.savefig('coverage_200.png', format='png', dpi=400, bbox_inches='tight')
plt.show()

# # computational time
# plt.plot(df['p_vals49'], df['time_vals49'], color='blue', label='49')
# plt.xlabel("Number of Facilities (P)", size=22)
# plt.ylabel("Computational Time (sec)", size=22)
# # plt.legend(fontsize=16)
# # plt.savefig('computational_time.png', format='png', dpi=400, bbox_inches='tight')
# plt.show()
#
# # optimality gap
# plt.plot(df['p_vals49'], df['gap_vals49'], color='blue', label='49')
# plt.xlabel("Number of Facilities (P)", size=22)
# plt.ylabel("Optimality Gap (%)", size=22)
# # plt.legend(fontsize=16)
# # plt.savefig('optimality.png', format='png', dpi=400, bbox_inches='tight')
# plt.show()
