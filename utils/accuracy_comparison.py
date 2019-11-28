import numpy as np
import matplotlib.pyplot as plt

# data to plot
n_groups = 3
acc_train = (0.88, 0.84, 0.82)
acc_test = (0.86, 0.82, 0.8)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.2
opacity = 0.6

rects1 = plt.bar(index, acc_train, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Train acc')

rects2 = plt.bar(index + bar_width, acc_test, bar_width,
                 alpha=opacity,
                 color='g',
                 label='Test acc')

plt.xlabel('Dataset')
plt.ylabel('Accuracies')
plt.title('Scores by dataset')
plt.xticks(index + bar_width, ('Lateral', 'Paralel', 'Single'))
plt.ylim(ymax=1, ymin=0.6)
plt.legend()

plt.tight_layout()
plt.show()