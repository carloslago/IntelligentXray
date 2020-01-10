import numpy as np
import matplotlib.pyplot as plt

# data to plot
n_groups = 4
acc_train = (0.9, 0.84, 0.87, 0.85)
acc_test = (0.87, 0.8, 0.8, 0.76)

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
plt.xticks(index + bar_width, ('Lateral', 'Frontal', 'Single', 'Paralel'))
plt.ylim(ymax=1, ymin=0.6)
plt.legend()

plt.tight_layout()
plt.show()