import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir("../../")

result_df = pd.read_csv('plots/fig1/barvals.csv')
print(result_df.head())

#colors = ['#999999', '#8390FA', '#333333', '#ff5252']
colors = ['#041b33','#eaece9']
labels = ['with air quality', 'without air quality']

fig, ax = plt.subplots(1, figsize=(10, 8))
w = 0.4

result_df.plot(x="scenario", y=["tot", "totnoaq"], color = [colors[0],colors[1]], kind="bar", ax=ax)

# x and y limits
plt.xlim(-0.2, 5.0)
plt.ylim(-910, 450)
# remove spines
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)

#font1 = {'family':'Palatino','color':'black','size':12}
#grid
ax.set_axisbelow(True)

# Major ticks every 200, minor ticks every 100
major_ticks = np.arange(-800, 500, 200)
minor_ticks = np.arange(-900, 500, 100)
ax.yaxis.grid(which='minor', alpha=0.5, linestyle='dashed')
ax.yaxis.grid(which='major', alpha=0.9, linestyle='dashed')
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)


# x ticks
xticks_labels = ['High crop \ndemand', 'Low crop \ndemand', 'Forest \nincentives', 'Urban \ncontainment', 'Natural habitat']
plt.xticks(result_df.index , labels = xticks_labels,rotation = 360)

plt.legend(labels, ncol = 4, bbox_to_anchor=([1, 1.05, 0, 0]), frameon = False)
plt.axhline(y=0.0, color='black', linestyle='-', xmin = 0.0, xmax = 1.5)
#plt.hlines(y=0.0, xmin=-0.2, xmax=1.5, colors='black', linestyle='-', lw=2)
#plt.hlines(y=433.893186, xmin=1.5, xmax=5.0, colors='black', linestyle='-', lw=2)
#plt.title('Value', loc='left')
#plt.ylabel("Monetized value of ecosystem services ($)", fontdict=fon1)
plt.ylabel("Monetized value of ecosystem services (billion $)")
plt.xlabel("")

plt.vlines(x=1.5, ymin=-900, ymax=400, colors='darkgrey', linestyle=':', lw=2)

#plt.savefig('./plots/panel1.png', bbox_inches='tight',dpi=300)
plt.savefig('./plots/panel1.pdf', bbox_inches='tight',dpi=300)
