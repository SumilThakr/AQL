import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir("../../")

result_df = pd.read_csv('plots/fig1/values-ludeaths.csv')
result_df.head()

colors = ['#79D021', '#1F3D0C', '#D9D40C']
labels = ['Pasture', 'Forest', 'Cropland']

fig, ax = plt.subplots(1, figsize=(8, 8))
w = 0.3
spacing = 0.85
gap = 1.5
new_x = [0.0, spacing, spacing + gap, 2*spacing + gap, 3*spacing + gap]


plt.bar(new_x, result_df['pasturelu'], color = colors[0], width =w)
plt.bar(new_x, result_df['forestlu'], bottom = result_df['pasturelu'], color = colors[1], width =w)
plt.bar(new_x, result_df['croplu'], bottom = result_df['forestlu'] + result_df['pasturelu'], color = colors[2], width =w)

plt.bar(new_x, result_df['pasturelu_minus'], color = colors[0], width =w)
plt.bar(new_x, result_df['croplu_minus'], bottom = result_df['pasturelu_minus'], color = colors[2], width =w)


# x and y limits
plt.xlim(-0.2, 5.0)
plt.ylim(-2300, 6000)
# remove spines
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Major ticks every 200, minor ticks every 100
major_ticks = np.arange(-2000, 6000, 1000)
minor_ticks = np.arange(-2000, 6000, 500)
ax.yaxis.grid(which='minor', alpha=0.5, linestyle='dashed')
ax.yaxis.grid(which='major', alpha=0.9, linestyle='dashed')
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

#font1 = {'family':'Palatino','color':'black','size':12}
#grid
ax.set_axisbelow(True)
#ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.7)
# x ticks
xticks_labels = ['High crop \n demand', 'Low crop \ndemand', 'Forest \nincentives', 'Urban \ncontainment', 'Natural \nhabitat']
plt.xticks(new_x , labels = xticks_labels)
# title and legend
plt.legend(labels, ncol = 4, bbox_to_anchor=([1, 1.05, 0, 0]), frameon = False)
plt.axhline(y=0.0, color='black', linestyle='-', xmin = 0.0, xmax = 5.0)
#plt.title('Value', loc='left')
#plt.ylabel("Monetized value of ecosystem services ($)", fontdict=fon1)
plt.ylabel("Change in annual air quality-related deaths")

plt.vlines(x=(spacing)+(gap/2), ymin=-2300, ymax=6000, colors='darkgrey', linestyle=':', lw=2)
#plt.axvline(x=2.5, color='black', linestyle='-', ymax = 800.0, ymin = -800.0)

#totnoaq
#plt.savefig('./plots/panel3.png', bbox_inches='tight',dpi=300)
plt.savefig('./plots/panel3.pdf', bbox_inches='tight',dpi=300)
#plt.show()
