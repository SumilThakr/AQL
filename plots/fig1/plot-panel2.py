import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

os.chdir("../../")

result_df = pd.read_csv('plots/fig1/values-plusminus.csv')
result_df.head()

colors = ['#656664', '#A5A6A2', '#E4E6E1', '#B3291C']
labels = ['Land rents', 'Timber sales', 'Carbon storage', 'Air quality']

fig, ax = plt.subplots(1, figsize=(8, 8))
w = 0.3
spacing = 0.85
gap = 1.5
new_x = [0.0, spacing, spacing + gap, 2*spacing + gap, 3*spacing + gap]


plt.bar(new_x, result_df['rents_plus'], color = colors[0], width =w)
plt.bar(new_x, result_df['timber_plus'], bottom = result_df['rents_plus'], color = colors[1], width =w)
plt.bar(new_x, result_df['cseq_plus'], bottom = result_df['timber_plus'] + result_df['rents_plus'], color = colors[2], width =w)
plt.bar(new_x, result_df['aq_plus'], bottom = result_df['cseq_plus'] + result_df['rents_plus'] + result_df['timber_plus'], color = colors[3], width =w)

plt.bar(new_x, result_df['timber_minus'], color = colors[1], width =w)
plt.bar(new_x, result_df['rents_minus'], bottom = result_df['timber_minus'], color = colors[0], width =w)
plt.bar(new_x, result_df['cseq_minus'], bottom = result_df['rents_minus'] + result_df['timber_minus'], color = colors[2], width =w)
plt.bar(new_x, result_df['aq_minus'], bottom = result_df['rents_minus'] + result_df['cseq_minus'] + result_df['timber_minus'], color = colors[3], width =w)
#plt.bar(result_df.index, result_df['rents_minus'], color = colors[0], width =w)

#plt.bar(result_df.index, result_df['cseq'], color = '#DB4444', width =0.5)
#plt.bar(result_df.index, result_df['aq'], bottom = result_df['cseq'], color = '#E17979', width =0.5)



# x and y limits
plt.xlim(-0.2, 5.0)
plt.ylim(-1125, 525)
# remove spines
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Major ticks every 200, minor ticks every 100
major_ticks = np.arange(-1000, 525, 200)
minor_ticks = np.arange(-1100, 525, 100)
ax.yaxis.grid(which='minor', alpha=0.5, linestyle='dashed')
ax.yaxis.grid(which='major', alpha=0.9, linestyle='dashed')
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

#font1 = {'family':'Palatino','color':'black','size':12}
#grid
ax.set_axisbelow(True)
#ax.yaxis.grid(color='gray', linestyle='dashed', alpha=0.7)
# x ticks
xticks_labels = ['High crop \n demand','Low crop \ndemand', 'Forest \nincentives', 'Urban \ncontainment', 'Natural \nhabitat']
plt.xticks(new_x , labels = xticks_labels)
# title and legend
plt.legend(labels, ncol = 4, bbox_to_anchor=([1, 1.05, 0, 0]), frameon = False)
plt.axhline(y=0.0, color='black', linestyle='-', xmin = 0.0, xmax = 5.0)
#plt.title('Value', loc='left')
#plt.ylabel("Monetized value of ecosystem services ($)", fontdict=fon1)
plt.ylabel("Monetized value of ecosystem services (billion $)")

plt.vlines(x=(spacing)+(gap/2), ymin=-1100, ymax=500, colors='darkgrey', linestyle=':', lw=2)
#plt.axvline(x=2.5, color='black', linestyle='-', ymax = 800.0, ymin = -800.0)

#totnoaq
#plt.savefig('./plots/panel2.png', bbox_inches='tight',dpi=300)
plt.savefig('./plots/panel2.pdf', bbox_inches='tight',dpi=300)
#plt.show()
