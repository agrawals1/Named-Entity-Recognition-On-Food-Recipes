import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd

with open("cluster", 'rb') as f:
    clusters = pickle.load(f)
cluster_freq = {k:len(v) for (k,v) in clusters.items()}

# Get the top 15 most frequent clusters
dec_cluster_freq = sorted(cluster_freq, key=cluster_freq.get, reverse=True)
incr_cluster_freq = sorted(cluster_freq, key=cluster_freq.get)
incr_freqs = [cluster_freq[key] for key in incr_cluster_freq]
cumulative_freq = [sum(incr_freqs[:i+1]) for i in range(len(incr_freqs))]
top_cluster_freq = dec_cluster_freq[:10]
idxs = ['Cluster '+ str(i) for i in range(1,(len(top_cluster_freq)+1))] 

# Get the frequencies of the top clusters
freqs = [cluster_freq[key] for key in dec_cluster_freq]
top_freqs = [cluster_freq[key] for key in top_cluster_freq]

# Convert tuples to strings for labels
top_keys_str = [str(key) for key in top_cluster_freq]

# Custom colors for the pie slices
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#c2c2f0',
          '#d279a6', '#e6ac00', '#66cc99', '#ff7f00', '#996600', '#d5ff80', '#ffa64d', '#80dfff']

# Figure and axes configuration
fig, ax = plt.subplots(figsize=(8, 8))
_, _, autotexts = ax.pie(top_freqs, labels=idxs, colors=colors, startangle=90, shadow=False,
                         autopct=lambda x: '{:.2f}%'.format(x), textprops={'fontsize': 7, 'fontstyle': 'italic'})

# Setting font weight for percentages to bold
for autotext in autotexts:
    autotext.set_fontweight('bold')

# Customizing the pie chart
ax.axis('equal')  # Equal aspect ratio ensures a circular pie
plt.title('Top 10 Most Frequent Clusters', fontsize=16, pad=35, fontstyle='italic')



# Display the pie chart
plt.show()

fig.savefig('Top10ClustersPIE.svg', dpi=300, format='svg')
fig.savefig('Top10ClustersPIE.jpeg', dpi=300, format='jpeg')


# Custom colors for the bars
colors = ['#66b3ff', '#66cc99', '#ff9999', '#ffb3e6', '#99ff99', '#c2c2f0', '#ffcc99', '#d279a6', '#e6ac00', '#996600']

# Figure and axes configuration
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(idxs, top_freqs, color=colors)

# Customizing the bar plot
plt.title('Top 10 Most Frequent Clusters', fontsize=16, fontstyle='italic')
plt.xlabel('Cluster', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xticks(rotation=0)

# Adding labels to the bars
for i, freq in enumerate(top_freqs):
    ax.text(i, freq, str(freq), ha='center', va='bottom', fontweight='bold', fontsize=10)

# Display the bar plot
plt.show()
fig.savefig('Top10ClustersBAR.svg', dpi=300, format='svg')
fig.savefig('Top10ClustersBAR.jpeg', dpi=300, format='jpeg')





import pandas as pd
import matplotlib.pyplot as plt

# Create DataFrame for the table
data = pd.DataFrame(data=[list(key) for key in top_cluster_freq],
                    columns=['name', 'quantity', 'unit', 'df', 'state', 'size', 'temp'],
                    index=[i for i in range(1, 11)])

# Set row labels
data.index.name = 'Row'

# Figure and axes configuration
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')

# Create the table
table = ax.table(cellText=data.values,
                 colLabels=data.columns,
                 rowLabels=['Cluster '+ str(i) for i in range(1, 11)],
                 loc='center',
                 cellLoc='center',
                 colWidths=[0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# Customizing the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.2)  # Scale up the table size


for i, key in enumerate(data.index):
    for j, column in enumerate(data.columns):
        cell = table[i, j]
        cell.set_edgecolor('black')  # Set border color for cells
        cell.set_linewidth(1.2)  # Set border width for cells
        
# Set the title of the table
table_title = "Top 10 vectors of most frequent clusters"
table_title_props = {'fontsize': 16, 'fontstyle': 'italic'}
ax.set_title(table_title, **table_title_props)


# Save and show the table as an image
fig.savefig('Vectortable.svg', dpi=300, format='svg')
fig.savefig('Vectortable.jpeg', dpi=300, format='jpeg')
plt.show()






# Figure and axes configuration
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(range(1, len(clusters)+1), cumulative_freq, linestyle='-', color='blue')

# Customizing the line plot
plt.xlabel("Cluster Number", fontsize=12)
plt.ylabel("Cumulative Cluster Size", fontsize=12)
plt.title("Cluster vs Cumulative Cluster Size", fontsize=16, fontstyle='italic')
plt.tight_layout()

# Adjusting the font size of axes ticks
ax.tick_params(axis='both', which='both', labelsize=9)

# Display the line plot
plt.show()

fig.savefig('CumulativeClusterSize.svg', dpi=300, format='svg')
fig.savefig('CumulativeClusterSize.jpeg', dpi=300, format='jpeg')


