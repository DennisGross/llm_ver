import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV data into a DataFrame
data = pd.read_csv("top_25_10_prob_analysis.csv", index_col=0)
# Remove meta-llama/Llama-2-13b-hf column
data = data.drop(columns=["meta-llama/Llama-2-13b-hf"])

# Plotting
plt.figure(figsize=(14, 8))  # Set figure size
for column in data.columns:
    plt.plot(data.index, data[column], marker='o', label=str(column).split("/")[1] if column != "bert-base-uncased" else column)

# Customizing the plot
#plt.title('Model Performance Comparison', fontsize=20)
plt.xlabel('Top k', fontsize=14)
plt.ylabel('Probability', fontsize=14)
plt.xticks(data.index)  # Ensure all index points are marked
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid for better readability
plt.legend(title='Models', title_fontsize='13', fontsize='12', bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside
# Adjusting y-ticks to better fit the squeezed height
plt.yticks(ticks=plt.yticks()[0], labels=[f"{tick:.2f}" for tick in plt.yticks()[0]])

plt.tight_layout()  # Adjust layout to not cut off content

plt.savefig("your_file_path.png")
