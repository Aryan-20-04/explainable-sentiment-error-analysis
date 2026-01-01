import os
import csv
from collections import Counter
import matplotlib.pyplot as plt

def generate_error_stats(misclassified, output_dir="analysis"):
    if not misclassified:
        print("No misclassified samples to analyze.")
        return
    os.makedirs(output_dir, exist_ok=True)
    
    error_types = [item[4]for item in misclassified]
    error_counter = Counter(error_types)
    
    print("\n==== Error Distribution ====")
    for error, count, in error_counter.most_common():
        print(f"{error:<25} : {count}")
        
    csv_path = os.path.join(output_dir, "error_distribution.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Error Type", "Count"])
        for error, count in error_counter.most_common():
            writer.writerow([error, count])
    print(f"\nError distribution saved to {csv_path}")
    
    errors = list(error_counter.keys())
    counts = list(error_counter.values())
    plt.figure(figsize=(8, 5))
    plt.bar(errors, counts)
    plt.xlabel("Error Type")
    plt.ylabel("Number of Misclassifications")
    plt.title("Distribution of Transformer Sentiment Model Errors")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "error_distribution.png")
    plt.savefig(plot_path)
    plt.show()

    print(f"Error distribution plot saved to {plot_path}")