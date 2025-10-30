# change all class labels to 0 in the yolo_runs ( having total 24 class labels )

import os

labels_dir = r"C:\\rishit\\bacteria_colony_count\\yolo_runs\\dataset_prepared\\labels"

for subset in ["train", "val"]:
    folder = os.path.join(labels_dir, subset)
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            path = os.path.join(folder, file)
            with open(path, "r") as f:
                lines = f.readlines()
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    parts[0] = "0"  # force all classes to 0
                    new_lines.append(" ".join(parts) + "\n")
            with open(path, "w") as f:
                f.writelines(new_lines)
print(" All labels converted to class 0.")
