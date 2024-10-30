import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv("shoot_angles_with_coordinates.csv")

# List all keypoints with their respective CSV column indices
keypoints = {
    "left_shoulder": (5, 6),
    "right_shoulder": (7, 8),
    "left_elbow": (9, 10),
    "right_elbow": (11, 12),
    "left_hip": (13, 14),
    "right_hip": (15, 16),
    "left_knee": (13, 14),
    "right_knee": (15, 16)
    # Add any other keypoints if they’re included in the CSV
}

# Function to draw the skeleton with all keypoints labeled
def draw_skeleton(row):
    fig, ax = plt.subplots()

    # Plot each keypoint with a label
    for part, (x_col, y_col) in keypoints.items():
        x = row[x_col]
        y = row[y_col]
        ax.plot(x, y, 'bo')  # Blue circle for each keypoint
        ax.text(x, y, part, fontsize=8, ha='right')  # Label each keypoint
    
    # Connect specific keypoints with lines to form a skeleton
    ax.plot([row[5], row[9]], [row[6], row[10]], 'r-')  # Left shoulder to left elbow
    ax.plot([row[7], row[11]], [row[8], row[12]], 'r-')  # Right shoulder to right elbow
    ax.plot([row[5], row[13]], [row[6], row[14]], 'r-')  # Left shoulder to left hip
    ax.plot([row[7], row[15]], [row[8], row[16]], 'r-')  # Right shoulder to right hip
    ax.plot([row[13], row[15]], [row[14], row[16]], 'r-')  # Left hip to right hip

    # Set axis limits and invert y-axis for correct orientation
    plt.xlim(0, 640)  # Adjust based on video frame width
    plt.ylim(0, 480)  # Adjust based on video frame height
    plt.gca().invert_yaxis()
    plt.show()

# Loop through each frame in the CSV and plot all keypoints
for _, row in data.iterrows():
    draw_skeleton(row)
