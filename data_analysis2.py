import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Create your DataFrame (or read it from a CSV file)
data = pd.read_csv('provided_data.csv')

# Display the first 5 rows
print(data.head())

# Display basic information about the dataset
print(data.info())

# Calculate and print summary statistics
print(data.describe())

# Rename columns for readability
data.columns = ['frame', 'x_center', 'y_center', 'width', 'height', 'effort']

# Calculate the change in x and y between consecutive frames
data['dx'] = data['x_center'].diff()
data['dy'] = data['y_center'].diff()

# Calculate the angle (in degrees) between consecutive motion vectors
data['angle_change'] = np.degrees(np.arctan2(data['dy'], data['dx']).diff().abs())

# Define a threshold for a "sudden change in direction" (adjust as needed)
threshold = 345  # degrees

# Identify frames with a sudden change in direction
data['sudden_change'] = data['angle_change'] > threshold

# Display the results
print(data[['frame', 'x_center', 'y_center', 'dx', 'dy', 'angle_change', 'sudden_change']])

print("data.iloc[:, 0]=", len(data.iloc[:, 0]))
print("data['sudden_change']=", len(data['sudden_change']))

# Calculate the derivative of the first column
derivative = np.diff(data.iloc[:, 1])

# Plotting
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 5), sharex=True)

# Plot original data
ax1.plot(data.iloc[:, 0], data.iloc[:, 1])
ax1.set_ylabel('Value')
ax1.set_title('Second Column vs Frame Number')
ax1.grid(True)

# Plot derivative
ax2.plot(data.iloc[1:, 0], derivative)
ax2.set_xlabel('Frame Number')
ax2.set_ylabel('Derivative')
ax2.set_title('Derivative of Second Column vs Frame Number')
ax2.grid(True)

# Plot sudden change
ax3.plot(data.iloc[:, 0], data['sudden_change'])
ax3.set_xlabel('Frame Number')
ax3.set_ylabel('sudden_change')
ax3.set_title('sudden_change of > ' + str(threshold) + ' degrees vs Frame Number')
ax3.grid(True)

plt.tight_layout()
#plt.savefig('plot2.png')
#plt.show()

# Choose entries with sudden_change True
df_new = data[data['sudden_change'] == True]

# Display the first 5 rows
print(df_new.head())

# Display basic information about the dataset
print(df_new.info())

# Calculate and print summary statistics
print(df_new.describe())

print("df_new.iloc[:, 0]=", len(df_new.iloc[:, 0]))
print("df_new['sudden_change']=", len(df_new['sudden_change']))

def create_animation(df):
    # Set up the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('animation_sudden_change.mp4', fourcc, 30.0, (800, 600))

    # Normalize coordinates to fit within the frame
    x_min, x_max = df.iloc[:, 1].min(), df.iloc[:, 1].max()
    y_min, y_max = df.iloc[:, 2].min(), df.iloc[:, 2].max()
    
    for _, row in df.iterrows():
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Normalize and scale coordinates
        x = int((row.iloc[1] - x_min) / (x_max - x_min) * 780 + 10)
        y = int((row.iloc[2] - y_min) / (y_max - y_min) * 580 + 10)
        
        # Draw the point
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        # Add frame number text
        cv2.putText(frame, f"Frame: {int(row.iloc[0])}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()

# Create the animation
#create_animation(df_new)

#print("Animation saved as 'animation_sudden_change.mp4'")

# Write predictions to CSV with the same syntax as target.csv
predictions_df = pd.DataFrame({'frame': df_new['frame'], 'value': map(int, df_new['sudden_change'])})
predictions_df.to_csv('sudden_change.csv', index=False)
print("Predictions written to sudden_change.csv.  Done.")


