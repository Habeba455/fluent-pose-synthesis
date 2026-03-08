from pose_format import Pose
import glob
import numpy as np
from tqdm import tqdm
import statistics

# -----------------------------------------
# DATASET PATH
# -----------------------------------------

DATASET_PATH = "/Users/habebaftooh/Desktop/sign pose/pose-diffusion-model/assets/dataset/**/*.pose"

files = glob.glob(DATASET_PATH, recursive=True)

print("\n==============================")
print("DATASET FILE SCAN")
print("==============================")
print("TOTAL POSE FILES:", len(files))

if len(files) == 0:
    print(" No pose files found. Check dataset path.")
    exit()


# -----------------------------------------
# DATASET ANALYSIS
# -----------------------------------------

lengths = []
keypoints_list = []
dims_list = []

nan_files = 0
zero_frames = 0

for f in tqdm(files):

    with open(f, "rb") as fp:
        pose = Pose.read(fp)

    data = pose.body.data

    # remove people dimension
    if data.ndim == 4:
        data = data[:,0]

    frames, keypoints, dims = data.shape

    lengths.append(frames)
    keypoints_list.append(keypoints)
    dims_list.append(dims)

    if np.isnan(data).any():
        nan_files += 1

    # check empty frames
    for frame in data:
        if np.all(frame == 0):
            zero_frames += 1


# -----------------------------------------
# REPORT
# -----------------------------------------

print("\n==============================")
print("DATASET REPORT")
print("==============================")

print("MIN FRAMES:", min(lengths))
print("MAX FRAMES:", max(lengths))
print("AVG FRAMES:", round(statistics.mean(lengths),2))

print("\nKEYPOINTS FOUND:", set(keypoints_list))
print("DIMS FOUND:", set(dims_list))

print("\nFILES WITH NAN:", nan_files)
print("EMPTY FRAMES:", zero_frames)

print("\n==============================")
print("FRAME LENGTH DISTRIBUTION")
print("==============================")

print(" <10 frames:", len([x for x in lengths if x < 10]))
print(" 10-20 frames:", len([x for x in lengths if 10 <= x < 20]))
print(" 20-50 frames:", len([x for x in lengths if 20 <= x < 50]))
print(" 50-100 frames:", len([x for x in lengths if 50 <= x < 100]))
print(" >100 frames:", len([x for x in lengths if x >= 100]))


# -----------------------------------------
# RECOMMENDED TRAINING SETTINGS
# -----------------------------------------

avg_frames = statistics.mean(lengths)

print("\n==============================")
print("RECOMMENDED TRAINING SETTINGS")
print("==============================")

if avg_frames < 15:
    print("Recommended chunk_len: 8")
elif avg_frames < 30:
    print("Recommended chunk_len: 12")
elif avg_frames < 60:
    print("Recommended chunk_len: 20")
else:
    print("Recommended chunk_len: 40")

print("Recommended diffusion_steps: 100")

print("\nDataset health check:")

if nan_files == 0:
    print(" No NaN values")
else:
    print("NaN values found")

if zero_frames == 0:
    print(" No empty frames")
else:
    print(" Empty frames detected")

print("\n==============================")
print("DATASET CHECK COMPLETE")
print("==============================")