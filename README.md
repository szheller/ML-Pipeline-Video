# ML-Pipeline-Video

Source Video file: https://youtu.be/7-O-mmEihA8

This Project exhibits use of an ML Pipeline to start with a base
video file of a sports game, volleyball, and pulls in tracking
data of the frames that correspond to the video file from a csv
file.

Then, taking in decisions of active/inactive frames from a target
csv file, there is training of the AI Model using the target data
and predictions of the target data as well.  These predictions are
then used to reduce the video file to include only those frames where
the active status has been predicted, with additional features such
as smoothing incorporated.

The resulting output video is found at:

