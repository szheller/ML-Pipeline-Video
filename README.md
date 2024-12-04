# ML-Pipeline-Video

Source Video file: https://youtu.be/Q9lme_fVFug

This Project exhibits use of an ML Pipeline to start with a base
video file of a sports game, volleyball, and pulls in tracking
data of the frames that correspond to the video file from a csv
file.

Then, taking in decisions of active/inactive frames from a target
csv file, there is training of the AI Model using the target data
and predictions of the target data as well.  These predictions are
then used to reduce the video file to include only those frames where
the active status has been predicted, with additional features such
as transitions incorporated.

Included in the predictions are sudden change in direction of the ball.

Final Project:

		Read in provided_data.csv, frames matching video

		Detect sudden change in direction, write out to sudden_change.csv

		Read in provided_data.csv, also read in sudden_change.csv & merge

		Create lagged features for time series data

		Split data into training & test using the best Window size,
		feed it to Classifier using best n_estimators.

		Create predictions of the test data & write to predictions.csv
		
		Read in predictions.csv
  
		Apply filters and transitions to frames in predictions.csv
		& write out to output video.

Feature engineering steps taken:

	Added feature sudden change in direction, because it should
	indicate when some action occurred that would indicate which
	side has the ball or may be making progress.  To do this added
	a column for sudden_change to the target data for the frames
	in the DataFrame, and used this combined data as the training
	data for the model & test data for the predictions and comparison
	of actual vs. predicted results.

Optimization approaches used:

	Trying different window sizes and estimators using the classifier
	model RandomForestClassifier, found the best number of windows was
	100, and best number of estimators was 10.  Also tried another
	classifier, AdaBoostClassifier, but it had a lower score of 0.853.

Numerical results showing the improvements from these changes:

WITHOUT additional feature (sudden change in direction) added:

(base) PS C:\Steve\OpenAvenues\Project> python .\time_classification.py
['xc', 'yc', 'w', 'h', 'effort']
              precision    recall  f1-score   support

           0       0.90      0.94      0.92     23008
           1       0.68      0.56      0.61      5578

    accuracy                           0.86     28586
   macro avg       0.79      0.75      0.76     28586
weighted avg       0.85      0.86      0.86     28586


WITH additional feature (sudden change in direction) added:

(base) PS C:\Steve\OpenAvenues\Project> python .\time_classification.py
['xc', 'yc', 'w', 'h', 'effort', 'sudden_change']
              precision    recall  f1-score   support

           0       0.90      0.94      0.92     23008
           1       0.68      0.57      0.62      5578

    accuracy                           0.86     28586
   macro avg       0.79      0.75      0.77     28586
weighted avg       0.86      0.86      0.86     28586

There was not much improvement, both with and without the added
feature have 0.86 average score, and both with and without the
feature (sudden change in direction) have 0 predicted with 0.92.
But 1 was predicted with score 0.62 with the feature while without
the feature, 1 was only predicted with score 0.61.


Installation:

Install Anaconda:

Windows:

- Download & run the Conda installer for Windows from 	https://docs.conda.io/en/latest/miniconda.html

- Open the Anaconda Prompt from the Start menu to use Conda

macOS:

- Open Terminal & run the following commands:
	curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
	bash Miniconda3-latest-MacOSX-x86_64.sh

- Restart your terminal or run source ~/.bash_profile

Linux:

- Open Terminal & run the following commands:
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	bash Miniconda3-latest-Linux-x86_64.sh

- Restart your terminal or run source ~/.bashrc


Setting Up the Conda Environment

- Open a terminal (Anaconda Prompt on Windows) to the directory containing environment.yaml

- Run the following command:
	conda env create -f environment.yaml

- Activate the new environment:
	conda activate mleng_env

---
To run this Project, follow these steps:

After Anaconda has been installed & activated following the Installation
steps above, open a terminal (Anaconda Prompt on Windows), then run below:

- python data_analysis2.py Generates sudden_change.csv from provided_data.csv
- python time_classification.py Generates predictions.csv from sudden_change.csv & target.csv
- python filter_predictions.py Generates smoothed_predictions.csv from predictions.csv, also creates predictions_comparison.png showing smoothing added.
- python video_transition.py generates output video in transition.mp4 from smoothed_predictions.csv & video.mp4


The resulting output video is found at:

  Output Video file:  <Comming Soon - 10 hours so far, 18 hours total to finish generating>

 ---


