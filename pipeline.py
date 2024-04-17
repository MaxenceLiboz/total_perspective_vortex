#! .venv/bin/python3

import click
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.pipeline import Pipeline

from mne import Epochs, events_from_annotations, pick_types, annotations_from_events
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.io import concatenate_raws, read_raw_edf



def cross_validation(epochs, labels):
	scores = []
	epochs_data = epochs.get_data(copy=False)
	cv = ShuffleSplit(10, test_size=0.2, random_state=42)

	# Assemble a classifier
	lda = LinearDiscriminantAnalysis(shrinkage="auto", solver="eigen")
	csp = CSP()

	# Use scikit-learn Pipeline with cross_val_score function
	clf = Pipeline([("CSP", csp), ("LDA", lda)])
	scores = cross_val_score(clf, epochs_data, labels, cv=cv, n_jobs=None, verbose=False)

	return scores


def create_epochs(raw, event_id, tmin = -1.0, tmax = 4.0):
	# Get the events
	events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

	# Define the picks
	picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

	# Define the epochs
	epochs = Epochs(
		raw,
		events,
		event_id,
		tmin,
		tmax,
		proj=True,
		picks=picks,
		baseline=None,
		preload=True,
	)

	# Get labels form epochs
	labels = epochs.events[:, -1] - 1
	# epochs.plot(block=True, n_channels=10, scalings={"eeg": 60e-6})
	epochs_train = epochs.copy()
	# epochs_train.plot(block=True, n_channels=10, scalings={"eeg": 60e-6})
	return epochs_train, labels


def preprocess_data(raws):
	eegbci.standardize(raws)

	# Create a standard montage
	montage = make_standard_montage("standard_1020")
	raws.set_montage(montage)

	# Apply a filter to go from 7 to 32 Hz
	raws.filter(8, 30, fir_design="firwin", skip_by_annotation="edge")

	return raws


def fetch_data(subject, runs):
	raw_fnames = eegbci.load_data(subject, runs)
	raws = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
	return raws


def do (subject, runs):

	raws_motor_left_right = preprocess_data(fetch_data(subject, runs[0]))
	raws_imagery_left_right = preprocess_data(fetch_data(subject, runs[1]))
	raws_motor_hands_feet = preprocess_data(fetch_data(subject, runs[2]))
	raws_imagery_hands_feet = preprocess_data(fetch_data(subject, runs[3]))

	# Show the standard montage
	# raws_motor_left_right.plot_sensors(show_names=True, block=True, sphere=(0, 0.015, 0, 0.095))

	# Get the epochs and labels for each run
	epochs_motor_left_right, labels_motor_left_right = create_epochs(raws_motor_left_right, dict(left=2, right=3))
	epochs_imagery_left_right, labels_imagery_left_right = create_epochs(raws_imagery_left_right, dict(left=2, right=3))
	epochs_motor_hands_feet, labels_motor_hands_feet = create_epochs(raws_motor_hands_feet, dict(hands=2, feet=3))
	epochs_imagery_hands_feet, labels_imagery_hands_feet = create_epochs(raws_imagery_hands_feet, dict(hands=2, feet=3))

	# Cross validation
	scores_motor_left_right = cross_validation(epochs_motor_left_right, labels_motor_left_right)
	scores_imagery_left_right = cross_validation(epochs_imagery_left_right, labels_imagery_left_right)
	scores_motor_hands_feet = cross_validation(epochs_motor_hands_feet, labels_motor_hands_feet)
	scores_imagery_hands_feet = cross_validation(epochs_imagery_hands_feet, labels_imagery_hands_feet)

	# Printing the results
	print(f"\n----------- Classification accuracy motor left right: {np.mean(scores_motor_left_right)} -----------\n")
	print(f"\n----------- Classification accuracy imagery left right: {np.mean(scores_imagery_left_right)} -----------\n")
	print(f"\n----------- Classification accuracy motor hands feet: {np.mean(scores_motor_hands_feet)} -----------\n")
	print(f"\n----------- Classification accuracy imagery hands feet: {np.mean(scores_imagery_hands_feet)} -----------\n")

	average_score = np.mean([np.mean(scores_motor_left_right), np.mean(scores_imagery_left_right), np.mean(scores_motor_hands_feet), np.mean(scores_imagery_hands_feet)])
	print(f"\n----------- Classification accuracy average: {average_score} -----------\n")
	return average_score

@click.command()
@click.option('-sr', '--subject_range', default=1, help='Number of subjects to evaluate')
def main(subject_range):
 
	runs_motor_left_right = [3, 7, 11]
	runs_imagery_left_right = [4, 8, 12]
	runs_motor_hands_feet = [5, 9, 13]
	runs_imagery_hands_feet = [6, 10, 14]
	runs = [runs_motor_left_right, runs_imagery_left_right, runs_motor_hands_feet, runs_imagery_hands_feet]

	# Assuming average_scores is a list of scores
	average_scores = [do(i, runs) for i in range(1, subject_range + 1)]

	# Create a list for the number of epochs
	epochs = list(range(1, subject_range + 1))

	index_max = np.argmax(average_scores)
	print(f"Max value [{index_max}]: {average_scores[index_max]}")
	print(average_scores)
	print(f'Total average score: {np.mean(average_scores)}')

	#Create the plot
	plt.figure(figsize=(10, 6))
	plt.plot(epochs, average_scores, marker='o')

	# Add title and labels
	plt.title('Average Scores Over Epochs')
	plt.xlabel('Epochs')
	plt.ylabel('Average Score')

	# Show the plot
	plt.show()


if __name__ == "__main__":
	main()
     
	# raw.plot(block=True, duration=60, n_channels=10, scalings={"eeg": 300e-6})
 
	
	# raw.plot(block=True, duration=20, n_channels=10, scalings={"eeg": 60e-6})

	# raw.plot_sensors(show_names=True, block=True, sphere=(0, 0.015, 0, 0.095))
	# raw.plot(block=True, duration=20, n_channels=10, scalings={"eeg": 60e-6})
 
	# subjects = 42 
	# runs_motor_left_right = [3, 7, 11]
	# runs_imagery_left_right = [4, 8, 12]
	# runs_motor_hands_feet = [5, 9, 13]
	# runs_imagery_hands_feet = [6, 10, 14]
	# runs = [runs_motor_left_right, runs_imagery_left_right, runs_motor_hands_feet, runs_imagery_hands_feet]

	# num_epochs = 10
	# # Assuming average_scores is a list of scores
	# average_scores = [do(i, runs) for i in range(1, num_epochs)]

	# # Create a list for the number of epochs
	# epochs = list(range(1, num_epochs))

	# index_max = np.argmax(average_scores)
	# print(f"Max value [{index_max}]: {average_scores[index_max]}")
	# print(average_scores)
	# print(f'Total average score: {np.mean(average_scores)}')

	# #Create the plot
	# plt.figure(figsize=(10, 6))
	# plt.plot(epochs, average_scores, marker='o')

	# # Add title and labels
	# plt.title('Average Scores Over Epochs')
	# plt.xlabel('Epochs')
	# plt.ylabel('Average Score')

	# # Show the plot
	# plt.show()


 
	# raws_motor_left_right = preprocess_data(fetch_data(subjects, runs_motor_left_right))
	# raws_imagery_left_right = preprocess_data(fetch_data(subjects, runs_imagery_left_right))
	# raws_motor_hands_feet = preprocess_data(fetch_data(subjects, runs_motor_hands_feet))
	# raws_imagery_hands_feet = preprocess_data(fetch_data(subjects, runs_imagery_hands_feet))

	# # Show the standard montage
	# # raws_motor_left_right.plot_sensors(show_names=True, block=True, sphere=(0, 0.015, 0, 0.095))

	# # Get the epochs and labels for each run
	# epochs_motor_left_right, labels_motor_left_right = creat+e_epochs(raws_motor_left_right, dict(left=2, right=3))
	# epochs_imagery_left_right, labels_imagery_left_right = create_epochs(raws_imagery_left_right, dict(left=2, right=3))
	# epochs_motor_hands_feet, labels_motor_hands_feet = create_epochs(raws_motor_hands_feet, dict(hands=2, feet=3))
	# epochs_imagery_hands_feet, labels_imagery_hands_feet = create_epochs(raws_imagery_hands_feet, dict(hands=2, feet=3))

	# # Montecarlo cross validation
	# scores_motor_left_right = cross_validation(epochs_motor_left_right, labels_motor_left_right)
	# scores_imagery_left_right = cross_validation(epochs_imagery_left_right, labels_imagery_left_right)
	# scores_motor_hands_feet = cross_validation(epochs_motor_hands_feet, labels_motor_hands_feet)
	# scores_imagery_hands_feet = cross_validation(epochs_imagery_hands_feet, labels_imagery_hands_feet)

	# # Printing the results
	# print(f"\n----------- Classification accuracy motor left right: {np.mean(scores_motor_left_right)} -----------\n")
	# print(f"\n----------- Classification accuracy imagery left right: {np.mean(scores_imagery_left_right)} -----------\n")
	# print(f"\n----------- Classification accuracy motor hands feet: {np.mean(scores_motor_hands_feet)} -----------\n")
	# print(f"\n----------- Classification accuracy imagery hands feet: {np.mean(scores_imagery_hands_feet)} -----------\n")
	# print(f"\n----------- Classification accuracy average: {np.mean([np.mean(scores_motor_left_right), np.mean(scores_imagery_left_right), np.mean(scores_motor_hands_feet), np.mean(scores_imagery_hands_feet)])} -----------\n")


	# # plot CSP patterns estimated on full data for visualization
	# csp.fit_transform(epochs_data, labels)

	# csp.plot_patterns(epochs.info, ch_type="eeg", units="Patterns (AU)", size=1.5)

	# sfreq = raw.info["sfreq"]
	# w_length = int(sfreq * 0.5)  # running classifier: window length
	# w_step = int(sfreq * 0.1)  # running classifier: window step size
	# w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

	# scores_windows = []

	# for train_idx, test_idx in cv_split:
	# 	y_train, y_test = labels[train_idx], labels[test_idx]

	# 	X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
	# 	X_test = csp.transform(epochs_data_train[test_idx])

	# 	# fit classifier
	# 	lda.fit(X_train, y_train)

	# 	# running classifier: test classifier on sliding window
	# 	score_this_window = []
	# 	for n in w_start:
	# 		X_test = csp.transform(epochs_data[test_idx][:, :, n : (n + w_length)])
	# 		score_this_window.append(lda.score(X_test, y_test))
	# 	scores_windows.append(score_this_window)

	# # Plot scores over time
	# w_times = (w_start + w_length / 2.0) / sfreq + epochs.tmin

	# plt.figure()
	# plt.plot(w_times, np.mean(scores_windows, 0), label="Score")
	# plt.axvline(0, linestyle="--", color="k", label="Onset")
	# plt.axhline(0.5, linestyle="-", color="k", label="Chance")
	# plt.xlabel("time (s)")
	# plt.ylabel("classification accuracy")
	# plt.title("Classification score over time")
	# plt.legend(loc="lower right")
	# plt.show()
	