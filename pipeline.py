#! .venv/bin/python3

import click
import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit, cross_val_score, KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from mne import Epochs, events_from_annotations, pick_types, set_log_level
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.io import concatenate_raws, read_raw_edf

# from my_csp import CSP



def cross_validation(X, Y, pipeline):
	scores = []

	# Use scikit-learn Pipeline with cross_val_score function
	scores = cross_val_score(pipeline, X, Y, cv=KFold(10))

	# Fit the pipeline
	pipeline.fit(X, Y)

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
	return epochs, labels


def preprocess_data(raws):
	eegbci.standardize(raws)

	# Create a standard montage
	montage = make_standard_montage("standard_1020")
	raws.set_montage(montage)

	# Apply a filter to go from 8 to 32 Hz
	raws.filter(8, 30, fir_design="firwin", skip_by_annotation="edge")

	return raws


def fetch_data(subject, runs):
	raw_fnames = eegbci.load_data(subject, runs)
	raws = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
	return raws


def do (subject, runs, pipeline):
	raws_motor_left_right = preprocess_data(fetch_data(subject, runs[0]))
	raws_imagery_left_right = preprocess_data(fetch_data(subject, runs[1]))
	raws_motor_hands_feet = preprocess_data(fetch_data(subject, runs[2]))
	raws_imagery_hands_feet = preprocess_data(fetch_data(subject, runs[3]))

	# Get the epochs and labels for each run
	epochs_motor_left_right, labels_motor_left_right = create_epochs(raws_motor_left_right, dict(left=2, right=3))
	epochs_imagery_left_right, labels_imagery_left_right = create_epochs(raws_imagery_left_right, dict(left=2, right=3))
	epochs_motor_hands_feet, labels_motor_hands_feet = create_epochs(raws_motor_hands_feet, dict(hands=2, feet=3))
	epochs_imagery_hands_feet, labels_imagery_hands_feet = create_epochs(raws_imagery_hands_feet, dict(hands=2, feet=3))

	# Cross validation
	scores_mlr= cross_validation(epochs_motor_left_right, labels_motor_left_right, pipeline)
	scores_ilr= cross_validation(epochs_imagery_left_right, labels_imagery_left_right, pipeline)
	scores_mhf= cross_validation(epochs_motor_hands_feet, labels_motor_hands_feet, pipeline)
	scores_ihf= cross_validation(epochs_imagery_hands_feet, labels_imagery_hands_feet, pipeline)

	# average_score = np.mean([np.mean(scores_motor_left_right), np.mean(scores_imagery_left_right), np.mean(scores_motor_hands_feet), np.mean(scores_imagery_hands_feet)])
	average_score = np.mean([np.mean(scores_mlr), np.mean(scores_ilr), np.mean(scores_mhf), np.mean(scores_ihf)])
	print(f"Classification accuracy average for subject {subject}: {average_score}")
	return average_score

@click.command()
@click.option('-sr', '--subject_range', default=1, help='Number of subjects to evaluate')
def main(subject_range):
 
	runs_motor_left_right = [3, 7, 11]
	runs_imagery_left_right = [4, 8, 12]
	runs_motor_hands_feet = [5, 9, 13]
	runs_imagery_hands_feet = [6, 10, 14]
	runs = [runs_motor_left_right, runs_imagery_left_right, runs_motor_hands_feet, runs_imagery_hands_feet]
	
	# pipeline_rf = make_pipeline(CSP(n_components=10), RandomForestClassifier(n_estimators=150, random_state=42))
	# print(f'\n----- Pipeline RF -----')
	# average_scores_rf = [do(i, runs, pipeline_rf) for i in range(1, subject_range + 1)]
	# print(f'Average scores RF: {np.mean(average_scores_rf)}')

	pipeline = make_pipeline(CSP(), StandardScaler(), LogisticRegression())
	average_scores = []
	for i in range (1, subject_range + 1):
		average_scores.append(do(i, runs, pipeline))
		print(f'Current average scores LDA shrinkage: {np.mean(average_scores)}')
	
	print(f'Final average scores LDA shrinkage: {np.mean(average_scores)}')
	
	# pipeline_lda_shrinkage = make_pipeline(CSP(), LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr"))
	# print(f'\n----- Pipeline LDA shrinkage -----')
	# average_scores_lda_shrinkage = [do(i, runs, pipeline_lda_shrinkage) for i in range(1, subject_range + 1)]
	# print(f'Average scores LDA shrinkage: {np.mean(average_scores_lda_shrinkage)}')
	

if __name__ == "__main__":
	set_log_level("ERROR")
	main()