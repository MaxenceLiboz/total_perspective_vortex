#! .venv/bin/python3

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


if __name__ == "__main__":
     
	# files_name = eegbci.load_data(1, [6, 10, 14], "./datasets")

	# subject = []
	# for file in files_name:
	# 	# subject.append(read_raw_edf(file))
	# 	raw = read_raw_edf(file)
	# 	# annot = read_annotations(file)
	# 	# raw.set_annotations(annot)
	# 	subject.append(raw)

	# # Concatanate all the raw data
	# raw = concatenate_raws(subject)

	# # print(raw)
	# # print(raw.annotations)
	# raw.load_data()
	# raw.plot(block=True, duration=60, n_channels=10, scalings={"eeg": 300e-6})
 
	# # Create the annotations mapping
	# mapping = {
	# 	1: 'rest',
	# 	2: 'motor/left',
	# 	3: 'motor/right',
	# }
	# default_events, _ = events_from_annotations(raw)
	# annot_from_events = annotations_from_events(
	# 	events=default_events,
	# 	event_desc=mapping,
	# 	sfreq=raw.info["sfreq"],
	# 	orig_time=raw.info["meas_date"],
	# )

	# raw.set_annotations(annot_from_events)

	# events, events_id = events_from_annotations(raw)

	# print(raw.annotations)
	# print(raw.info['sfreq'])

	# # Filter the data between 8 and 32 Hz
	# raw.plot(block=True, duration=20, n_channels=10, scalings={"eeg": 60e-6})
	# raw.filter(8,32, fir_design='firwin')

	# # Create the standard montage
	# print(raw.info['ch_names'])
	# raw.rename_channels(lambda x: x.strip('.'))
	# montage = make_standard_montage('standard_1005')
	# eegbci.standardize(raw)
	# raw.set_montage(montage)
	# raw.plot_sensors(show_names=True, block=True, sphere=(0, 0.015, 0, 0.095))
	# raw.plot(block=True, duration=20, n_channels=10, scalings={"eeg": 60e-6})
 
	tmin, tmax = -1.0, 4.0
	event_id = dict(hands=2, feet=3)
	subject = 1
	runs = [5, 9, 13]  # motor imagery: hands vs feet

	raw_fnames = eegbci.load_data(subject, runs)
	raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
	eegbci.standardize(raw)  # set channel names
	montage = make_standard_montage("standard_1005")
	raw.set_montage(montage)

	# Apply band-pass filter
	raw.filter(7.0, 30.0, fir_design="firwin", skip_by_annotation="edge")

	events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

	picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude="bads")

	epochs = Epochs(
		raw,
		events,
		event_id,
		-1.0,
		4.0,
		proj=True,
		picks=picks,
		baseline=None,
		preload=True,
	)
	epochs_train = epochs.copy().crop(tmin=1.0, tmax=2.0)
	labels = epochs.events[:, -1] - 2


	# Define a monte-carlo cross-validation generator (reduce variance):
	scores = []
	epochs_data = epochs.get_data(copy=False)
	epochs_data_train = epochs_train.get_data(copy=False)
	cv = ShuffleSplit(10, test_size=0.2, random_state=42)
	cv_split = cv.split(epochs_data_train)

	# Assemble a classifier
	lda = LinearDiscriminantAnalysis()
	csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)

	# Use scikit-learn Pipeline with cross_val_score function
	clf = Pipeline([("CSP", csp), ("LDA", lda)])
	scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=None)

	# Printing the results
	class_balance = np.mean(labels == labels[0])
	class_balance = max(class_balance, 1.0 - class_balance)
	print(
		"\n\n-----------Classification accuracy: %f / Chance level: %f\n" % (np.mean(scores), class_balance)
	)

	# plot CSP patterns estimated on full data for visualization
	csp.fit_transform(epochs_data, labels)

	csp.plot_patterns(epochs.info, ch_type="eeg", units="Patterns (AU)", size=1.5)

	sfreq = raw.info["sfreq"]
	w_length = int(sfreq * 0.5)  # running classifier: window length
	w_step = int(sfreq * 0.1)  # running classifier: window step size
	w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)

	scores_windows = []

	for train_idx, test_idx in cv_split:
		y_train, y_test = labels[train_idx], labels[test_idx]

		X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
		X_test = csp.transform(epochs_data_train[test_idx])

		# fit classifier
		lda.fit(X_train, y_train)

		# running classifier: test classifier on sliding window
		score_this_window = []
		for n in w_start:
			X_test = csp.transform(epochs_data[test_idx][:, :, n : (n + w_length)])
			score_this_window.append(lda.score(X_test, y_test))
		scores_windows.append(score_this_window)

	# Plot scores over time
	w_times = (w_start + w_length / 2.0) / sfreq + epochs.tmin

	plt.figure()
	plt.plot(w_times, np.mean(scores_windows, 0), label="Score")
	plt.axvline(0, linestyle="--", color="k", label="Onset")
	plt.axhline(0.5, linestyle="-", color="k", label="Chance")
	plt.xlabel("time (s)")
	plt.ylabel("classification accuracy")
	plt.title("Classification score over time")
	plt.legend(loc="lower right")
	plt.show()
	