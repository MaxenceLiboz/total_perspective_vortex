#! .venv/bin/python3

import matplotlib.pyplot as plt

from mne import Epochs, events_from_annotations, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf, Raw

# from my_csp import CSP


def create_epochs(raw, tmin = 1.0, tmax = 4.0):
	# Get the events
	events, event_id = events_from_annotations(raw, event_id=dict(T1=2, T2=3))

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


def preprocess_data(raws: Raw, plotting=False):
	if plotting: raws.plot(duration=30, proj=False, n_channels=5, remove_dc=False, scalings=dict(eeg=100e-6))
	
	eegbci.standardize(raws)

	# Create a standard montage
	montage = make_standard_montage("standard_1020")
	raws.set_montage(montage)
	if plotting: raws.plot_sensors(show_names=True, sphere=(0, 0.015, 0, 0.095))

	# Apply a filter to go from 8 to 32 Hz
	raws.filter(8, 32, fir_design="firwin", skip_by_annotation="edge")
	if plotting: 
		raws.plot(duration=30, proj=False, n_channels=5, remove_dc=False, scalings=dict(eeg=100e-6))
		plt.show()
	return raws


def fetch_data(subject, runs):
	raw_fnames = eegbci.load_data(subject, runs, './datasets')
	return concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])