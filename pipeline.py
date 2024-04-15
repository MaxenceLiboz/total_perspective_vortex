#! .venv/bin/python3

import mne

if __name__ == "__main__":
     
	files_name = mne.datasets.eegbci.load_data(1, [6, 10, 14], "./datasets")

	subject = []
	for file in files_name:
		subject.append(mne.io.read_raw_edf(file))
		
	raw = mne.io.concatenate_raws(subject)

	print(raw)
	print(raw.info)
	print(raw.info["ch_names"])
	print(raw.annotations)

	raw.plot(block=True)