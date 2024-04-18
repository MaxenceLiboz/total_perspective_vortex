
import click
import numpy as np
from pipeline import fetch_data, preprocess_data, create_epochs, cross_validation
from mne.decoding import CSP
from mne import set_log_level
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

from run_enum import Run

@click.group()
def main_commands():
    pass

@main_commands.command()
@click.option('-s', '--subject', help='Subject number between 1-109', type=int)
@click.option('-r', '--run', help='Run number Run 1:[3, 7, 11], run 2: [4, 8, 12], run 3: [5, 9, 13] ou run 4: [6, 10, 14]', type=int)
@click.option('-as', '--all_subject', is_flag=True, help='Run all subjects', type=bool)
@click.option('-ar', '--all_runs', is_flag=True, help='Run all runs', type=bool)
def train(subject, run, all_subject, all_runs):
    if not all_subject and subject is None: subject = click.prompt("Enter the subject number between 1 and 109", type=int)
    if not all_runs and run is None: run = click.prompt("Enter the run number Run 1:[3, 7, 11], run 2: [4, 8, 12], run 3: [5, 9, 13] ou run 4: [6, 10, 14]", type=int)
    
    runs = Run.get_all() if all_runs else [Run.get_by_index(run)]
    subjects = [i for i in range(1, 110)] if all_subject else [subject]

    pipeline = make_pipeline(CSP(n_components=8), LDA())

    accuracies_test = []
    for subject in subjects:
        for index, run in enumerate(runs):
            print(f"Subject: {subject}, Run: {run}")
            raws = preprocess_data(fetch_data(subject, run))
            
            # Get the epochs and labels for each run
            epochs, labels = create_epochs(raws, dict(T1=2, T2=3))

            X_train, X_test, y_train, y_test = train_test_split(epochs.get_data(), labels, test_size=0.2, random_state=42)
            
            # Cross validation
            score = cross_validation(X_train, y_train, pipeline)
            accuracy_train = pipeline.score(X_train, y_train)
            accuracy_test = pipeline.score(X_test, y_test)
            print(f'Cross validation scores for subject {subject} and run {run}: {np.mean(score)}')
            
            print(f'Pipeline accuracy train set: {accuracy_train}')
            print(f'Pipeline accuracy test set: {accuracy_test}')
            try:
                accuracies_test[index].append(accuracy_test)
            except:
                accuracies_test.append([accuracy_test])

        print(f"Current accuracy: {np.mean(accuracy_test)}")
        

    
    for index, run in enumerate(runs):
        print(f"Final accuracy for subjects: {subjects} and run: {run}: {np.mean(accuracies_test[index])}")

    print(f"Final accuracy for all runs of subjects {subjects}:  {np.mean(accuracies_test)}")


if __name__ == "__main__":
    try:
        set_log_level("ERROR")
        main_commands()
    except Exception as e:
        print(f'Error: {e}')