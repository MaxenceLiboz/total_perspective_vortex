
import click
import numpy as np
from pipeline import fetch_data, preprocess_data, create_epochs, cross_validation
from mne.decoding import CSP
from mne import set_log_level
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from run_enum import Run

@click.group()
def main_commands():
    pass

# @main_commands.command()
# @click.option('-s', '--subject', help='Subject number between 1-109', type=int)
# @click.option('-r', '--run', help='Run number Run 1:[3, 7, 11], run 2: [4, 8, 12], run 3: [5, 9, 13] ou run 4: [6, 10, 14]', type=int)
# @click.option('-as', '--all_subject', is_flag=True, help='Run all subjects', type=bool)
# @click.option('-ar', '--all_runs', is_flag=True, help='Run all runs', type=bool)
def train(subject, run, all_subject, all_runs, pipeline):
    if not all_subject and subject is None: subject = click.prompt("Enter the subject number between 1 and 109", type=int)
    if not all_runs and run is None: run = click.prompt("Enter the run number Run 1:[3, 7, 11], run 2: [4, 8, 12], run 3: [5, 9, 13] ou run 4: [6, 10, 14]", type=int)
    
    runs = Run.get_all() if all_runs else [Run.get_by_index(run)]
    subjects = [i for i in range(1, 110)] if all_subject else [subject]

    # pipeline = make_pipeline(CSP(), StandardScaler(), LogisticRegression())

    accuracies_test = []
    for subject in subjects:
        for index, run in enumerate(runs):
            raws = preprocess_data(fetch_data(subject, run))
            
            # Get the epochs and labels for each run
            epochs, labels = create_epochs(raws)

            X_train, X_test, y_train, y_test = train_test_split(epochs.get_data(), labels, test_size=0.2, random_state=42)
            
            # Cross validation
            # score = cross_validation(X_train, y_train, pipeline)
            pipeline.fit(X_train, y_train)
            accuracy_test = pipeline.score(X_test, y_test)
            # print(f'Cross validation scores for subject {subject} and run {run}: {np.mean(score)}')
            
            print(f'Subject: {subject} Task: {run} Accuracy: {accuracy_test}')
            try:
                accuracies_test[index].append(accuracy_test)
            except:
                accuracies_test.append([accuracy_test])

        print(f"Subject: {subject} all taks done, Current accuracy: {np.mean(accuracies_test)}")
        

    
    for index, run in enumerate(runs):
        print(f"Final accuracy for all subjects of the run: {run}: {np.mean(accuracies_test[index])}")

    print(f"Final accuracy for all runs of subject range [{subjects[0]}-{subjects[-1]}]:  {np.mean(accuracies_test)}")


if __name__ == "__main__":
    set_log_level("ERROR")
    train(None, None, True, True, make_pipeline(CSP(n_components=8), LDA()))
    train(None, None, True, True, make_pipeline(CSP(), StandardScaler(), LogisticRegression()))
    train(None, None, True, True, make_pipeline(CSP(n_components=8), LDA(shrinkage="auto", solver="lsqr")))
    # try:
    #     main_commands()
    # except Exception as e:
    #     print(f'Error: {e}')