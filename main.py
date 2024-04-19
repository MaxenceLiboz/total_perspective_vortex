
import click
import joblib
import numpy as np
from pipeline import fetch_data, preprocess_data, create_epochs
from mne.decoding import CSP
from mne import set_log_level
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit, cross_val_score, KFold

from run_enum import Run

@click.group(invoke_without_command=True)
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose mode', type=bool)
@click.pass_context
def main_commands(ctx, verbose):
    if verbose: set_log_level("INFO")
    else: set_log_level("ERROR")
    if ctx.invoked_subcommand is None:
        accuracy()
    else:
        ctx.invoked_subcommand

@main_commands.command()
@click.option('-s', '--subject', help='Subject number between 1-109', type=int)
@click.option(
    '-r', 
    '--run', 
    help=f"Run number {Run.__str__()}",
    type=int
)
@click.option('-as', '--all_subject', is_flag=True, help='Run all subjects', type=bool)
@click.option('-ar', '--all_runs', is_flag=True, help='Run all runs', type=bool)
@click.option('-p', '--plot', is_flag=True, help='Plot the data', type=bool)
@click.option('-nc', '--no_cross_validation', is_flag=True, help='Disable cross validation', type=bool)
def train(subject, run, all_subject, all_runs, plot, no_cross_validation):
    if not all_subject and subject is None: subject = click.prompt("Enter the subject number between 1 and 109", type=int)
    if not all_runs and run is None: run = click.prompt(f"Enter the run number {Run.__str__()}", type=int)
    
    runs = Run.get_all() if all_runs else [Run.get_by_index(run)]
    subjects = [i for i in range(1, 110)] if all_subject else [subject]

    pipeline = make_pipeline(CSP(n_components=8), LDA())

    # accuracies_test = []
    for subject in subjects:
        for run in runs:
            print(f'Starting training for subject {subject} and run {run.name}')
            raws = preprocess_data(fetch_data(subject, run.value), plot)
            
            # Get the epochs and labels for each run
            epochs, labels = create_epochs(raws)

            X_train, _, y_train, _ = train_test_split(epochs.get_data(), labels, test_size=0.2, random_state=42)
            
            # Cross validation
            if not no_cross_validation:
                print(f'Starting cross validation')
                scores = cross_val_score(pipeline, X_train, y_train, cv=KFold(10))
                print(scores)
                print(f'Cross validation scores: {np.mean(scores)}')

            print(f'Fitting the pipeline')
            pipeline.fit(X_train, y_train)
            

            with open(f'./models/subject_{subject}_run_{run.name}.joblib', 'wb') as f:
                joblib.dump(pipeline, f, compress=True)
            
            print(f'Training complete and model saved\n')


@main_commands.command()
@click.option('-s', '--subject', help='Subject number between 1-109', type=int)
@click.option(
    '-r', 
    '--run', 
    help=f"Run number {Run.__str__()}",
    type=int
)
def predict(subject, run):
    if subject is None: subject = click.prompt("Enter the subject number between 1 and 109", type=int)
    if run is None: run = click.prompt(f"Enter the run number {Run.__str__()}", type=int)

    run = Run.get_by_index(run)

    with open(f'./models/subject_{subject}_run_{run.name}.joblib', 'rb') as f:
        pipeline = joblib.load(f)
    
    if pipeline is None:
        raise Exception(f'Model not found for subject {subject} and run {run.name}')
    
    raws = preprocess_data(fetch_data(subject, run.value))
    epochs, labels = create_epochs(raws)
    _, X_test, _, y_test = train_test_split(epochs.get_data(), labels, test_size=0.2, random_state=42)

    print(f'Predictions for subject {subject} and run {run.name}')
    prediction = pipeline.predict(X_test)
    print('Epoch nb: [prediction] [truth] equal?')
    for index, prediction in enumerate(prediction):
        print(f'epoch {index}: \t[{prediction}] \t[{y_test[index]}] \t{prediction == y_test[index]}')
    print(f'Accuracy: {np.mean(prediction == y_test)}')


def accuracy():
    accuracies = []
    for subject in range(1, 110):
        for run in Run.get_all():
            with open(f'./models/subject_{subject}_run_{run.name}.joblib', 'rb') as f:
                pipeline = joblib.load(f)
            
            if pipeline is None:
                raise Exception(f'Model not found for subject {subject} and run {run.name}')
            
            raws = preprocess_data(fetch_data(subject, run.value))
            epochs, labels = create_epochs(raws)
            _, X_test, _, y_test = train_test_split(epochs.get_data(), labels, test_size=0.2, random_state=42)

            accuracy = pipeline.score(X_test, y_test)
            accuracies.append(accuracy)
            print(f'Experiment: {run.name} subject {subject} accuracy: {accuracy}')

        print(f'Current accuracy subject [{1}-{subject}]: {np.mean(accuracies)}')
    
    print(f'Final accuracy: {np.mean(accuracies)}')

if __name__ == "__main__":
    try:
        main_commands()
    except Exception as e:
        print(f'Error: {e}')