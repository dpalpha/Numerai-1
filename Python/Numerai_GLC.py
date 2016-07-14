import graphlab
import pandas as pd

#load data
test_data = graphlab.SFrame.read_csv('./numerai_tournament_data.csv')
train_data = graphlab.SFrame.read_csv('./numerai_training_data.csv')

#train
model = graphlab.classifier.logistic_classifier.create(train_data, target='target')

#make predictions
prediction = model.predict(test_data, output_type='probability')

#make submission
def make_submission(prediction, filename):
    with open(filename, 'w') as f:
        f.write('t_id,probability\n')
        submission_strings = test_data['t_id'].astype(str) + ',' + prediction.astype(str)
        for row in submission_strings:
            f.write(row + '\n')

make_submission(prediction, 'submission_Numerai_GLC.csv')