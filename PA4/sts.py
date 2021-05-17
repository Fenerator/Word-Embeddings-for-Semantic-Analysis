import csv
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from simpletransformers.classification import ClassificationModel
from transformers import logging
import pandas as pd

# Set the logging verbosity
logging.set_verbosity_error()

# Setting a random seed in order to ensure result reproducibility
RANDOM_SEED = 64
np.random.seed(RANDOM_SEED)

if __name__ == '__main__':

    # Reading data
    gold_scores = pd.read_csv('STS.news.sr.txt', sep='\t', encoding='utf-8', header=None,
                              quoting=csv.QUOTE_NONE)[0].tolist()
    sentences1 = pd.read_csv('STS.news.sr.txt', sep='\t', encoding='utf-8', header=None,
                             quoting=csv.QUOTE_NONE)[6].tolist()
    sentences2 = pd.read_csv('STS.news.sr.txt', sep='\t', encoding='utf-8', header=None,
                             quoting=csv.QUOTE_NONE)[7].tolist()

    # Select the models - this is done by simply stating their full name from the Hugging Face model page:
    # https://huggingface.co/models
    # For each model we need to specify the model architecture - 'bert, 'electra', 'xlm', etc.
    # TODO: Select the model we are working with and store the selection in an appropriate variable
    models_dict = {'classla/bcms-bertic': 'electra'}
    
    

    # Pearson correlation coefficient is the standard performance metric for this task
    # Define a simple version of the Pearson correlation function that returns only the correlation score,
    # without the p-value
    def pearson_corr(preds, labels):
        return pearsonr(preds, labels)[0]


    # Parameter dict contains all the hyper-parameters related to the (Simple) Transformers library
    # The full list of available hyper-parameters is available here:
    # https://simpletransformers.ai/docs/usage/#configuring-a-simple-transformers-model
    parameter_dict = {}

    # Setting the random seed to ensure experiment reproducibility
    parameter_dict['manual_seed'] = RANDOM_SEED

    # We must set the regression option to True since the similarity scores are continuous numerical values
    parameter_dict['regression'] = True

    # Disables the mixed precision training mode, since it may cause calculation issues on some configurations
    parameter_dict['fp16'] = False

    # These options force the library to create/train a new model every time the code is run,
    # which enables experiment reproducibility
    parameter_dict['overwrite_output_dir'] = True
    parameter_dict['reprocess_input_data'] = True
    parameter_dict['no_cache'] = True
    parameter_dict['no_save'] = True
    parameter_dict['save_eval_checkpoints'] = False
    parameter_dict['save_model_every_epoch'] = False
    parameter_dict['use_cached_eval_features'] = False

    # TODO: Choose an appropriate working directory for the transformers models
    # The parameters that you need to be set are 'output_dir', 'cache_dir' and 'tensorboard_dir'
    # Call the subfolders of the working directory 'outputs', 'cache' and 'runs' respectively.
    parameter_dict['output_dir'] = '/home/csmt_st16/outputs_bert/outputs'
    parameter_dict['cache_dir'] = '/home/csmt_st16/outputs_bert/cache'
    parameter_dict['tensorboard_dir'] = '/home/csmt_st16/outputs_bert/runs'

    # Reduce the output details - set to False to enable a detailed overview of the training process
    parameter_dict['silent'] = True

    # The model we consider retains text casing
    parameter_dict['do_lower_case'] = False

    # Experiment with increasing the number of training epochs and see how it affects the results
    # TODO change
    parameter_dict['num_train_epochs'] = 8

    # Some other options you could explore that influence the model - the default values are given below
    # If your GPU runs out of memory, try lowering the batch size parameter
    parameter_dict['train_batch_size'] = 8
    parameter_dict['eval_batch_size'] = 8
    parameter_dict['learning_rate'] = 4e-5

    score_numerical_precision = '.4f'
    # In order to speed up the experiments, the number of CV folds is set to 5
    CV_SPLITS = 5
    cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    # Iterate over all models
    for model_name, model_type in models_dict.items():
        print('@@@@@@@@@@ ' + model_name + ' @@@@@@@@@@')

        X1 = np.array(sentences1)
        X2 = np.array(sentences2)
        y = np.array(gold_scores)
        score_per_fold = []

        for train_index, test_index in cv.split(X1, y):
            X1_train, X2_train, X1_test, X2_test = X1[train_index], X2[train_index], X1[test_index], X2[test_index]
            y_train, y_test = y[train_index], y[test_index]
            train_df = pd.DataFrame(list(zip(X1_train, X2_train, y_train)), columns=['text_a', 'text_b', 'labels'])
            eval_df = pd.DataFrame(list(zip(X1_test, X2_test, y_test)), columns=['text_a', 'text_b', 'labels'])

            # Change the use_cuda to False if you do not have GPU support
            model = ClassificationModel(model_type, model_name, num_labels=1, use_cuda=True, args=parameter_dict)

            # Train the model
            model.train_model(train_df, show_running_loss=False)

            # Evaluate the model and print the results for each CV fold
            result, model_outputs, wrong_predictions = model.eval_model(eval_df, corr=pearson_corr)
            score_per_fold.append(result['corr'])
            print(result['corr'])

        # Print the final results
        print('Final CV performance: ' + format(sum(score_per_fold) / CV_SPLITS, score_numerical_precision))
        print()
