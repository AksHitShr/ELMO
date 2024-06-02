## iNLP Assignment-4
### Akshit Sharma (2021101029)

#### Trained separate models for forward and backward input sentences (for ELMo).

#### Files included:

- backward_model.pt, forward_model.pt, classifier.pt (trained models for the ELMo and downstream classification task)

- classification.py (Code to train the model for downstream task, using the pre-trained ELMo model)

- ELMO.py (Code to train the ELMo forward and backward bi-LSTM models and save them for future use)

- .ipynb files for the 3 cases of hyperparameter tuning, already run

- Report.pdf (analysis of results and comparisons with previous assignment)

#### Loading pre-trained models:

##### Loading forward and backward models:
```
forward_model=torch.load('forward_model.pt')
backward_model=torch.load('backward_model.pt')
```

##### Loading the classifier model for downstream task:
```
classification_model=torch.load('classifier.pt')
```

Note: Models have been trained with limited number of epochs and tuning due to lack of compute and high training time.

Onedrive link for Models: <href>https://iiitaphyd-my.sharepoint.com/:f:/g/personal/akshit_sharma_students_iiit_ac_in/Epo86kyorRxMhZbZJCb7jVsBDx4nCiThpy5MuCwhFIChCg?e=thIusG</href>