Replace the following files with the ones provided:
==================================================
model.py --> .local/lib/python3.7/site-packages/flair/nn/model.py

sequence_tagger_model.py --> .local/lib/python3.7/site-packages/flair/models/sequence_tagger_model.py

trainer.py --> .local/lib/python3.7/site-packages/flair/trainers/trainer.py

train_flair.py --> Flair/TrainFlair.ipynb


Flair Data Preperation for training :
====================================
flair requires data in conll format, which has 3 rows: text, parts_of_speech, and label. Every entry in the text column is a word . An empty line denotes end of sentence or ingridient phrase. 

Example:
--------
"Format A: not in conll format":	        												
--------------------------------

sentence,tags

"2 tablespoons vegetable oil , divided",QUANTITY UNIT NAME NAME O STATE

2 tablespoons dried marjoram,QUANTITY UNIT DF NAME

"1 large red onion , 1/4-inch slices pulled into rings",QUANTITY SIZE NAME NAME O O O O O O

"2 jalapeno peppers , seeded and minced",QUANTITY NAME NAME O STATE O STATE												

													
"Format B: conll format":
-------------------------

tablespoons NOUN B-UNIT

vegetable NOUN B-NAME

oil NOUN I-NAME

, PUNCT O

divided VERB B-STATE



2 NUM B-QUANTITY

tablespoons NOUN B-UNIT

dried VERB B-DF

marjoram NOUN B-NAME

Steps to convert from format A to format B use scripts in the Utility folder:
----------------------------------------------------------------------------
1. Use ConvertToBio.py to encode your data with BIO encoding
2. Add parts of speech using AddPosColumn.py
3. Finally convert to Conll format using ConvertBioToConll.py

Training:
========

1. Use train_flair.py to train your own NER model
2. Put your data in the data folder. Open train_flair.py and change the train test and dev file names accordingly
3. After training, the model and metrics can be found inside the resources folder. You will also find a ClassificationReports pickle file which you can unpickle to get epoch wise classification report for every class. 

Inference:
=========

1. Provide your input in Format A to the global variable "user_input" (without the tags column)
2. You will recieve the output in the file "NER_Tagged_Results.csv"

Extras:
======

1. Cluster&Sample.py allows you to work on a reduced dataset without loosing on much information
2. VisualizeClusters.py provides you various visualizations of the clusters
3. Plots&Visualize.ipynb generates various visualizations for various train and test metrics
4. The logs folder houses Tensorboard logs





