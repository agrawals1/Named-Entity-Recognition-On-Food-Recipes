from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import os
import json

# define columns
columns = {0: 'text', 1: 'pos', 2: 'ner'}
data_folder = 'NER_IP/Flair/data'
label_type = 'ner'
print("----------------------------------------------")
print(os.getcwd())
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='BioConllTest.txt',
                              test_file='BioConllTest.txt',
                              dev_file='BioConllTest.txt')

print(len(corpus.train))
print(corpus.train[0].to_tagged_string('ner'))
print(corpus.train[1].to_tagged_string('pos'))

label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=False)
print(label_dict)

embeddings = TransformerWordEmbeddings(model='xlm-roberta-large',
                                       layers="-1",
                                       subtoken_pooling="first",
                                       fine_tune=True,
                                       use_context=True,
                                       )

tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type='ner',
                        use_crf=False,
                        use_rnn=False,
                        reproject_embeddings=False,
                        )

trainer = ModelTrainer(tagger, corpus)

results_dict = trainer.fine_tune('./NER_IP/Flair/resources/taggers/sota-ner-flert',
                  learning_rate=5.0e-6,
                  mini_batch_size=20)        #mini_batch_chunk_size=1, remove this parameter to speed up computation if you have a big GPU
                  
with open('FinalResults.json', 'w') as fp:
    json.dump(results_dict, fp)
