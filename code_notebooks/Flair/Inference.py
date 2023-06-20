from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.datasets import DataLoader
import pandas as pd
import pickle
from AddPosColumn import AddPosCol
from ConvertBioToConll import df_to_conll
from flair.models import SequenceTagger

user_input = "final_test.csv"
model_input = "ConllData.txt"
def Inference():

# define columns
    columns = {0: 'text', 1: 'pos', 2: 'ner'}
    data_folder = '.'
    label_type = 'ner'
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                              test_file = model_input, train_file = model_input, dev_file = model_input)

    with open('LabelDict', 'rb') as f:
        label_dict = pickle.load(f)
    print(label_dict)

    test_dataloader = DataLoader(corpus.test, batch_size=1)

    tagger = SequenceTagger.load("/home/ayush22095/NER_IP/Flair/resources/taggers/sota-ner-flert/final-model.pt")
    # with open('./tagger', 'rb') as f:
    #     tagger = pickle.load(f)

    pred_df = pd.DataFrame(columns=["sentence", "tags"])
    for batch in test_dataloader:
        tagger.predict(batch)    
        for item in batch:            
            for item in batch:
                tag_list = []
                for i, phrase in enumerate(item.annotation_layers['ner']):
                    tag_list.append(phrase[i]._value)
                tags = ' '.join(tag_list)
                sentence = item.text
                pred_df = pred_df.append([sentence, tags], ignore_index = True)
    pred_df.to_csv("NER_Tagged_Results.csv")
                  
if __name__ == "__main__":
    df = pd.read_csv(user_input)
    df = df[['sentence']]
    df_pos = AddPosCol(df, cols=['sentence','parts_of_speech'])
    conll = df_to_conll(df_pos, list(df_pos.columns))    
    with open(f"./{model_input}", "w") as f:
        f.write(conll)
    Inference()
    
