import pandas as pd
import spacy

# cols may be [sentence, tags] or [sentence,]
# df must contain a column named "sentence" which must have the actual ingridient phrases
def AddPosCol(df, cols):
    sp = spacy.load('en_core_web_sm')
    pos_col = []
    df_dict = df.to_dict('records')
    for row in df_dict:
        txt = row['sentence']
        sen = sp(txt)
        pos_str = ' '.join([str(word.pos_) for word in sen])
        pos_col.append(pos_str)

    df.insert(1, 'parts_of_speech', pos_col)
    df = df.loc[:, cols]
    df.to_csv('pos_bio_augmented.csv')
    return df

if __name__ == "__main__":
    sp = spacy.load('en_core_web_sm')
    df = pd.read_csv('bio_augmented.csv')
    df = df.iloc[:, 1:]
    pos_df = AddPosCol(df, [])
