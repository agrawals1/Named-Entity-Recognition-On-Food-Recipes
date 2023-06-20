import pandas as pd

def df_to_conll(df, cols):
    conll = ''
    for i in range(len(df)):
        for j in range(len(str(df.iloc[i]['sentence']).split())):
            conll += str(df.iloc[i]['sentence']).split()[j] + ' ' 
            conll += str(df.iloc[i]['parts_of_speech']).split()[j] + ' '
            if('tags' in cols):
                conll+= str(df.iloc[i]['tags']).split()[j]
            conll += '\n'
        conll += '\n'    
    return conll



if __name__ == "__main__":
    df = pd.read_csv('pos_bio_augmented.csv')
    conll = df_to_conll(df, list(df.columns))
    with open("./ConllBioAugmented.txt", "w") as f:
        f.write(conll)
