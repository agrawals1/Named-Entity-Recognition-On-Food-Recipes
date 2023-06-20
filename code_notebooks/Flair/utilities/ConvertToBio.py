import pandas as pd
import os
def Convert2Bio(df):    
    df_dict = df.to_dict('records')
    for index, row in enumerate(df_dict):
        txt = row["tags"]
        txt = txt.split(" ")
        j=0
        while(j<len(txt)):
            if txt[j] == 'O':
                j+=1
                continue
            else: 
                B_idx = j
                I_idx = []
                while(j<len(txt)-1 and (txt[j+1]) == txt[j]):                
                    I_idx.append(j+1)
                    j+=1            
                for idx in I_idx:
                    txt[idx] = 'I-' + txt[idx]
                txt[B_idx] = 'B-' + txt[B_idx]
                j+=1
        txt = ' '.join(txt)
        df.loc[index,"tags"] = txt
    df.to_csv("bio_augmented.csv")
    return df

if __name__ == "__main__":
    df = pd.read_csv('final_train_augmented.csv')
    df_bio = Convert2Bio(df)
    
