import pandas as pd

csv = "s2_run.csv"
df = pd.read_csv(csv)

acc =  df[["a_x", "a_y", "a_z"]].to_numpy(dtype=float)  
        
print(acc.shape)
print(acc)

