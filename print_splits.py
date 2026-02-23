import pandas as pd

train = pd.read_csv("/home/sasha/LPOSS/datasets/february_march_2026_data/splits/protocol_A/manifests/train.csv")
val   = pd.read_csv("/home/sasha/LPOSS/datasets/february_march_2026_data/splits/protocol_A/manifests/val.csv")

print("source_id overlap:", len(set(train.source_id) & set(val.source_id)))
print("facade_id overlap:", len(set(train.facade_id) & set(val.facade_id)))

print("\nTrain years:\n", train.year.value_counts().sort_index())
print("\nVal years:\n", val.year.value_counts().sort_index())