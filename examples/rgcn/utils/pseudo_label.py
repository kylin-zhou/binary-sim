import json
import pandas as pd

item_id, score = [], []
with open(
    "results/2022-08-24-07:34:26/pyg_pred_0.05477_0.96210.json",
    "r",
) as f:
    for line in f.readlines():
        pred = json.loads(line)
        item_id.append(pred["item_id"])
        score.append(pred["score"])

session2 = pd.DataFrame({"item_id": item_id, "score": score})
session2.head()

session2 = session2[(session2["score"] > 0.6) | (session2["score"] < 0.4)]
session2["pseudo_label"] = session2["score"].apply(lambda x: 1 if x > 0.5 else 0)
session2_label = session2[session2["pseudo_label"].apply(lambda x: x in [0, 1])]
session2_label.drop("score", axis=1).to_csv(
    "data/icdm2022_session2_pseudo_labels2.csv",
    header=False,
    index=False,
)
