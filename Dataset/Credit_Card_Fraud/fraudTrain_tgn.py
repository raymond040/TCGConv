
import pandas as pd
from datetime import date

def create_id_dict_catagorical(column):
        # Creates a dictionary that creates a dictionary that using old id and creates a new id
        # Starts the ID at zero
        Old_ID = column.unique()
        New_ID = range(0,len(Old_ID))
        return dict(zip(Old_ID, New_ID))

def create_id_vector_catagorical(column):
    # Applys the dictionary on all the Old ID to create the new ID
    oldID_To_NewID = create_id_dict_catagorical(column)
    return list(map(lambda oldID : oldID_To_NewID[oldID], column))


df = pd.read_csv("/workspaces/TCGConv/Dataset/Credit_Card_Fraud/fraudTrain.csv")
df = df.sort_values(by = "unix_time")

column_list =["cc_num", # Customer ID (nodeID)
                    "merchant", # Merchant name (merchant ID)
                    "category", # Transaction category (Edge Feature)
                    "amt", # Amount of transaction (Edge Feature)
                    "gender", # Gender (Node feature)
                    "dob", # Date of birth (Node Feature)
                    "trans_num", # Edge ID
                    "unix_time",# Edge timestamp
                    "is_fraud",# Edge Label
                    "merch_lat", # Merchant Feature
                    "merch_long"] # Merchant Feature

df_lc = df[column_list].copy(deep=True) 
#   Customer ID
df_lc["cc_num"] = create_id_vector_catagorical(df_lc["cc_num"])
#   Merchant ID
df_lc["merchant"] = create_id_vector_catagorical(df_lc["merchant"])
#   Gender # Male is 1 Female is Zero
df_lc["gender"] = (df_lc["gender"] == "M")* 1 # times one to convert bool to int
#   Turning Date of Birth into Unix Time
#   Could try to make an age variable but then the node would have time changing features
df_lc["dob"] = list(map(lambda x: int(date.fromisoformat(x).strftime("%s")), df_lc["dob"]))
#   Edge ID
df_lc["trans_num"] = create_id_vector_catagorical(df_lc["trans_num"])

#   Set up time
df_lc["timestamp"] = df_lc["unix_time"] - min(df_lc["unix_time"])

#   Category -> Numeric
df_lc["category"] = pd.factorize(df_lc['category'])[0]


#   Final DF
cols_tgn = ['cc_num','merchant','unix_time','is_fraud','category','amt']
cols_tgn_alt = ['cc_num','merchant','timestamp','is_fraud','category','amt']
df_tgn = df_lc[cols_tgn]
df_tgn_alt = df_lc[cols_tgn_alt]

# to CSV
df_tgn.to_csv("fraudTrainTGN.csv", index = False)
df_tgn_alt.to_csv("fraudTrainTGN_alt.csv", index = False)