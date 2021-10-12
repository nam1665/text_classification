from fast_bert.prediction import BertClassificationPredictor

import pandas as pd

import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

MODEL_PATH = "../models/output/model_out"
LABEL_PATH = "./labels"
predictor = BertClassificationPredictor(
				model_path=MODEL_PATH,
				label_path=LABEL_PATH, # location for labels.csv file
				multi_label=True,
				model_type='xlnet',
				do_lower_case=False,
				device=None) # set custom torch.device, defaults to cuda if available


df = pd.read_excel('traindata_clean.xlsx')

array = []
dict = {}
for index, row in df.iterrows():

    text = str(row[0])

    single_prediction = predictor.predict(text)

    predict_array = []

    for val in single_prediction:
        predict_array.append(val[1])

    (m, i) = max((v, i) for i, v in enumerate(predict_array))

    if row[1] == 1:
        dict['text'] = row[0]
        dict['label'] = "phim_truyen"
        dict['probability'] = m
        dict['predict_label'] = single_prediction[i][0]
        if single_prediction[i][0] != dict['label']:
            dict['result'] = "false"
        else:
            dict['result'] = "true"
        array.append(dict.copy())

    if row[1] == 2:
        dict['text'] = row[0]
        dict['label'] = "thoi_su"
        dict['probability'] = m
        dict['predict_label'] = single_prediction[i][0]
        if single_prediction[i][0] != dict['label']:
            dict['result'] = "false"
        else:
            dict['result'] = "true"
        array.append(dict.copy())
    if row[1] == 3:
        dict['text'] = row[0]
        dict['label'] = "ca_nhac"
        dict['probability'] = m
        dict['predict_label'] = single_prediction[i][0]
        if single_prediction[i][0] != dict['label']:
            dict['result'] = "false"
        else:
            dict['result'] = "true"
        array.append(dict.copy())
    if row[1] == 4:
        dict['text'] = row[0]
        dict['label'] = "the_thao"
        dict['probability'] = m
        dict['predict_label'] = single_prediction[i][0]
        if single_prediction[i][0] != dict['label']:
            dict['result'] = "false"
        else:
            dict['result'] = "true"
        array.append(dict.copy())
    if row[1] == 5:
        dict['text'] = row[0]
        dict['label'] = "tong_hop"
        dict['probability'] = m
        dict['predict_label'] = single_prediction[i][0]
        if single_prediction[i][0] != dict['label']:
            dict['result'] = "false"
        else:
            dict['result'] = "true"
        array.append(dict.copy())

    print(dict)

df = pd.DataFrame(array).to_excel("./predicted_output.xlsx")
