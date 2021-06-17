# text_classification# Text Classification using Fastbert

Fastbert github [repo](https://github.com/utterworks/fast-bert) 
## Installation
Install anaconda: [link]( https://conda.io/projects/conda/en/latest/index.html
) 

Create conda enviroment first:
```bash
conda create -n text_classification python=3.8
```

Then use the package manager [pip](https://pip.pypa.io/en/stable/) to install all the requirement libs.
```bash
pip install -r requirements.txt
```

## Running

To run api on background:

```bash
screen -L python api.py
```

## Usage
You can access the api 
swagger here: http://13.212.26.55:8008/api/

## Training
 - First you need to convert your data into xlsx format like: [traindata.xlsx](/traindata.xlsx)
- Then config the file path in [train.py](/train.py)
- You need to config:
```python
label_cols = ["phim_truyen", "thoi_su", "ca_nhac", "the_thao", "tong_hop"]
```
```
databunch = BertDataBunch(args['data_dir'], LABEL_PATH, args.model_name, train_file='train.csv', val_file='train.csv',
                          test_data='train.csv',
                          text_col="text", label_col=label_cols,
                          batch_size_per_gpu=args['train_batch_size'], max_seq_length=args['max_seq_length'],
                          multi_gpu=args.multi_gpu, multi_label=True, model_type=args.model_type)
```
- Then simple run
```
python train.py
```

## Text Clustering 

- You can find the example on [clustering_nlp.py](/clustering_nlp.py)


