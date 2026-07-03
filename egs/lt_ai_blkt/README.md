# lt_ai_blkt

This setup requires a text corpus split into sentences.

Original corpus: https://huggingface.co/datasets/VSSA-SDSA/LT_AI_BLKT


## Steps
```bash
make prepare/data
make train
make test
make export
```

## Model

Trained on 1 376 195 015 words and 98 565 509 sentences.

```bash
PYTHONPATH=./../../ LOGLEVEL=INFO python3 ./../../train.py \
	--world-size 3 \
	--do_finetune False \
	--data_dir ./data/texts \
	--exp_dir ./data/exp \
	--bpe_model ./data/lang_bpe_5000/bpe.model \
	--base-lr 0.002 \
	--epochs 20 \
	--streaming-shuffle-buffer 20000 \
	--batch_size 512 
```    

### Structure

```txt
Model_new(
  (embedding): Embedding(5000, 100)
  (encoder): Encoder(
    (layers): ModuleList(
      (0-2): 3 x ConvLayer(
        (conv): Conv1d(100, 100, kernel_size=(3,), stride=(1,), padding=same)
        (norm): LayerNorm()
      )
    )
  )
  (biGRU): LSTM(100, 384, num_layers=2, batch_first=True, bidirectional=True)
  (GRU): LSTM(768, 384, batch_first=True)
  (decoder_case): Linear(in_features=768, out_features=3, bias=True)
  (decoder_punct): Linear(in_features=768, out_features=6, bias=True)
  (dropout1): Dropout(p=0.5, inplace=False)
  (dropout2): Dropout(p=0.5, inplace=False)
)
```


### Parameters

Number of parameters: 7 408 445


## Results

```bash
PYTHONPATH=./../../ LOGLEVEL=INFO python3 ./../../decode.py \
	--world-size 1 \
	--data_dir ./data/texts \
	--exp_dir ./data/exp \
	--bpe_model ./data/lang_bpe_5000/bpe.model \
	--epoch 20  \
	--batch 185000 \
	--avg 20 \
	--batch_size 512 \
	--file ./data/texts/test_features.parquet
```

### Case metrics:
```txt
----------------------------------------------------------------------------------------
LOWER: 	Prec [0.984], 	Rec [0.990], 	F1 [0.987], 
UPPER: 	Prec [0.977], 	Rec [0.972], 	F1 [0.975], 
CAP: 	Prec [0.933], 	Rec [0.897], 	F1 [0.915], 
Overall: 	Prec [0.935], 	Rec [0.900], 	F1 [0.917], 
```

### Case label counts
```txt
0 -> LOWER: predicted [9504718], expected [9448553] (86.29%), correct [9355604]
1 -> UPPER: predicted [47322], expected [47569] (0.43%), correct [46245]
2 -> CAP: predicted [1398309], expected [1454227] (13.28%), correct [1305083]
```

### Punctuation metrics:
```txt
: 	Prec [0.978], 	Rec [0.988], 	F1 [0.983], 
,: 	Prec [0.882], 	Rec [0.846], 	F1 [0.864], 
.: 	Prec [0.886], 	Rec [0.884], 	F1 [0.885], 
?: 	Prec [0.643], 	Rec [0.488], 	F1 [0.555], 
:: 	Prec [0.639], 	Rec [0.367], 	F1 [0.466], 
—: 	Prec [0.834], 	Rec [0.204], 	F1 [0.328], 
Overall: 	Prec [0.880], 	Rec [0.842], 	F1 [0.861], 
```

### Punctuation label counts
```txt
0 -> : predicted [8950631], expected [8859389] (80.91%), correct [8756020]
1 -> ,: predicted [1069132], expected [1113937] (10.17%), correct [942896]
2 -> .: predicted [894282], expected [896179] (8.18%), correct [792188]
3 -> ?: predicted [14850], expected [19564] (0.18%), correct [9545]
4 -> :: predicted [11273], expected [19605] (0.18%), correct [7199]
5 -> —: predicted [10181], expected [41675] (0.38%), correct [8496]
```


## Model Usage

```bash
./build/bin/sherpa-onnx-online-punctuation --cnn-bilstm=lt_ai_blkt.punctuation-casing.v02/model/model.int8.onnx --bpe-vocab=lt_ai_blkt.punctuation-casing.v02/model/bpe.vocab "p stankaus pasirodymai vyks kaune vilniuje ir panevėžyje lrv nusprendė kompensuoti už šildymą žmonėms tada jie atėjo namo"
```
```txt
Elapsed seconds: 0.008 s
Input text: p stankaus pasirodymai vyks kaune vilniuje ir panevėžyje lrv nusprendė kompensuoti už šildymą žmonėms tada jie atėjo namo
Output text: P. Stankaus pasirodymai vyks Kaune, Vilniuje ir Panevėžyje. LRV nusprendė kompensuoti už šildymą žmonėms. Tada jie atėjo namo
```

