###################################################################
cfg?=Makefile.options
-include $(cfg)
###################################################################
log?=INFO
python_cmd=PYTHONPATH=./../../ LOGLEVEL=$(log) python3

corpus_dir?=corpus
data_dir?=data
hf_dataset?=VSSA-SDSA/LT_AI_BLKT
hf_text_field?=text
tagger_url_cfg?=--tagger-url http://localhost:8001/tag --tagger-url http://localhost:8002/tag
limit?=10000000000
gpus?=1
bpe_vocab_size?=5000
exp_dir?=$(data_dir)/exp
gpus?=1
epochs?=20
# for test, we can specify epoch to use, by default it will be the same as epochs for training
epoch?=$(epochs) 
batch_size?=256
###################################################################
info:
	@echo "data_dir: $(data_dir)"
	@echo "exp_dir: $(exp_dir)"
	@echo "hf_dataset: $(hf_dataset)"
	@echo "hf_text_field: $(hf_text_field)"
	@echo "limit: $(limit)"
	@echo "gpus: $(gpus)"
	@echo "bpe_vocab_size: $(bpe_vocab_size)"
	@echo "epochs: $(epochs)"
	@echo "batch_size: $(batch_size)"
.PHONY: info
###################################################################
###################################################################
## Prepare corpus: download, split into sentences, fix casing, drop having too long sentences or unsupported symbols
###################################################################
$(data_dir) $(corpus_dir) $(data_dir)/texts $(data_dir)/lang_bpe_$(bpe_vocab_size) \
	$(exp_dir):
	mkdir -p $@
# $(corpus_dir)/.dwn: | $(corpus_dir)
# 	$(python_cmd) ./local/dwn.py --dataset $(hf_dataset) --text-field $(hf_text_field) --output $(corpus_dir)/data \
# 	 	--limit $(limit) 
# 	touch $@
$(corpus_dir)/.sentences:
	$(python_cmd) ./local/tagger.py --input $(hf_dataset) --output $(corpus_dir)/sentences $(tagger_url_cfg) \
		--checkpoint-every 100 --state-file $(corpus_dir)/.tagger_state.json --limit $(limit) 
	touch $@
$(corpus_dir)/.sentences_c: $(corpus_dir)/.sentences
	$(python_cmd) ./local/fix_case.py --input $(corpus_dir)/sentences --output $(corpus_dir)/sentences_c
	touch $@
$(corpus_dir)/.cleaned: $(corpus_dir)/.sentences_c
	$(python_cmd) ./local/clean.py --input $(corpus_dir)/sentences_c --output $(corpus_dir)/cleaned
	touch $@
###################################################################
start/tagger:
	docker compose up -d
###################################################################
prepare/corpus: $(corpus_dir)/.cleaned
.PHONY: prepare/corpus
###################################################################
###################################################################
###################################################################
$(data_dir)/texts/train.txt $(data_dir)/texts/dev.txt $(data_dir)/texts/test.txt: $(corpus_dir)/.cleaned | $(data_dir)/texts
	$(python_cmd) ./local/split.py --input $(corpus_dir)/cleaned --output-dir $(data_dir)/texts
$(data_dir)/texts/lower.txt: $(data_dir)/texts/train.txt 
	$(python_cmd) ./local/lower.py --input $^ --output $@_
	mv $@_ $@
$(data_dir)/lang_bpe_$(bpe_vocab_size)/bpe.model: $(data_dir)/texts/lower.txt | $(data_dir)/lang_bpe_$(bpe_vocab_size)
	$(python_cmd) ./local/sentence_piece.py --input $(data_dir)/texts/lower.txt --vocab_size $(bpe_vocab_size) --output_dir $(data_dir)/lang_bpe_$(bpe_vocab_size)
$(data_dir)/texts/%_features.txt: $(data_dir)/texts/%.txt $(data_dir)/lang_bpe_$(bpe_vocab_size)/bpe.model
	$(python_cmd) ./local/make_features.py --input $(data_dir)/texts/$*.txt --output $@_ --bpe_model $(data_dir)/lang_bpe_$(bpe_vocab_size)/bpe.model	
	mv $@_ $@
###################################################################
prepare/data: $(data_dir)/lang_bpe_$(bpe_vocab_size)/bpe.model \
	$(data_dir)/texts/train_features.txt $(data_dir)/texts/dev_features.txt $(data_dir)/texts/test_features.txt
.PHONY: prepare/data
###################################################################
train: $(data_dir)/lang_bpe_$(bpe_vocab_size)/bpe.model $(data_dir)/texts/train_features.txt $(data_dir)/texts/dev_features.txt | $(exp_dir)
	$(python_cmd) ./../../train.py \
		--world-size $(gpus) \
		--do_finetune False \
		--data_dir $(data_dir)/texts \
		--exp_dir $(exp_dir) \
		--bpe_model $(data_dir)/lang_bpe_$(bpe_vocab_size)/bpe.model \
		--base-lr 0.002 \
		--epochs $(epochs) \
		--batch_size $(batch_size) 
.PHONY: train
###################################################################
###################################################################
test: $(data_dir)/lang_bpe_$(bpe_vocab_size)/bpe.model $(data_dir)/texts/test_features.txt | $(exp_dir)
	$(python_cmd) ./../../decode.py \
		--world-size $(gpus) \
		--data_dir $(data_dir)/texts \
		--exp_dir $(exp_dir) \
		--bpe_model $(data_dir)/lang_bpe_$(bpe_vocab_size)/bpe.model \
		--epoch $(epoch) \
		--batch_size $(batch_size) 
.PHONY: test
_info/%: $(data_dir)/texts/%.txt
	$(python_cmd) ./local/data_info.py --input $^
info/test: _info/test
.PHONY: info/test	
###################################################################
###################################################################
export: 
	$(python_cmd) ./../../export-onnx.py \
		--exp_dir $(exp_dir) \
		--epoch $(epoch)
.PHONY: export
###################################################################
## just clean data after download
clean/data:
	@echo -n "Are you sure you want to delete $(data_dir)? [y/N] " && read ans && [ $${ans:-N} = y ]
	rm -rf $(data_dir)
.PHONY: clean/data
clean/corpus:
	@echo -n "Are you sure you want to delete $(corpus_dir)? [y/N] " && read ans && [ $${ans:-N} = y ]
	rm -rf $(corpus_dir)
.PHONY: clean/corpus
###################################################################
.EXPORT_ALL_VARIABLES:
