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
tagger_url?=http://localhost:8000
limit?=10000000000
gpus?=1
bpe_vocab_size?=5000
exp_dir?=$(data_dir)/exp
gpus?=1
epochs?=20
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
$(corpus_dir)/data.txt: | $(corpus_dir)
	$(python_cmd) ./local/dwn.py --dataset $(hf_dataset) --text-field $(hf_text_field) --output $(corpus_dir)/data.txt \
	 	--limit $(limit) 
$(corpus_dir)/sentences.txt: $(corpus_dir)/data.txt 
	$(python_cmd) ./local/tagger.py --input $^ --output $@ --tagger_url $(tagger_url)
$(corpus_dir)/cleaned.txt: $(corpus_dir)/sentences.txt 
	$(python_cmd) ./local/drop_sentences.py --input $^ --output $@
###################################################################
start/tagger:
	docker compose up -d
###################################################################
prepare/corpus: $(corpus_dir)/cleaned.txt
.PHONY: prepare/corpus
###################################################################
###################################################################
###################################################################
$(data_dir)/texts/cleaned.txt: $(corpus_dir)/data.txt | $(data_dir)/texts
	cp $(corpus_dir)/data.txt $@
$(data_dir)/texts/train.txt: $(data_dir)/texts/cleaned.txt | $(data_dir)/texts
	cp $^ $@
$(data_dir)/lang_bpe_$(bpe_vocab_size)/bpe.model: $(data_dir)/texts/cleaned.txt | $(data_dir)/lang_bpe_$(bpe_vocab_size)
	$(python_cmd) ./local/sentence_piece.py --input $(data_dir)/texts/cleaned.txt --vocab_size $(bpe_vocab_size) --output_dir $(data_dir)/lang_bpe_$(bpe_vocab_size)
$(data_dir)/texts/train_text.txt $(data_dir)/texts/train_labels.txt: $(data_dir)/texts/train.txt
	$(python_cmd) ./local/preprocess.py --input $(data_dir)/texts/train.txt --output_text $(data_dir)/texts/train_text.txt --output_labels $(data_dir)/texts/train_labels.txt
###################################################################
prepare/data: $(data_dir)/lang_bpe_$(bpe_vocab_size)/bpe.model
.PHONY: prepare/data
###################################################################
train: $(data_dir)/lang_bpe_$(bpe_vocab_size)/bpe.model $(data_dir)/texts/train_text.txt | $(exp_dir)
	$(python_cmd) ./train.py \
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
