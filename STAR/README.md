# DST-STAR

The [STAR](https://arxiv.org/abs/2101.09374) model is another primary model we considered. It is an ontology-based model and has achieved the SOTA performance on MultiWOZ 2.4.

## Usage

### Data Preprocessing

There are two steps that preprocess the dataset to the format required by the model.

```console
❱❱❱ python3 create_data.py --mwz_ver 2.0
❱❱❱ python3 preprocess_data.py --data_dir data/mwz2.0
```

or

```console
❱❱❱ python3 create_data.py --mwz_ver 2.4
❱❱❱ python3 preprocess_data.py --data_dir data/mwz2.4
```

### Pseudo Labels

The pseudo labels can be downloaded [here](https://drive.google.com/file/d/1xrzhbEIou7h-qS1yRd83vKVnR6ZGmotp/view?usp=sharing). After downloading,
```console
❱❱❱ unzip Pseudo\ Labels.zip
```

### Training

We train the primary model using both the original noisy labels and the generated pseudo labels. We have proposed three schemes to automatically learn weighting parameters that can balance the pseudo labels and vanilla labels.

Here I list the training scripts on MultiWOZ 2.4 as examples:
```console
❱❱❱ python3 train-S1.py --data_dir data/mwz2.4 --save_dir output-meta24-S1/exp --train_batch_size 16 --meta_batch_size 8 --enc_lr 4e-5 --dec_lr 1e-4 --sw_lr 5e-5 --init_weight 0.5 --n_epochs 15 --do_train
```

or 

```console
❱❱❱ python3 train-S2.py --data_dir data/mwz2.4 --save_dir output-meta24-S2/exp --train_batch_size 16 --meta_batch_size 8 --enc_lr 4e-5 --dec_lr 1e-4 --wnet_lr 1e-5 --n_epochs 15 --do_train
```

or 

```console
❱❱❱ python3 train-S3.py --data_dir data/mwz2.4 --save_dir output-meta24-S3/exp --train_batch_size 16 --meta_batch_size 8 --enc_lr 4e-5 --dec_lr 1e-4 --wnet_lr 3e-5 --n_epochs 12 --do_train
```

<font color='red'>Note: More than 40G GPU memory will be occupied if the default hyperparameters are used. In case you don't have sufficient GPU memory, please use a smaller batch size.</font>


## Links

* The model checkpoints on MultiWOZ 2.4 can be downloaded here: [S1](https://drive.google.com/file/d/1ZjZrrfV8lJWccEU0qFVmWC71tEHQvOTF/view?usp=sharing), [S2](https://drive.google.com/file/d/1EjYofdtmHZ38RXh2I17FsK5uIZ7X6pmG/view?usp=sharing), and [S3](https://drive.google.com/file/d/1pGen4emjU8CGBDir7A4HshuMvkpMLvvZ/view?usp=sharing).