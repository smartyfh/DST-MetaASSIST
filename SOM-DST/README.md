# SOM-DST

The SOM-DST model is an open vocabulary-based approach. We took SOM-DST as one primary model to test if the generated pseudo labels can also help train models that generate the states directly.

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

Here I use Scheme S1 on MultiWOZ 2.4 as the example:
```console
❱❱❱ python3 train-S1.py --data_root data/mwz2.4 --save_dir output-meta24-S1/exp --batch_size 16 --meta_batch_size 8 --enc_lr 4e-5 --dec_lr 1e-4 --wnet_lr 4e-5 --init_weight 0.5 --n_epochs 30 --do_train
```

**Note**: More than 40G GPU memory will be occupied if the default hyperparameters are used. In case you don't have sufficient GPU memory, please use a smaller batch size.

