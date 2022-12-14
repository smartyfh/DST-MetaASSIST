# AUX-DST

This model is used as both the auxiliary model and the primary model. When used as the auxiliary model, it is trained on the clean dev set and we chose the best checkpoint according to the performance on the noisy training set. When used as the primary model, it is trained on the noisy training set and the best model was chosen based on the performance on the dev set.

## Usage

Here I show how we can run this model on MultiWOZ 2.4. The same procedure can be applied to MultiWOZ 2.0 (a modified version of the original one).

### Data Preprocessing

There are two steps that preprocess the dataset to the format required by the model.

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

Here I use Scheme S1 as the example:
```console
❱❱❱ python3 train-S1.py --data_dir data/mwz2.4 --save_dir output-meta24-S1/exp --base_lr 2.5e-5 --sw_lr 1e-4 --init_weight 0.5 --n_epochs 12 --do_train
```

<font color='red'>Note: More than 30G GPU memory will be occupied if the default hyperparameters are used. In case you don't have sufficient GPU memory, please use a smaller batch size.</font>
