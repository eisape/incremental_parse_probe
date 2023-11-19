
## Probing for Incremental Parse States in Autoregressive Language Models

Supplementary materials and demo for "Probing for Incremental Parse States in Autoregressive Language Models" (Eisape et al., 2022).

## Environment

Our [dockerfile](Dockerfile) contains the necessary dependencies to run the code in this repository and can be built with the following command:

    docker build -t incremental_parse_probe .

The rest of the walkthrough assumes you are working in a suitable environment.

## Preprocessing

The necessary datasets are 1) PTB formatted constituency parses and 2) conllx formatted dependency parses (i.e. `$SPLIT.txt`, `$SPLIT.conllx`; conllx formatted tree can be generated with [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/)). After adding those files to `data/`, running `python3 src/preprocess.py` will generate preprocessed versions of the dataset in `data/`.

## Training

The following command trains a probe specified by `config.yaml` with PyTorch Lightning:

    python3 src/train.py --config $CONFIG_PATH

The result of training is a new repository in `./experiment_checkpoints` with model parameters and hyperparameters. We provide config files for each of the models in the paper in [configs/](configs). **NOTE**: the geometric action probe is pretrained on the regression task from Hewitt and Manning (2019), to train these probes first train a geometric regression probe on the relevant model and layer, then point to its weights from the config file. See [configs/](configs) for an example.

## Evaluation

To evaluate the probes with probe-based word-synchronous beam search, run the following command with the path of a model training run:

    python3 src/parse.py --experiment_path $EXPERIMENT_PATH

Where `experiment` points to the directory with the probe that was created during training. This script uses utilities from gpt2.py to decode an incremental parse state up to and including the current word from GPT2 encodings of a sentence prefix up to that word. The result is a new CSV file in `results/` with parsing statistics (e.g. UAS).

In addition to these, the paper includes several more involved experiments, including behavioural and causal intervention experiments on GPT-2 processing garden path sentences. This codebase contains all of the necessary utilities to replicate these experiments, mainly in gpt2.py; we also include the dataset used there in  ([data/npz_experiment](data/npz_experiment)).  Please contact [eisape@mit.edu]([mailto:eisape@mit.edu](https://eisape.github.io/)) with any difficulties or questions.

## Citation

    ```
    @inproceedings{eisape-etal-2022-probing,
        title = "Probing for Incremental Parse States in Autoregressive Language Models",
        author = "Eisape, Tiwalayo and Gangireddy, Vineet  and Levy, Roger and Kim, Yoon",
        booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
        address = "Abu Dhabi, United Arab Emirates",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2022.findings-emnlp.203",
        pages = "2801--2813",
        month = dec,
        year = "2022",
    }
    ```

## Acknowledgments

This project builds on code based from the following repositories:

- [https://github.com/john-hewitt/structural-probe](https://github.com/john-hewitt/structural-probe)
- [https://github.com/aistairc/rnng-pytorch](https://github.com/aistairc/rnng-pytorch)
- [https://github.com/qipeng/arc-swift](https://github.com/qipeng/arc-swift)
