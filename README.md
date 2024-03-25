# Welcome to KSAA- CAD: Contemporary Arabic Reverse Dictionary and Word Sense Disambiguation at ArabicNLP 2024!

This is the repository for the KSAA- CAD: Contemporary Arabic Reverse Dictionary and Word Sense Disambiguation at ArabicNLP 2024!
This repository currently contains the baseline programs,  a scorer, and a format-checker to help participants get started.

Participants may be interested in the script revdict_entrypoint.py. It contains a number of useful features, such as scoring submissions, a format checker, and a few simple baseline architectures. It is also an exact copy of what is used on the Codalab platform.


# Introduction
The KSAA-CAD shared task highlight the Contemporary Arabic Language Dictionary within the scenario of developing a Reverse Dictionary (RD) system and enhancing Word Sense Disambiguation (WSD) capabilities. The first KSAA-RD (Al-Matham et al., 2023) highlighted significant gaps in the domain of reverse dictionaries, which are designed to retrieve words by their meanings or definitions. This shared task comprises two tasks: RD and WSD. The RD task focuses on identifying word embeddings that most accurately match a given definition, termed a "gloss," in Arabic. Conversely, the WSD task involves determining the specific meaning of a word in context, particularly when the word has multiple meanings. KSAA-CAD presents novel directions for researchers to investigate and offer significant contributions to the discipline. Detailed description provided in this [link](https://arai.ksaa.gov.sa/sharedTask2024/)

# Tasks
## Task 1: Reverse Dictionary 
RDs, identified by their sequence-to-vector format, characterized as sequence-to-vector, introduce a differentiated strategy in contrast to traditional dictionary lookup methods. The RD task concentrates on the conversion of human-readable glosses into word embedding vectors.
This process entails reconstructing the word embedding vector corresponding to the defined word, a methodology aligning with the approaches of (Mickus et al., 2022; Zanzotto et al., 2010; Hill et al., 2016). 
The dataset includes, lemma, lemma vector representations, and their respective gloss. 

The developed model is expected to generate novel lemma vector representations for the unseen human-readable definitions in the test set. Such a strategy allows users to search for words based on anticipated definitions or meanings.


## Task 2: Word Sense Disambiguation 
WSD focuses on identifying the specific sense of a word in a given context. The WSD gloss-based approach is categorized as a knowledge-based WSD method. This approach utilizes external resources, especially dictionaries. This technique involves determining a word's intended meaning by calculating the overlap between its contextual use and the provided gloss or definition.
In the realm of contemporary Arabic language, dictionaries have been utilized in the development of gloss-based WSD datasets, as evidenced in the works of (Jarrar et al., 2023; El-Razzaz et al., 2021). These studies employed the Ahmed Mokhtar Omar dictionary (Omar, 2008). Furthermore, the research conducted by (Jarrar et al., 2023) also incorporated the Al-Ghani Al-Zaher dictionary (Abul-Azm, 2014). 
The dataset consists of word, context, and context ID, and corresponding gloss ID. The developed model is expected to retrieve the suitable gloss ID for the word in the context from the WSD dictionary.


# Datasets

Datasets can be downloaded from [CODALAB-Task1](https://codalab.lisn.upsaclay.fr/competitions/18112) and  [CODALAB-Task2](https://codalab.lisn.upsaclay.fr/competitions/18121). This section details the structure of the JSON dataset files provided. 

### RD task:

The dataset itself comprises two core components: the dictionary data and the word embedding vectors. In the generation of these word embeddings, our approach is to utilize three distinct architectures of contextualized word embedding.
**Dictionary data**.
In the first iteration of KSAA-RD (Al-Matham et al., 2023), the dataset derived from a single source: the "Contemporary Arabic Language Dictionary" by Ahmed Mokhtar Omar (Omar, 2008). In this revised edition, we endeavor to expand our sources to encompass three dictionaries of Contemporary Arabic Language. The first of these is the "Contemporary Arabic Language Dictionary" by Ahmed Mokhtar Omar (Omar, 2008), a resource previously utilized in the first iteration KSAA-RD. The second is the newly released dictionary of the Arabic contemporary language "Mu'jam Arriyadh" (Altamimi et al., 2023). The third is the "Al Wassit LMF Arabic Dictionary" (Namly, 2015).
The three dictionaries employ the transferred version of this lexicon which conforms to the ISO standard, specifically the Lexical Markup Framework (LMF) (Aljasim et al., 2022; Altamimi et al., 2023; Namly, 2015). These dictionaries are based on lemmas rather than roots.
These dictionaries comprise words, commonly referred to as lemmas, and these may come with glosses, part of speech (POS), and examples. 
**Embedding data**
Experiments conducted on the first iteration of KSAA-RD (Al-Matham et al., 2023) revealed that fixed word embedding representations such as word2vec (Mikolov et al., 2013; Soliman et al., 2017) did not yield satisfactory performance. Consequently, in this edition, our focus will shift to contextualized word embeddings, which demonstrate improved performance in KSAA-RD. Accordingly, we will utilize advanced models such as Electra (Clark et al., 2020) and BERT (Devlin et al., 2019), to enhance the effectiveness of the system. Specifically, employing AraELECTRA (Antoun et al., 2021), AraBERTv2 (Antoun et al., 2020),, and camelBERT-MSA (Inoue et al., 2021)—referred to respectively as electra, bertseg, and bertmsa—for our methodologies. Specifically, our objective is to employ AraELECTRA (Antoun et al., 2021), AraBERTv2 (Antoun et al., 2020), and camelBERT-MSA (Inoue et al., 2021) in our methodologies. AraELECTRA, developed based on the ELECTRA framework. Instead of training the model to recover masked tokens, ELECTRA is designed to train a discriminator model. AraBERTv2 and camelBERT-MSA are both Arabic language models developed based on BERT architecture. The former utilizes Farasa segmentation, while the latter, camelBERT-MSA, is pretrained on a Modern Standard Arabic (MSA) corpus. 
As a concrete instance, here is an example from the training dataset for the Arabic dictionary: 

```json
#Arabic dictionary
{
"id":"ar.45",
"word":"عين",
"gloss":"عضو الإبصار في ...",
"pos":"n",
"electra":[0.4, 0.3, …],
"bertseg":[0.7, 2.9, …],
"bertmsa":[0.8, 1.4, …],
 }


```



### WSD task:

The dataset itself comprises two core components: the WSD context gloss mapping and dictionary data. 
The WSD context gloss mapping consists of word, context, context ID, and corresponding gloss ID. As a concrete instance, here is an example from the training dataset for the corresponding WSD JSON:

```json
#Arabic WSD
{
"context_id":"context.301",
"context":"يأتي برمجان اللغة العربية...",
"word": "اللغة",
"gloss_id":"gloss.305",
"lemma_id":"ar.200"

}

```

The dictionary data contains word, gloss, and gloss ID. The dictionary data is derived from the "Contemporary Arabic Language Dictionary" by Ahmed Mokhtar Omar (Omar, 2008). As a concrete instance, here is an example from the WSD dictionary:

```json
#Arabic WSD dictionary
{
"lemma_id": "ar.200",
"gloss_id":"gloss.305",
"gloss":" كُلُّ وسيلة لتبادل المشاعر والأفكار كالإشارات ..."
}
```


### Dataset description

The RD and WSD datasets are both in JSON format. The RD dataset includes about 222K entries, while the WSD dataset features around 28K entries, along with a dictionary that holds about 76K glosss useful for training. Both datasets are divided into three sections: a training split comprising 80%, a validation split representing 10%, and a test split representing 10% of the data points. Refer to Table 1 for data statistics.





# Baseline results
Here are the baseline results on the development set for the two tracks.
Scores were computed using the scoring script provided in this git (`RD/code/score.py` and `WSD/score.ipynb` ).

**RD Task:**
We leverage SOTA MARBERT (Abdul-Mageed, 2021) and CamelBERT-MSA (Inoue et al., 2021) models, employing fine-tuning techniques to excel in Arabic RD. These models are SOTA, proven by their superior performance in the shared task of KSAA-RD (Al-Matham et al., 2023), representing a winning approach.

|   Model    | Embedding   | Cosine  |  MSE    | Ranking
|------------|------------:|--------:|--------:|--------:
|            |   bertmsa   | 81.85   | 21.95   | 1.09
| CamelBERT  |   bertseg   | 84.36   | 5.55    | 1.26
|            |   electra   | 51.13   | 24.28   | 3.34
|            |   bertmsa   | 69.48   | 50.16   | 3.34
| MARBERT    |   bertseg   | 76.03   | 8.18    | 3.34
|            |   electra   | 73.68   | 14.57   | 0.84


**WSD Task:**
Initially, the dataset is enriched using lemma id by joining WSD entries with WSD dictionary to incorporate both relevant and irrelevant glosses, and cleaned. The model is trained to determine the relevance of a gloss to a word in context. The highest probability gloss is then calculated for each word in context, improving its ability to accurately identify context-appropriate meanings. We employ two approaches for WSD: 
+ Fine-tuning: The approach leverages BertForSequenceClassification, specifically with CamelBERT-MSA and AraBERTv2, due to their exceptional precision in identifying context-sensitive words. The target word in context is wrapped with special tokens "<token>word</token>". 
+ Neural Network: This approach involves feeding the three text embeddings (context, word, and gloss) from the multilingual-E5-base model into a simple LSTM neural network consisting of a 3D input layer, a single LSTM layer, a dense layer, and an output layer. The integrating these advanced embeddings enhances the LSTM's ability to accurately distinguish and disambiguate word senses, lead to improve performance. We refer to this configuration as the E5+LSTM model.

|            |   Dev       | Test    
|------------|------------:|--------:
| CamelBERT  |   91.54%    | 91.61%
| AraBERTv2  |   91.32%    | 91.25%
| E5 +LSTM   |   89.50%    | 88.83%

# Submission and evaluation

**RD:** The model evaluation process follows a hierarchy of metrics. The primary metric is the ranking metric, which is used to assess how well the model ranks predictions compared to ground truth values. If models have similar rankings, the secondary metric, mean squared error (MSE), is considered. Lastly, if further differentiation is needed, the tertiary metric, cosine similarity, provides additional insights. This approach ensures the selection of a top-performing and well-rounded model.
**WSD:** Accuracy is the primary metric, measures if the gloss ID of a word is correctly identified. It calculates the proportion of correct predictions overall.

The evaluation of shared tasks will be hosted through CODALAB. Here are the CODALAB links For each task:


+ **[CODALAB link for task 1](https://codalab.lisn.upsaclay.fr/competitions/18112).**
+ **[CODALAB link for task 2](https://codalab.lisn.upsaclay.fr/competitions/18121).**

### Expected output format

**RD:** During the evaluation phase, submissions are expected to reconstruct the same JSON format. The test JSON files will contain the "id" and the gloss keys. The participants should construct JSON files that contain at least the two following keys:

 + the original "id",
 + Any of the valid embeddings ("electra", "bertseg", and "bertmsa")

**WSD:** During the evaluation phase, submissions are expected to reconstruct the same JSON format. The test JSON files will contain the "context_id" two corresponding "gloss_id" entries, each accompanied by their ranking scores. The following is a detailed example for clarification:

```json
{
"context_id":"context.301",
"gloss_id":"gloss.305",
"ranking_score": 0.9
}
{
"context_id":"context.301",
"gloss_id":"gloss.466",
"ranking_score": 0.7
}

```


# Using this repository
To install the exact environment used for the scripts, see the `requirements.txt` file which lists the library used. Do note that the exact installation in the competition underwent supplementary tweaks: in particular, Colab Pro was utilized to run the experiment.

**RD:** The code useful to participants is stored in the `code/` directory.

+ To see options for a simple baseline on the reverse dictionary track, use:
```sh
$ python3 RD/code/revdict_entrypoint.py revdict --help
```
+ To verify the format of a submission, run:
```sh
$ python3 RD/code/revdict_entrypoint.py check-format $PATH_TO_SUBMISSION_FILE
```
+ To score a submission, use  
```sh
$ python3 RDcode/revdict_entrypoint.py score $PATH_TO_SUBMISSION_FILE --reference_files_dir $PATH_TO_DATA_FILE
```
Note that this requires the gold files, not available at the start of the
competition.

Other useful files to look at include `code/models.py`, where our baseline
architectures are defined, and `code/data.py`, which shows how to use the JSON
datasets with the PyTorch dataset API.

**WSD:** Colab notebook avalible 


