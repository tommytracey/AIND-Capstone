#### Udacity Artificial Intelligence Nanodegree
### Term 2: Capstone Project
# Machine Translation with RNNs

##### &nbsp;

<img src="images/translation.gif" width="100%" align="top-left" alt="" title="RNN" />

*Image credit: [xiandong79.github.io](https://xiandong79.github.io/seq2seq-%E5%9F%BA%E7%A1%80%E7%9F%A5%E8%AF%86)*


##### &nbsp;

## Goal
In this project, we build a deep neural network that functions as part of an end-to-end machine translation pipeline. The pipeline accepts English text as input and returns the French translation. The goal is to achieve the highest translation accuracy possible.

##### &nbsp;

## Background
The ability to communicate with one another is a fundamental part of being human. There are nearly 7,000 different languages worldwide. As our world becomes increasingly connected, language translation provides a critical cultural and economic bridge between people from different countries and ethnic groups. Some of the more obvious use-cases include:
- **business**: international investment, contracts, trade, and finance
- **commerce**: travel, purchase of foreign goods and services, customer support
- **media**: accessing information via search, sharing information via social networks, localization of content and advertising
- **education**: sharing of ideas, collaboration, translation of research papers
- **government**: foreign relations, negotiation  


To meet this need, technology companies are investing heavily in machine translation. This investment paired with recent advancements in deep learning have yielded major improvements in translation quality. According to Google, [switching to deep learning produced a 60% increase in translation accuracy](https://www.washingtonpost.com/news/innovations/wp/2016/10/03/google-translate-is-getting-really-really-accurate) compared to the phrase-based approach used previously. Today, translation applications from Google and Microsoft can translate over 100 different languages and are approaching human-level accuracy for many of them.

However, while machine translation has made a ton of progress, it's still not perfect. :grimacing:

<img src="images/fuck-veges.png" width="50%" align="top-left" alt="" title="RNN" />


##### &nbsp;

## Approach
Below is a summary of the various preprocessing and modeling steps. The high-level steps include:

1. Preprocessing: load and examine data, cleaning, tokenization, padding
1. Modeling: build, train, and test the model
1. Prediction: generate specific translations of English to French, and compare the output translations to the ground truth translations
1. Iteration: iterate on the model, experimenting with different architectures

For a more detailed walkthrough including the source code, check out the Jupyter notebook in the main directory ([machine_translation.ipynb](https://github.com/tommytracey/AIND-Capstone/blob/master/machine_translation.ipynb)).

### Preprocessing
- cleaning
- tokenization
- padding

### Modeling

Here we evaluate different RNN architectures:
  - simple
  - embeddings
  - encoder-decoder
  - bidirectional (backward and forward context)
  - GRU
  - LSTM (not tested in this project, but done is separate Udacity project found [here](https://github.com/tommytracey/udacity/tree/master/deep-learning-nano/projects/4-language-translation#build-the-neural-network)


##### &nbsp;

## Results
- _coming soon_


##### &nbsp;

## Future Improvements
- do proper data split (train, validation, test)
- LSTM + attention
- Residual layers (Google paper)
- Embedding Language Model (ELMo)
- Transformer model
- train on different text corpuses

##### &nbsp;
##### &nbsp;

---

# Project Starter Code
In case you want to run this project yourself, below is the project starter code.

## Setup
The original Udacity repo for this project can be found [here]().

This project requires GPU acceleration to run efficiently. Support is available to use either of the following two methods for accessing GPU-enabled cloud computing resources.

### Udacity Workspaces (Recommended)

Udacity Workspaces provide remote connection to GPU-enabled instances right from the classroom. Refer to the classroom lesson for this project to find an overview of navigating & using Jupyter notebook Workspaces.

### Amazon Web Services (Optional)

Please refer to the Udacity instructions for setting up a GPU instance for this project, and refer to the project instructions in the classroom for setup. The recommended AMI should include compatible versions of all required software and libraries to complete the project. [link for AIND students](https://classroom.udacity.com/nanodegrees/nd889/parts/16cf5df5-73f0-4afa-93a9-de5974257236/modules/53b2a19e-4e29-4ae7-aaf2-33d195dbdeba/lessons/2df3b94c-4f09-476a-8397-e8841b147f84/project)

### Install
- Python 3
- NumPy
- TensorFlow 1.x
- Keras 2.x

## Submission
When you are ready to submit your project, do the following steps:
1. Ensure you pass all points on the [rubric](https://review.udacity.com/#!/rubrics/1004/view).
2. Submit the following in a zip file:
  - `helper.py`
  - `machine_translation.ipynb`
  - `machine_translation.html`

### Converting to HTML

There are several ways to generate an HTML copy of the notebook:

 - Running the last cell of the notebook will export an HTML copy

 - Navigating to **File -> Download as -> HTML (.html)** within the notebook

 - Using `nbconvert` from the command line

    $ pip install nbconvert
    $ nbconvert machine_translation.ipynb
