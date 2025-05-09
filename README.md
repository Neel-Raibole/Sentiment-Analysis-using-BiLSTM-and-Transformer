# Comparative Study: BiLSTM vs. DistilBERT for Sentiment Analysis

## 1. Overview

Sentiment analysis is a core NLP task that aims to extract subjective information—positive, neutral, or negative sentiment—from text. In this project, I perform a comparative study of two deep learning approaches on Yelp review data:

- **Bidirectional LSTM (BiLSTM)**: A recurrent architecture that processes text sequentially in both directions to capture context.
- **DistilBERT**: A distilled transformer model that leverages self-attention to encode rich bidirectional context more efficiently than the original BERT.

I re-labeled the Yelp reviews into three clear sentiment categories (negative = 0, neutral = 1, positive = 2), using 420K training samples and 30K test samples. Both models were trained and fine-tuned with Optuna-driven hyperparameter optimization on a stratified 10% subset to find their best configurations.

### Key findings:

- DistilBERT significantly outperforms BiLSTM across all metrics—achieving ~95.5% accuracy and F1 versus ~87% for the BiLSTM baseline.
- Transformer-based self-attention proves more robust, especially on neutral and context-dependent reviews, while the BiLSTM shows greater sensitivity to explicit sentiment words.

This study highlights how modern transformer architectures can deliver both higher predictive performance and more balanced error rates in real-world sentiment analysis tasks.

## 2. Tech Stack

- **Language & Runtime**: Python 3.9+
- **Data Handling**: pandas, NumPy
- **Deep Learning Frameworks**: PyTorch, TensorFlow & Keras (for embedding and preprocessing)
- **Transformers & Tokenization**: Hugging Face Transformers (DistilBertTokenizerFast, DistilBertForSequenceClassification)
- **Hyperparameter Optimization**: Optuna
- **Evaluation & Interpretability**: scikit-learn, LIME

## 3. Dataset

I use a curated Yelp review corpus, focused exclusively on clear sentiment classes:

- **Processed Split**:
  - Training set: 420,000 samples
  - Test set: 30,000 samples
- **Sentiment Mapping**:
  - 0 = Negative 
  - 1 = Neutral
  - 2 = Positive

Each review is stored as a text string with its integer sentiment label. Minimal cleaning was required, as the raw reviews contained no missing entries or corrupt records.

## 4. Methodology & Pipeline

This section walks through my end-to-end workflow—from raw Yelp reviews to final analyses—ensuring both BiLSTM and DistilBERT are trained and evaluated under identical conditions.

### 4.1 Data Loading & Splitting

- **Raw CSVs**:
  - yelp_dataset_train.csv (420K samples)
  - yelp_dataset_test.csv (30K samples)
- **Final Test Set**: the held-out 30K samples.
- **Training/Validation Split**: 80/20 split of the 420K training set → full-scale train (336K) + val (84K).
- **HPO Subset**: 10% stratified sample of the 420K → tune/train_small (33.6K) + tune/val_small (8.4K).

### 4.2 Tokenization & Preprocessing

- **BiLSTM**
  1. Keras Tokenizer (vocab_size=20,000, <OOV> token) → word→index sequences.
  2. pad_sequences(..., maxlen=512) for uniform length.
- **DistilBERT**
  1. Hugging Face DistilBertTokenizerFast (base-uncased) → input_ids + attention_mask.
  2. Auto-add special tokens ([CLS], [SEP]) + truncate/pad to 512 tokens.

### 4.3 Hyperparameter Optimization (Optuna) ([BiLSTM HPO File Hyperlink], [DistilBERT HPO File Hyperlink])

- **Common Setup**
  - Train on train_small, evaluate on val_small.
  - 20 trials, each running 3 epochs.
  - Studies persisted in SQLite for resumption and analysis.
- **BiLSTM Search Space** (maximize val accuracy)
  - lstm_units: [64, 256]
  - learning_rate: 1e-5→1e-2 (log scale)
  - weight_decay: 1e-6→1e-2 (log scale)
- **DistilBERT Search Space** (minimize val loss)
  - learning_rate: 1e-6→1e-3 (log scale)
  - weight_decay: 1e-6→1e-2 (log scale)
  - lr_scheduler_type: linear ∣ cosine ∣ polynomial

### 4.4 Final Model Training

- **BiLSTM** ([BiLSTM Training File Hyperlink])
  1. Instantiate BiLSTMClassifier with tuned units, LR, weight decay.
  2. Train 10 epochs on 420K train set; track train/val loss, acc, precision, recall, F1 per epoch.
  3. Save bilstm_model.pth + tokenizer.pkl.
- **DistilBERT** ([DistilBERT Training File Hyperlink])
  1. Load DistilBertForSequenceClassification with num_labels=3.
  2. Configure TrainingArguments using tuned hyperparameters, 500 warmup steps, fp16, epoch checkpointing.
  3. Train 3 epochs on 420K train set; track metrics via a TrainerCallback.
  4. Save best model & tokenizer under ./final_model.

### 4.5 Evaluation & Post-Hoc Analyses

- **Held-out Test Evaluation** ([BiLSTM Test File Hyperlink], [DistilBERT Test File Hyperlink])
  - Tokenize/pad the 30K reviews; batch inference → compute accuracy, precision, recall, F1 (weighted); plot confusion matrix.
- **Short vs. Long Reviews**
  - Split test set by word count threshold (≤50 vs. >50 words); evaluate both subsets separately and visualize accuracy differences.
- **Interpretability with LIME**
  - Sample N reviews per class; use LimeTextExplainer to extract top influential words; aggregate and plot top tokens per sentiment.
- **Attention Heatmaps** (DistilBERT only)
  - For a sample review, extract first-layer attention weights; average across heads and plot token×token heatmap.

## 5. Explaining BiLSTM

### 5.1 What Is a Bidirectional LSTM?

<img src="https://github.com/Neel-Raibole/Sentiment-Analysis-using-BiLSTM-and-Transformer/blob/main/images/BiLSTM%20Model.png" alt="BiLSTM - Model Architecture" width="1000"/>

A Bidirectional Long Short-Term Memory (BiLSTM) network is an extension of the standard LSTM a type of recurrent neural network (RNN) designed to model sequential data where two LSTM layers process the sequence in opposite directions:

1. **Forward LSTM**
   - Reads the input sequence from token 1 → token n.
   - Captures past context for each position.
2. **Backward LSTM**
   - Reads the same sequence from token n → token 1.
   - Captures future context for each position.

By concatenating the hidden states from both directions at each time step, a BiLSTM can leverage information from both preceding and succeeding tokens. This is critically important in natural language understanding, where the meaning of a word often depends on the words that come after it as well as those that came before.

Key advantages:
- **Richer Context**: Unlike unidirectional RNNs that only see past tokens, BiLSTMs incorporate bidirectional context, improving understanding of polysemous words and long-range dependencies.
- **Mitigated Vanishing Gradients**: Built-in LSTM gating mechanisms (input, forget, and output gates) help maintain long-term dependencies without suffering from vanishing or exploding gradients.
- **Flexible for Sequence Classification**: BiLSTM outputs can be pooled, attended over, or further processed for classification, sequence labeling, or other downstream tasks.

### 5.2 My BiLSTM Model Architecture

Below is a layer-by-layer breakdown of the BiLSTM I implemented for three-class sentiment analysis:

| Layer | Configuration | Purpose |
|-------|--------------|---------|
| Embedding | nn.Embedding(num_embeddings=20000, embedding_dim=100) | Maps each integer token to a 100-dimensional dense vector, learning word representations during training. |
| Bidirectional LSTM | nn.LSTM(input_size=100, hidden_size=H, num_layers=1, bidirectional=True, batch_first=True) | Processes the sequence in both forward and backward directions. Outputs a sequence of hidden states of size H*2 per time step. |
| Mean Pooling | torch.mean(lstm_output, dim=1) | Aggregates the sequence of hidden states into a single fixed-length vector by averaging across time steps. |
| Dropout | nn.Dropout(p=0.3) | Randomly zeroes 30% of the pooled features during training to reduce overfitting. |
| Fully Connected (FC) | nn.Linear(in_features=H*2, out_features=3) | Projects the pooled representation to three logits—one per sentiment class. |
| Softmax | nn.Softmax(dim=1) | Converts logits into normalized class probabilities. |

Here, H (hidden_dim) is a hyperparameter tuned by Optuna (best value found: 64 units per direction). Since the LSTM is bidirectional, the concatenated hidden dimension is H_forward + H_backward = 2H.

**Python Code:**
```python
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: [batch_size, seq_len]
        emb = self.embedding(x)                     # [batch_size, seq_len, embedding_dim]
        lstm_out, _ = self.lstm(emb)                # [batch_size, seq_len, hidden_dim*2]
        pooled = torch.mean(lstm_out, dim=1)        # [batch_size, hidden_dim*2]
        dropped = self.dropout(pooled)              # [batch_size, hidden_dim*2]
        logits = self.fc(dropped)                   # [batch_size, 3]
        return self.softmax(logits)                 # [batch_size, 3] probabilities
```

- **Why Mean Pooling?**
  By averaging across all time steps, mean pooling produces a global representation of the entire sequence. It is simple, parameter-free, and empirically effective for sequence classification tasks where an overall summary vector suffices.

- **Regularization & Generalization**
  - Dropout (p=0.3) reduces overfitting by randomly deactivating neurons during training.
  - Weight Decay (AdamW optimizer) further penalizes large weights, improving generalization.

This architecture strikes a balance between capacity (through bidirectional recurrence) and efficiency (single LSTM layer, modest embedding size), making it a solid baseline for sentiment classification on large-scale text data.

## 6. Explaining DistilBERT

### 6.1 What Is DistilBERT?

<img src="https://github.com/Neel-Raibole/Sentiment-Analysis-using-BiLSTM-and-Transformer/blob/main/images/DistilBERT%20Model.png" alt="DistilBERT - Model Architecture" width="1000"/>

DistilBERT is a lightweight, distilled version of the original BERT model designed to retain most of BERT's language understanding capabilities while being faster and smaller. It achieves this through knowledge distillation, where a smaller "student" model learns to mimic the outputs of a larger "teacher" model.

Key characteristics:
- **Reduced Depth**:
  - Original BERT-base has 12 transformer encoder layers.
  - DistilBERT reduces this to 6 layers, halving the number of parameters.
- **Same Hidden Size**:
  - Retains the same hidden dimension (768) and intermediate feed-forward size (3072) as BERT-base.
- **Self-Attention Mechanism**:
  - Each encoder layer uses multi-head self-attention to relate every token in the sequence to every other token, capturing rich contextual relationships.
- **Knowledge Distillation Objectives**:
  - During pre-training, DistilBERT's student model is trained to match the teacher's soft target distributions (logits), attention patterns, and hidden states, combining three loss terms:
    1. Soft Target Loss (KL divergence between teacher and student logits)
    2. Cosine Embedding Loss (matching hidden representations)
    3. Language Modeling Loss (predicting masked tokens)

The result is a model that runs approximately 60% faster and is 40% smaller than BERT-base, yet retains around 97% of its performance on language understanding benchmarks.

### 6.2 My DistilBERT Fine-Tuning Architecture

For sequence classification, I fine-tune a pre-trained DistilBertForSequenceClassification with the following components:

| Component | Configuration | Purpose |
|-----------|--------------|---------|
| Pre-trained Encoder | 6 × Transformer Encoder Layers<br/>Hidden size = 768<br/>12 attention heads per layer | Extracts contextualized token representations via self-attention and feed-forward. |
| Dropout | p = 0.3 | Regularizes fine-tuning by randomly zeroing-out hidden units before the classification head. |
| Classification Head (Linear) | nn.Linear(in_features=768, out_features=3) | Projects the pooled [CLS] token embedding to three logits (negative, neutral, positive). |
| Softmax Activation | Implicit in Trainer's cross-entropy loss implementation | Converts logits into class probabilities. |

Under the hood:

**Python Code:**
```python
from transformers import DistilBertForSequenceClassification, TrainingArguments, Trainer
# Load the pre-trained DistilBERT model with a classification head
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=3       # three sentiment classes
)

training_args = TrainingArguments(
    output_dir = "./final_model",
    num_train_epochs = 3,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size = 64,
    warmup_steps = 500,
    learning_rate=1.336606944413412e-05,     # tuned via Optuna
    weight_decay=2.5038567374664683e-05,     # tuned via Optuna
    lr_scheduler_type="linear",              # or cosine / polynomial as tuned    
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    load_best_model_at_end = True,
    metric_for_best_model = "accuracy",
    logging_dir = "./final_logs",
    logging_steps = 10,
    save_total_limit = 3,
    fp16 = True
)
```

- **Input Representation**
  - Each review is tokenized into subword tokens with special tokens:
    - [CLS] at the start (whose final hidden state is used for classification)
    - [SEP] at the end
  - Generates two tensors per review: input_ids and attention_mask.

- **Forward Pass**
  1. Token Embeddings + Positional / Segment Embeddings → combined input to first encoder.
  2. Transformer Stack (6 layers) → contextualized token embeddings.
  3. Pooled Output: the embedding corresponding to the [CLS] token is passed to the classification head.
  4. Dropout applied to this pooled vector.
  5. Linear Projection → logits for each sentiment class.
  6. Cross-Entropy Loss computed against true labels during training; during inference, softmax yields probabilities.

- **Fine-Tuning Details**
  - Optimizer: AdamW with learning_rate=1.3366e-05 & weight_decay=2.5039e-05
  - Learning Rate Scheduler: linear (or cosine/polynomial, tuned via Optuna)
  - Warm-up Steps: 500 (gradually ramp up learning rate at start)
  - Mixed Precision (fp16): accelerates training on modern GPUs.
  - Checkpointing: Save model at each epoch; load best checkpoint based on validation accuracy.

This fine-tuning setup leverages the rich pre-trained language representations of DistilBERT, adapting them to my three-class sentiment analysis task with a minimal, efficient classification head.

## 7. Evaluation & Results

### 7.1 Overall Performance

| Metric | BiLSTM | DistilBERT |
|--------|--------|------------|
| Accuracy | 86.88% | 95.46% |
| F1 Score | 86.76% | 95.45% |
| Precision | 86.88% | 95.45% |
| Recall | 86.88% | 95.46% |

Key takeaway: DistilBERT outperforms the BiLSTM across all standard classification metrics by ~8–9%.

<p align="center">
  <img src="https://github.com/Neel-Raibole/Sentiment-Analysis-using-BiLSTM-and-Transformer/blob/main/images/cm%20-%20BiLSTM.png" alt="Confusion Matrix for BiLSTM" width="400"/>
  <img src="https://github.com/Neel-Raibole/Sentiment-Analysis-using-BiLSTM-and-Transformer/blob/main/images/cm%20-%20DistilBERT.png" alt="Confusion Matrix for DistilBERT" width="400"/>
</p>

### 7.2 Short vs. Long Reviews

| Model | Short Reviews Accuracy | Long Reviews Accuracy |
|-------|------------------------|------------------------|
| BiLSTM | 85.74% | 87.26% |
| DistilBERT | 94.83% | 96.13% |

- Both models improve on longer reviews, but DistilBERT gains more from additional context (1.3 pp vs. 1.5 pp for BiLSTM).
- BiLSTM shows greater sensitivity to review length, performing ~1.5 pp better on long vs. short.

<p align="center">
  <img src="https://github.com/Neel-Raibole/Sentiment-Analysis-using-BiLSTM-and-Transformer/blob/main/images/cm%20-%20BiLSTM%20-%20LR.png" alt="LR Confusion Matrix for BiLSTM" width="400"/>
  <img src="https://github.com/Neel-Raibole/Sentiment-Analysis-using-BiLSTM-and-Transformer/blob/main/images/cm%20-%20BiLSTM%20-%20SR.png" alt="SR Confusion Matrix for BiLSTM" width="400"/>
</p>
<p align="center">
  <img src="https://github.com/Neel-Raibole/Sentiment-Analysis-using-BiLSTM-and-Transformer/blob/main/images/cm%20-%20DistilBERT-%20LR.png" alt="LR Confusion Matrix for DistilBERT" width="400"/>
  <img src="https://github.com/Neel-Raibole/Sentiment-Analysis-using-BiLSTM-and-Transformer/blob/main/images/cm%20-%20DistilBERT-%20SR.png" alt="SR Confusion Matrix for DistilBERT" width="400"/>
</p>
### 7.3 Model Interpretability (LIME)

I used LIME to surface the top influential words driving each model's predictions, sampling 50–100 reviews per class.

- **Negative Sentiment**
  - Common to both: "horrible", "rude", "not"
  - BiLSTM emphasis: "disgusting", "never"
  - DistilBERT emphasis: more balanced on "bad", "worst"

- **Neutral Sentiment**
  - Common to both: "but", "good", "okay"
  - BiLSTM emphasis: vague descriptors like "pretty", "bit"
  - DistilBERT emphasis: transition words that indicate mixed sentiment

- **Positive Sentiment**
  - Common to both: "great", "love", "amazing"
  - BiLSTM emphasis: emotionally expressive terms ("favourite", "highly")
  - DistilBERT emphasis: context-rich words ("delicious", "friendly")

Insight: The BiLSTM relies more heavily on overt sentiment words, whereas DistilBERT captures a blend of emotional and contextual cues—aligning with its superior performance on nuanced or context-dependent inputs.

This combined evaluation highlights not only how the models perform quantitatively, but also why they make certain predictions.


<img src="https://github.com/Neel-Raibole/Sentiment-Analysis-using-BiLSTM-and-Transformer/blob/main/images/BiLSTM%20-%20Influential%20Words.png" alt="BiLSTM- Influential Words" width="600"/>
<img src="https://github.com/Neel-Raibole/Sentiment-Analysis-using-BiLSTM-and-Transformer/blob/main/images/DistilBERT%20-%20Influential%20Words.png" alt="DistilBERT Influential Words" width="600"/>

## Repository Structure

```
project-root/
│   .gitattributes
│   ReadMe.docx
│   README.md
│   structure.txt
│   yelp_dataset_train.csv
│   yelp_dataset_test.csv
│
├── DistilBERT/
│   │   config.json
│   │   model.safetensors
│   │   optuna_study.db
│   │   special_tokens_map.json
│   │   tokenizer_config.json
│   │   vocab.txt
│   │   distilbert_hpo.db
│   │   distilbert_hpo(1).db
│   │
│   ├── DistilBERT_HPO.ipynb
│   ├── DistilBERT_Training.ipynb
│   ├── DistilBERT_Testing.ipynb
│   └── DistilBERT_Additional_Testing.ipynb
│
└── LSTM/
    │   bilstm_model.pth
    │   LSTM_HPO.db
    │   tokenizer.pkl
    │
    ├── LSTM_HPO.ipynb
    ├── LSTM_Training.ipynb
    ├── LSTM_Testing.ipynb
    └── LSTM_Additional_Testing.ipynb
```

## 8. Future Work

- **Incorporate Ambiguous Ratings**
  Extend the task to include 2-star and 4-star reviews, exploring methods (e.g., soft labels or multi-task learning) to handle mixed or subtle sentiment.

- **Fine-Grained Sentiment & Regression**
  Move beyond three classes to either more granular categories (e.g., very negative ↔ very positive) or a continuous regression score for sentiment intensity.

- **Domain Adaptation & Robustness**
  - Fine-tune on other review domains (e.g., Amazon, Yelp restaurants vs. services) to test generalizability.
  - Investigate adversarial robustness and bias mitigation for more reliable real-world deployment.
