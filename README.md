# tech_fluencer_rnn
A Jupyter notebook lab that trains an LSTM neural network on real LinkedIn influencer posts to generate new content word-by-word. It covers the full NLP pipeline — text cleaning, tokenization, n-gram sequencing, and temperature-controlled text generation.

# 🤖 Tech-Fluencer RNN — LinkedIn Post Generator

> *"I woke up at 4 AM today to..."* — an LSTM writes the rest.

A hands-on Jupyter notebook that teaches you to build a **next-word prediction model** trained on LinkedIn influencer posts. You implement every step — cleaning, tokenizing, sequencing, training, and generating — and a built-in grader scores your code as you go.

---

## 📓 Notebook

`Copy_of_tech_fluencer_rnn.ipynb`

---

## 🎯 Objective

Train an LSTM to generate cringe LinkedIn posts word-by-word. Instead of *classifying* text (spam detection), you flip the task: teach an RNN to *predict the next word*, then sample from it in a loop to generate full sentences.

---

## 🗂️ What's Inside

| Part | Topic | Points |
|------|-------|--------|
| Part 0 | Setup, imports, dataset loading | — |
| Part 1 | Text cleaning — lowercase + strip punctuation | 🟡 10 |
| Part 2 | Keras `Tokenizer` + `texts_to_sequences` | 🟢 5 + 🟢 5 |
| Part 3 | N-gram sequence builder ⭐ | 🔴 20 |
| Part 4 | `pad_sequences` + X / y split | 🟡 10 + 🟢 5 |
| Part 5 | Build the model — `Embedding → LSTM → Dropout → Dense` | 🟡 10 |
| Part 6 | Compile (`sparse_categorical_crossentropy`) + train | 🟡 10 + 🟢 5 |
| Part 7 | Temperature-sampled text generation ⭐⭐ | 🔴 20 + 🟡 10 |
| Part 8 | Bonus challenges | — |

**Total: 110 points across 11 tasks**

---

## 🏆 Rank Ladder

| Score | Rank |
|-------|------|
| 0–25% | 🌱 Curious Onlooker |
| 25–50% | 💼 Networking Newbie |
| 50–70% | 🚀 Aspiring Thought Leader |
| 70–85% | 📈 Growth Hacker |
| 85–99% | 🦄 Senior Disruptor |
| 100% | 👑 Tech-Fluencer Royalty |

---

## 🌡️ Temperature Sampling

After training, `generate_text()` takes a `temperature` argument that controls creativity:

| Temperature | Output style |
|-------------|-------------|
| `0.4` | Repetitive, safe — sticks to common words |
| `1.0` | Balanced — model's true distribution |
| `1.5` | Wild and unpredictable — maximum chaos |

```python
print(generate_text("I woke up at 4 am today to",
                    num_words=40, model=model,
                    max_seq_len=max_seq_len, temperature=1.0))
```

---

## 📦 Requirements

```
tensorflow>=2.10.0
numpy>=1.23.0
pandas>=1.5.0
jupyter>=1.0.0
```

Install with:

```bash
pip install -r requirements.txt
```

---

## 📥 Dataset

The notebook uses the **LinkedIn Influencers' Data** from Kaggle:
🔗 [kaggle.com/datasets/shreyasajal/linkedin-influencers-data](https://www.kaggle.com/datasets/shreyasajal/linkedin-influencers-data)

Download and place as `influencers_data.csv` in the same folder as the notebook.
**No dataset?** No problem — the notebook auto-loads 60 built-in sample posts so you can keep moving.

---

## ▶️ How to Run

```bash
jupyter notebook Copy_of_tech_fluencer_rnn.ipynb
```

Or open directly in **Google Colab** for free GPU access (recommended for the full dataset).

---

## 🔥 Bonus Challenges

- **SimpleRNN vs LSTM** — swap the layer, compare quality
- **Top-k sampling** — only sample from the top-k most likely tokens
- **Nucleus (top-p) sampling** — cumulative probability cutoff
- **Stacked LSTMs** — `LSTM(150, return_sequences=True)` → `LSTM(150)`
- **Overfitting analysis** — add `validation_split=0.1`, plot train vs val loss
- **Tiny Transformer decoder** — replace the LSTM entirely

---

## 📄 License

MIT
