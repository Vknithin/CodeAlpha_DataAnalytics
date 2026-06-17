"""
============================================================
CODEALPHA INTERNSHIP — TASK 4: SENTIMENT ANALYSIS
============================================================
Author     : [Nithin Varghese Kurian]
Dataset    : Sample Amazon-style book reviews (built-in)
Libraries  : nltk (VADER), pandas, matplotlib, seaborn,
             wordcloud, collections
NLP Tools  : VADER (Valence Aware Dictionary & sEntiment
             Reasoner) — ideal for social/review text.
Description: Classifies 300 sample book reviews as
             Positive, Neutral, or Negative, visualises
             sentiment distribution, identifies top words
             per sentiment, and generates word clouds.
============================================================
"""

import os, re, random
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from collections import Counter
from wordcloud import WordCloud
import warnings
warnings.filterwarnings("ignore")

# ── NLTK downloads ─────────────────────────────────────────
for pkg in ["vader_lexicon", "punkt", "stopwords", "punkt_tab"]:
    nltk.download(pkg, quiet=True)

# ── Paths ──────────────────────────────────────────────────
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────
sns.set_theme(style="whitegrid"); plt.rcParams["figure.dpi"] = 150
POS_COLOR = "#2ecc71"; NEG_COLOR = "#e74c3c"; NEU_COLOR = "#3498db"


# ============================================================
# 1. SAMPLE DATASET  (simulates Amazon-style book reviews)
# ============================================================
POSITIVE_REVIEWS = [
    "Absolutely loved this book! The characters were so well developed and the plot kept me hooked.",
    "A masterpiece! Beautifully written with incredible depth. Highly recommend to everyone.",
    "One of the best books I have ever read. The author's storytelling is truly exceptional.",
    "Couldn't put it down. Stayed up all night to finish it. Brilliant writing!",
    "Fantastic read! The descriptions are vivid and the pacing is perfect throughout.",
    "This book changed my perspective on life. Deeply moving and wonderfully crafted.",
    "Superb writing! Every chapter left me wanting more. An absolute must-read.",
    "Outstanding! The narrative is engaging, the characters feel real. Loved every page.",
    "Phenomenal book. The author has a gift for storytelling. Definitely a favourite.",
    "Incredible journey from start to finish. Emotional, thought-provoking, and beautiful.",
    "I adored this novel. The plot twists were surprising yet perfectly logical.",
    "A wonderful escape! Transports you to another world entirely. Five stars easily.",
    "Brilliant pacing, compelling characters, and an unforgettable ending. Loved it.",
    "This book is a gem! Engrossing from the very first page. Highly recommended.",
    "Captivating storytelling with rich imagery. A delightful and uplifting read.",
    "Loved the depth of the characters. The story is both moving and inspiring.",
    "What a wonderful book! The author has created something truly special here.",
    "Thoroughly enjoyed this. Smart, witty, and emotionally resonant. Great book.",
    "Engaging, well-paced, and thought-provoking. One of my favourite reads this year.",
    "The writing is exquisite. Every sentence feels intentional and beautifully crafted.",
    "Excellent book. The world-building is extraordinary and the plot is gripping.",
    "Heartwarming and inspiring. This book left me with a big smile on my face.",
    "Perfect blend of mystery and emotion. Couldn't stop reading. Highly recommend!",
    "Brilliant! A rare book that is both intellectually stimulating and emotionally rich.",
    "Loved every moment of this book. Fresh, original, and endlessly entertaining.",
]

NEGATIVE_REVIEWS = [
    "Very disappointing. The plot was predictable and the characters were flat.",
    "Couldn't get past chapter three. The writing style was tedious and hard to follow.",
    "Complete waste of money. The story goes nowhere and the ending is unsatisfying.",
    "Dull and unoriginal. Nothing in this book felt fresh or exciting at all.",
    "The characters are one-dimensional and the dialogue feels forced and unrealistic.",
    "Expected so much more based on the reviews. Ended up being very boring.",
    "Poor pacing and a confusing narrative. I struggled to finish this book.",
    "Terrible book. Lots of potential wasted on a messy, incoherent plot.",
    "Badly written with too many clichés. Very hard to engage with the story.",
    "I regret buying this. The story dragged and the ending was deeply unsatisfying.",
    "Not worth reading. The author spends too much time on irrelevant details.",
    "A big letdown. The cover and description promised so much more than delivered.",
    "Painfully slow plot and weak character development. Not recommended at all.",
    "Disjointed writing and a nonsensical conclusion. Very frustrating read.",
    "The worst book I have read this year. Dull, lifeless, and deeply boring.",
    "The hype was completely unjustified. This book is mediocre at best.",
    "Failed to connect with any character. The story felt hollow and pointless.",
    "Repetitive and lacking originality. The author needed a much stronger editor.",
    "Confusing structure with too many subplots that go nowhere. Disappointing.",
    "Would not recommend this to anyone. A poorly executed idea with weak writing.",
    "Overlong and self-indulgent. The author needed to cut at least 100 pages.",
    "Dreadful pacing. Took forever to get anywhere, and the payoff was not worth it.",
    "Flat, lifeless prose with no sense of urgency or tension. Very dull.",
    "Hard to follow and ultimately pointless. Didn't enjoy a single chapter.",
    "Disappointing from start to finish. Nothing redeemable about this book.",
]

NEUTRAL_REVIEWS = [
    "The book was okay. Some parts were interesting but others felt slow.",
    "Not bad but not great. Decent writing but the story didn't fully grab me.",
    "A mixed bag. Some chapters were engaging while others felt unnecessary.",
    "Average read. Has its moments but overall nothing particularly memorable.",
    "The story was fine. A decent way to pass the time but nothing special.",
    "It was alright. Some good ideas that were only partially developed.",
    "Fairly standard in terms of plot. Readable but not remarkable.",
    "Mediocre. Not the worst but certainly not among the best I have read.",
    "Had potential but didn't quite deliver. A serviceable but forgettable read.",
    "The writing was competent but the story lacked real depth or originality.",
    "It was okay. Some interesting moments but nothing that left a lasting impression.",
    "Passable. The kind of book you read once and don't think much about.",
    "Neither here nor there. An average book with a few good scenes.",
    "Decent enough. Kept me mildly entertained but did not leave a lasting impact.",
    "A moderate read. The author has some talent but the execution was inconsistent.",
    "Not terrible but not impressive either. A mostly forgettable experience.",
    "The characters were believable but the plot was a bit too predictable.",
    "Hit and miss. Some chapters were genuinely good; others felt like filler.",
    "An unremarkable book. Pleasant enough but I won't be seeking out more by this author.",
    "Just okay. You could read it or skip it and not miss much either way.",
    "Average writing and an average story. Neither impressed nor disappointed me.",
    "It had some nice ideas but the execution was only moderately successful.",
    "Competent but uninspiring. The book did what it set out to do, just about.",
    "Okay for a light read. Not something I would enthusiastically recommend though.",
    "A passable story with forgettable characters. Exactly what you'd expect.",
]


def build_dataset(n: int = 300) -> pd.DataFrame:
    """Assemble a labelled review dataset from the sample pools."""
    random.seed(42)
    n_each = n // 3
    data = (
        [(rev, "Positive") for rev in random.choices(POSITIVE_REVIEWS, k=n_each)] +
        [(rev, "Negative") for rev in random.choices(NEGATIVE_REVIEWS, k=n_each)] +
        [(rev, "Neutral")  for rev in random.choices(NEUTRAL_REVIEWS,  k=n_each)]
    )
    random.shuffle(data)
    df = pd.DataFrame(data, columns=["review_text", "true_label"])
    df.index.name = "review_id"
    return df


# ============================================================
# 2. VADER SENTIMENT ANALYSER
# ============================================================
def run_vader(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[2/5] Running VADER sentiment analyser …")
    sia = SentimentIntensityAnalyzer()

    scores = df["review_text"].apply(sia.polarity_scores)
    df["neg"]      = scores.apply(lambda x: x["neg"])
    df["neu"]      = scores.apply(lambda x: x["neu"])
    df["pos"]      = scores.apply(lambda x: x["pos"])
    df["compound"] = scores.apply(lambda x: x["compound"])

    def classify(compound):
        if compound >= 0.05:  return "Positive"
        if compound <= -0.05: return "Negative"
        return "Neutral"

    df["predicted_label"] = df["compound"].apply(classify)

    # Accuracy
    correct = (df["predicted_label"] == df["true_label"]).sum()
    print(f"  Accuracy vs. true labels: {correct/len(df)*100:.1f}% ({correct}/{len(df)})")
    return df


# ============================================================
# 3. TEXT PRE-PROCESSING
# ============================================================
STOP_WORDS = set(stopwords.words("english"))
STOP_WORDS.update(["book","read","story","author","reading","books","also","one","get"])

def preprocess(text: str) -> list[str]:
    text  = text.lower()
    text  = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    return [t for t in tokens if t.isalpha() and t not in STOP_WORDS and len(t) > 2]

def get_top_words(df: pd.DataFrame, label: str, n: int = 15) -> pd.Series:
    texts  = df[df["predicted_label"] == label]["review_text"]
    words  = [w for review in texts for w in preprocess(review)]
    return pd.Series(Counter(words).most_common(n)).apply(lambda x: pd.Series(x, index=["word","count"]))


# ============================================================
# 4. VISUALISATIONS
# ============================================================
def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  ✔ Saved: {path}")


def viz_overview(df):
    print("\n[3/5] Generating overview charts …")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Sentiment Analysis — Overview", fontsize=15, fontweight="bold")

    # Pie — predicted
    ax = axes[0]
    counts = df["predicted_label"].value_counts()
    colors = [POS_COLOR if l=="Positive" else NEG_COLOR if l=="Negative" else NEU_COLOR
              for l in counts.index]
    wedges, texts, autos = ax.pie(counts.values, labels=counts.index,
                                   autopct="%1.1f%%", colors=colors,
                                   startangle=90, wedgeprops={"edgecolor":"white"})
    for auto in autos: auto.set_fontsize(11)
    ax.set_title("Predicted Sentiment Distribution")

    # Compound score distribution
    ax = axes[1]
    for label, color in [("Positive", POS_COLOR),("Neutral", NEU_COLOR),("Negative", NEG_COLOR)]:
        subset = df[df["predicted_label"] == label]["compound"]
        if len(subset) > 5:
            sns.histplot(subset, ax=ax, color=color, label=label, bins=20,
                         kde=True, stat="density", alpha=0.4, edgecolor="white")
        else:
            ax.bar(subset.mean(), 1, color=color, alpha=0.6, width=0.05, label=label)
    ax.axvline(0.05,  color="grey", ls="--", lw=1, alpha=0.7)
    ax.axvline(-0.05, color="grey", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("VADER Compound Score"); ax.set_title("Score Distribution by Sentiment")
    ax.legend()

    # True vs Predicted confusion-style bar
    ax = axes[2]
    ct = pd.crosstab(df["true_label"], df["predicted_label"])
    ct.plot.bar(ax=ax, color=[NEG_COLOR, NEU_COLOR, POS_COLOR], edgecolor="white", rot=0)
    ax.set_xlabel("True Label"); ax.set_ylabel("Count")
    ax.set_title("True Label vs Predicted"); ax.legend(title="Predicted")

    plt.tight_layout()
    save(fig, "sa_overview.png")


def viz_scores(df):
    ax_kw = dict(edgecolor="white")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("VADER Component Score Analysis", fontsize=15, fontweight="bold")

    for ax, score, color, label in [
        (axes[0], "pos", POS_COLOR, "Positive Score"),
        (axes[1], "neu", NEU_COLOR, "Neutral Score"),
        (axes[2], "neg", NEG_COLOR, "Negative Score"),
    ]:
        for sent, ls in [("Positive","--"),("Neutral",":"),("Negative","-")]:
            clr = POS_COLOR if sent=="Positive" else NEG_COLOR if sent=="Negative" else NEU_COLOR
            sub = df[df["predicted_label"]==sent][score]
            if len(sub) > 5 and sub.std() > 0.001:
                sub.plot.kde(ax=ax, color=clr, lw=2, ls=ls, label=sent)
            else:
                ax.axvline(sub.mean(), color=clr, lw=2, ls=ls, label=f"{sent} (mean)")
        ax.set_xlabel(label); ax.set_title(f"{label} Distribution")
        ax.legend()

    plt.tight_layout()
    save(fig, "sa_scores.png")


def viz_top_words(df):
    print("[4/5] Top words per sentiment …")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Top 15 Words by Sentiment Category", fontsize=15, fontweight="bold")

    for ax, label, color in [
        (axes[0], "Positive", POS_COLOR),
        (axes[1], "Neutral",  NEU_COLOR),
        (axes[2], "Negative", NEG_COLOR),
    ]:
        top = get_top_words(df, label, 15)
        if top.empty: continue
        bars = ax.barh(top["word"], top["count"], color=color, edgecolor="white")
        ax.invert_yaxis()
        ax.set_title(f"{label} Reviews", fontweight="bold")
        ax.set_xlabel("Frequency")
        for bar in bars:
            ax.text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2,
                    str(int(bar.get_width())), va="center", fontsize=8)

    plt.tight_layout()
    save(fig, "sa_top_words.png")


def viz_wordclouds(df):
    print("[5/5] Generating word clouds …")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Word Clouds by Sentiment", fontsize=15, fontweight="bold")

    for ax, label, bg, color in [
        (axes[0], "Positive", "#e8f8e8", POS_COLOR),
        (axes[1], "Neutral",  "#e8f0ff", NEU_COLOR),
        (axes[2], "Negative", "#fce8e8", NEG_COLOR),
    ]:
        texts = " ".join(df[df["predicted_label"] == label]["review_text"])
        tokens = preprocess(texts)
        if not tokens:
            ax.set_visible(False); continue
        freq = Counter(tokens)
        wc = WordCloud(
            width=600, height=400, background_color=bg,
            color_func=lambda *args, **kwargs: color,
            max_words=80, prefer_horizontal=0.9,
            collocations=False,
        ).generate_from_frequencies(freq)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off"); ax.set_title(f"{label} Reviews", fontweight="bold")

    plt.tight_layout()
    save(fig, "sa_wordclouds.png")


def viz_dashboard(df):
    fig = plt.figure(figsize=(18, 11), facecolor="#F8F9FA")
    fig.suptitle("📊  Sentiment Analysis — Summary Dashboard",
                 fontsize=17, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    counts     = df["predicted_label"].value_counts()
    colors_map = {"Positive": POS_COLOR, "Negative": NEG_COLOR, "Neutral": NEU_COLOR}

    # Pie
    ax = fig.add_subplot(gs[0, 0])
    cols = [colors_map[l] for l in counts.index]
    ax.pie(counts.values, labels=counts.index, autopct="%1.0f%%",
           colors=cols, startangle=140, wedgeprops={"edgecolor":"white"})
    ax.set_title("Sentiment Breakdown")

    # Compound score box
    ax = fig.add_subplot(gs[0, 1])
    data_bp = [df[df["predicted_label"]==l]["compound"].values
               for l in ["Positive","Neutral","Negative"]]
    bp = ax.boxplot(data_bp, patch_artist=True, notch=False,
                    medianprops={"color":"black","lw":2})
    for patch, color in zip(bp["boxes"], [POS_COLOR, NEU_COLOR, NEG_COLOR]):
        patch.set_facecolor(color)
    ax.set_xticklabels(["Positive","Neutral","Negative"])
    ax.set_ylabel("Compound Score"); ax.set_title("Compound Score by Sentiment")
    ax.axhline(0, color="grey", ls="--", alpha=0.5)

    # Avg VADER scores bar
    ax = fig.add_subplot(gs[0, 2])
    avg = df.groupby("predicted_label")[["pos","neu","neg"]].mean()
    avg.plot.bar(ax=ax, color=[POS_COLOR, NEU_COLOR, NEG_COLOR], rot=0, edgecolor="white")
    ax.set_ylabel("Average Score"); ax.set_title("Avg VADER Scores by Sentiment")
    ax.legend(["Positive","Neutral","Negative"])

    # Scatter compound vs pos
    ax = fig.add_subplot(gs[1, 0])
    for label, color in colors_map.items():
        sub = df[df["predicted_label"]==label]
        ax.scatter(sub["compound"], sub["pos"], color=color, alpha=0.4, s=20, label=label)
    ax.set_xlabel("Compound Score"); ax.set_ylabel("Positive Score")
    ax.set_title("Compound vs Positive Score"); ax.legend()

    # Review length vs sentiment
    ax = fig.add_subplot(gs[1, 1])
    df["review_length"] = df["review_text"].apply(len)
    for label, color in colors_map.items():
        sub = df[df["predicted_label"]==label]["review_length"]
        if len(sub) > 5:
            sns.kdeplot(sub, ax=ax, color=color, lw=2, label=label, fill=True, alpha=0.15)
        else:
            ax.axvline(sub.mean(), color=color, lw=2, label=label)
    ax.set_xlabel("Review Length (chars)"); ax.set_title("Review Length by Sentiment")
    ax.legend()

    # Accuracy summary table
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    summary = {
        "Total Reviews"    : len(df),
        "Positive"         : (df["predicted_label"]=="Positive").sum(),
        "Neutral"          : (df["predicted_label"]=="Neutral").sum(),
        "Negative"         : (df["predicted_label"]=="Negative").sum(),
        "Accuracy vs True" : f"{(df['predicted_label']==df['true_label']).mean()*100:.1f}%",
        "Avg Compound"     : f"{df['compound'].mean():.3f}",
    }
    cell_text = [[k, str(v)] for k, v in summary.items()]
    tbl = ax.table(cellText=cell_text, colLabels=["Metric","Value"],
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11)
    tbl.scale(1.2, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2C3E50"); cell.set_text_props(color="white", fontweight="bold")
        elif c == 0:
            cell.set_facecolor("#EBF5FB")
    ax.set_title("Summary Statistics", fontweight="bold", pad=20)

    path = os.path.join(OUTPUT_DIR, "sa_dashboard.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  ✔ Saved: {path}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  CodeAlpha — Task 4: Sentiment Analysis")
    print("=" * 60)

    print("\n[1/5] Building sample review dataset …")
    df = build_dataset(n=300)
    print(f"  Total reviews: {len(df)}")
    print(f"  Label distribution:\n{df['true_label'].value_counts().to_string()}")

    df = run_vader(df)
    df.to_csv(os.path.join(OUTPUT_DIR, "sentiment_results.csv"))
    print(f"\n  Predicted distribution:\n{df['predicted_label'].value_counts().to_string()}")

    print("\n  Sample predictions:")
    sample = df[["review_text","compound","predicted_label"]].sample(6, random_state=1)
    for _, row in sample.iterrows():
        short = row.review_text[:65] + "…"
        print(f"    [{row.predicted_label:8s}] (score={row.compound:+.3f}) {short}")

    viz_overview(df)
    viz_scores(df)
    viz_top_words(df)
    viz_wordclouds(df)
    viz_dashboard(df)

    print("\n" + "="*60)
    print("  ✅ Sentiment analysis complete — outputs in output/")
    print("="*60)
