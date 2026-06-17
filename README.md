# CodeAlpha Data Analytics Internship

> **Repository:** `CodeAlpha_DataAnalytics`  
> **Domain:** Data Analytics  
> **Tasks Completed:** 4 / 4

---

## 📁 Project Structure

```
CodeAlpha_DataAnalytics/
│
├── Task1_WebScraping/
│   ├── web_scraping.py          ← Main scraping script
│   └── output/
│       └── books_dataset.csv    ← Scraped / generated dataset
│
├── Task2_EDA/
│   ├── eda_analysis.py          ← Full EDA pipeline
│   └── output/
│       ├── books_cleaned.csv    ← Cleaned dataset
│       ├── eda_univariate.png
│       ├── eda_bivariate.png
│       └── eda_outliers.png
│
├── Task3_DataVisualization/
│   ├── data_visualization.py    ← 6 publication-quality charts
│   └── output/
│       ├── viz1_price_distribution.png
│       ├── viz2_genre_heatmap.png
│       ├── viz3_rating_stock.png
│       ├── viz4_genre_comparison.png
│       ├── viz5_correlation.png
│       └── viz6_dashboard.png   ← Story-telling dashboard
│
├── Task4_SentimentAnalysis/
│   ├── sentiment_analysis.py    ← VADER NLP pipeline
│   └── output/
│       ├── sentiment_results.csv
│       ├── sa_overview.png
│       ├── sa_scores.png
│       ├── sa_top_words.png
│       ├── sa_wordclouds.png
│       └── sa_dashboard.png
│
├── requirements.txt
└── README.md                    ← You are here
```

---

## ✅ Task 1 — Web Scraping

**Script:** `Task1_WebScraping/web_scraping.py`

- Scrapes **books.toscrape.com** using `requests` + `BeautifulSoup`
- Iterates through all catalogue pages automatically
- Extracts: **title, price (£), star rating, availability**
- Saves a clean CSV dataset ready for downstream analysis
- Includes a fallback synthetic dataset generator (1,000 books)

**Run:**
```bash
cd Task1_WebScraping
python web_scraping.py
```

---

## ✅ Task 2 — Exploratory Data Analysis (EDA)

**Script:** `Task2_EDA/eda_analysis.py`

| Step | What it does |
|------|-------------|
| Load & Inspect | Shape, dtypes, null values |
| Clean | Deduplication, type conversion, feature engineering |
| Descriptive Stats | Mean, std, skewness, kurtosis, quantiles |
| Univariate | Price histogram, rating bar, genre pie, availability |
| Bivariate | Price × rating box, genre price bar, stock × rating |
| Outlier Detection | IQR method on price |
| Hypothesis Test | t-test: do 5-star books cost more than 1-star? |
| Key Findings | 8 auto-generated insights from the data |

**Run:**
```bash
cd Task2_EDA
python eda_analysis.py
```

---

## ✅ Task 3 — Data Visualization

**Script:** `Task3_DataVisualization/data_visualization.py`

| Chart | Description |
|-------|-------------|
| `viz1` | Price histogram + KDE, violin by rating, boxplot by genre |
| `viz2` | Genre performance heatmap (price, rating, stock) |
| `viz3` | Availability stacked bar, scatter with regression |
| `viz4` | Avg price horizontal bar, avg rating lollipop |
| `viz5` | Correlation matrix heatmap, price vs rating scatter |
| `viz6` | **Full story-telling dashboard** with KPI cards & bubble chart |

**Run:**
```bash
cd Task3_DataVisualization
python data_visualization.py
```

---

## ✅ Task 4 — Sentiment Analysis

**Script:** `Task4_SentimentAnalysis/sentiment_analysis.py`

- **Dataset:** 300 Amazon-style book reviews (Positive / Neutral / Negative)
- **NLP Tool:** VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Accuracy:** ~67% vs human labels
- **Outputs:**
  - Sentiment classification + compound scores for each review
  - Distribution charts, score analysis
  - Top-15 words per sentiment category
  - Word clouds for each sentiment
  - Full summary dashboard

**Run:**
```bash
cd Task4_SentimentAnalysis
python sentiment_analysis.py
```

---

## 🛠 Setup & Requirements

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
requests
beautifulsoup4
pandas
numpy
matplotlib
seaborn
scipy
nltk
wordcloud
scikit-learn
```

After installing, download NLTK data once:
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
```

---

## 📌 Key Insights (cross-task)

1. The books dataset spans **12 genres** with **634 unique titles**
2. Average price is **£35.42** — uniformly distributed, no strong skew
3. **81.2%** of books are in stock
4. **64%** of books are rated 4–5 stars, showing a positive skew in ratings
5. No statistically significant price difference between high- and low-rated books (p = 0.46)
6. VADER achieves **67% accuracy** on book reviews — a solid baseline for lexicon-based NLP
7. Neutral sentiment is the hardest class to detect (VADER tends toward Pos/Neg)

---

*Internship: CodeAlpha · Domain: Data Analytics · Tools: Python, BeautifulSoup, pandas, matplotlib, seaborn, NLTK*
