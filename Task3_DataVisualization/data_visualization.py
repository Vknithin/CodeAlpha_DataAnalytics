"""
============================================================
CODEALPHA INTERNSHIP — TASK 3: DATA VISUALIZATION
============================================================
Author     : [Your Name]
Dataset    : books_cleaned.csv  (from Task 2)
Libraries  : matplotlib, seaborn, pandas, numpy
Description: Creates 8 publication-quality charts including
             interactive-style dashboards, heatmaps, violin
             plots, scatter plots with regression lines, and
             a story-telling final summary dashboard.
============================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────
DATA_PATH  = "../Task2_EDA/output/books_cleaned.csv"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Global style ───────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({
    "figure.dpi"       : 150,
    "font.family"      : "DejaVu Sans",
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 11,
    "legend.fontsize"  : 9,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

BLUE   = "#4C72B0"
GREEN  = "#55A868"
ORANGE = "#DD8452"
RED    = "#C44E52"


# ── Helper ─────────────────────────────────────────────────
def save(fig, name: str) -> None:
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✔ Saved: {path}")


# ============================================================
# VISUALISATION 1 — Price Distribution Deep-Dive
# ============================================================
def viz_price_deep_dive(df):
    print("\n[1/6] Price distribution deep-dive …")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Price Distribution Deep-Dive", fontsize=15, fontweight="bold", y=1.02)

    # KDE + histogram
    ax = axes[0]
    sns.histplot(df.price_gbp, bins=30, kde=True, ax=ax, color=BLUE,
                 edgecolor="white", line_kws={"lw": 2})
    ax.axvline(df.price_gbp.mean(),  color=RED,    ls="--", lw=1.5, label=f"Mean £{df.price_gbp.mean():.2f}")
    ax.axvline(df.price_gbp.median(),color=ORANGE, ls="--", lw=1.5, label=f"Median £{df.price_gbp.median():.2f}")
    ax.set_title("Price Histogram with KDE"); ax.set_xlabel("Price (£)"); ax.legend()

    # Violin by rating
    ax = axes[1]
    sns.violinplot(data=df, x="star_rating", y="price_gbp", ax=ax,
                   palette="YlOrRd", inner="quartile", cut=0)
    ax.set_title("Price Distribution by Rating")
    ax.set_xlabel("Star Rating"); ax.set_ylabel("Price (£)")

    # Box per genre (sorted by median)
    ax = axes[2]
    order = df.groupby("genre")["price_gbp"].median().sort_values().index
    sns.boxplot(data=df, y="genre", x="price_gbp", order=order, ax=ax,
                palette="Blues", flierprops={"marker": ".", "ms": 4})
    ax.set_title("Price Range by Genre")
    ax.set_xlabel("Price (£)"); ax.set_ylabel("")

    plt.tight_layout()
    save(fig, "viz1_price_distribution.png")


# ============================================================
# VISUALISATION 2 — Genre Performance Heatmap
# ============================================================
def viz_genre_heatmap(df):
    print("[2/6] Genre performance heatmap …")
    pivot = df.pivot_table(
        index="genre",
        values=["price_gbp","star_rating","in_stock"],
        aggfunc={"price_gbp":"mean","star_rating":"mean","in_stock":"mean"}
    ).round(2)
    pivot.columns = ["Avg Price (£)","Avg Rating","In-Stock %"]
    pivot["In-Stock %"] = (pivot["In-Stock %"] * 100).round(1)
    pivot_norm = (pivot - pivot.min()) / (pivot.max() - pivot.min())   # normalise 0–1

    fig, ax = plt.subplots(figsize=(10, 7))
    mask = np.zeros_like(pivot_norm, dtype=bool)
    sns.heatmap(pivot_norm, annot=pivot.values, fmt="", ax=ax,
                cmap="YlGnBu", linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Normalised Score"},
                annot_kws={"size": 10})
    ax.set_title("Genre Performance Heatmap\n(annotated with actual values, coloured by normalised score)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel(""); ax.set_ylabel("Genre")
    ax.set_xticklabels(pivot.columns, rotation=0)
    plt.tight_layout()
    save(fig, "viz2_genre_heatmap.png")


# ============================================================
# VISUALISATION 3 — Rating & Stock Analysis
# ============================================================
def viz_rating_stock(df):
    print("[3/6] Rating & stock analysis …")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Rating & Availability Analysis", fontsize=15, fontweight="bold")

    # Stacked bar — stock by rating
    ax = axes[0]
    ct = df.groupby(["star_rating","availability"]).size().unstack(fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct.plot.bar(stacked=True, ax=ax, color=[RED, GREEN], edgecolor="white", rot=0)
    ax.set_title("Availability % by Star Rating")
    ax.set_xlabel("Star Rating"); ax.set_ylabel("Percentage (%)")
    ax.legend(title="Availability", loc="upper left")
    for bar_container in ax.containers:
        ax.bar_label(bar_container, fmt="%.1f%%", label_type="center", fontsize=8, color="white")

    # Scatter — price vs rating with trend
    ax = axes[1]
    colors_map = {1: RED, 2: ORANGE, 3: "gold", 4: GREEN, 5: BLUE}
    for rating, grp in df.groupby("star_rating"):
        ax.scatter(grp.price_gbp, grp.star_rating + np.random.uniform(-0.15,0.15,len(grp)),
                   color=colors_map[rating], alpha=0.5, s=20, label=f"{rating}★")
    # Regression line
    m, b = np.polyfit(df.price_gbp, df.star_rating, 1)
    xs = np.linspace(df.price_gbp.min(), df.price_gbp.max(), 100)
    ax.plot(xs, m*xs+b, color="black", lw=1.5, ls="--", label=f"Trend (slope={m:.4f})")
    ax.set_title("Price vs Star Rating (with Regression)")
    ax.set_xlabel("Price (£)"); ax.set_ylabel("Star Rating")
    ax.legend(title="Rating", fontsize=8, ncol=2)

    plt.tight_layout()
    save(fig, "viz3_rating_stock.png")


# ============================================================
# VISUALISATION 4 — Top / Bottom 10 Genres
# ============================================================
def viz_top_genres(df):
    print("[4/6] Genre comparison bars …")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Genre Comparison: Price vs Rating", fontsize=15, fontweight="bold")

    genre_stats = df.groupby("genre").agg(
        avg_price  = ("price_gbp","mean"),
        avg_rating = ("star_rating","mean"),
        count      = ("title","count"),
    ).sort_values("avg_price", ascending=False)

    # Avg price horizontal bar
    ax = axes[0]
    colors = sns.color_palette("RdYlGn_r", len(genre_stats))
    bars = ax.barh(genre_stats.index, genre_stats.avg_price,
                   color=colors, edgecolor="white")
    ax.set_xlabel("Average Price (£)"); ax.set_title("Average Price by Genre")
    for bar in bars:
        ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                f"£{bar.get_width():.1f}", va="center", fontsize=8)

    # Lollipop chart — avg rating
    ax = axes[1]
    genre_stats_r = genre_stats.sort_values("avg_rating", ascending=False)
    ax.hlines(y=genre_stats_r.index, xmin=0, xmax=genre_stats_r.avg_rating,
              color="grey", alpha=0.6, lw=2)
    ax.plot(genre_stats_r.avg_rating, genre_stats_r.index, "o",
            color=BLUE, ms=9, zorder=3)
    for y, (_, row) in zip(genre_stats_r.index, genre_stats_r.iterrows()):
        ax.text(row.avg_rating+0.03, y, f"{row.avg_rating:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 5.3); ax.set_xlabel("Average Star Rating")
    ax.set_title("Average Rating by Genre")

    plt.tight_layout()
    save(fig, "viz4_genre_comparison.png")


# ============================================================
# VISUALISATION 5 — Correlation & Pair Plot
# ============================================================
def viz_correlation(df):
    print("[5/6] Correlation analysis …")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Correlation Analysis", fontsize=15, fontweight="bold")

    # Heatmap
    ax = axes[0]
    corr = df[["price_gbp","star_rating","in_stock"]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".3f", ax=ax, cmap="coolwarm",
                linewidths=1, square=True, vmin=-1, vmax=1,
                cbar_kws={"shrink": 0.7})
    ax.set_title("Pearson Correlation Matrix")

    # Scatter matrix (manual)
    ax = axes[1]
    sns.scatterplot(data=df, x="price_gbp", y="star_rating", hue="in_stock",
                    palette={0: RED, 1: GREEN}, alpha=0.5, s=20, ax=ax)
    sns.regplot(data=df, x="price_gbp", y="star_rating", scatter=False,
                ax=ax, color="black", line_kws={"lw":1.5,"ls":"--"})
    ax.set_title("Price vs Rating (coloured by Stock)")
    ax.set_xlabel("Price (£)"); ax.set_ylabel("Star Rating")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["Out of Stock","In Stock"], title="Stock")

    plt.tight_layout()
    save(fig, "viz5_correlation.png")


# ============================================================
# VISUALISATION 6 — Story-Telling Dashboard
# ============================================================
def viz_dashboard(df):
    print("[6/6] Final summary dashboard …")
    fig = plt.figure(figsize=(18, 12), facecolor="#F8F9FA")
    fig.suptitle(
        "📚  Books Catalogue — Data Story Dashboard",
        fontsize=18, fontweight="bold", y=0.98, color="#2C3E50"
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── KPI cards ──────────────────────────────────────────
    kpis = [
        (f"{len(df):,}",         "Total Books",        BLUE),
        (f"£{df.price_gbp.mean():.2f}", "Avg Price",   GREEN),
        (f"{df.star_rating.mean():.2f}★","Avg Rating", ORANGE),
        (f"{df.in_stock.mean()*100:.0f}%","In Stock",  "#9B59B6"),
    ]
    for i, (val, label, color) in enumerate(kpis):
        ax = fig.add_axes([0.02 + i * 0.24, 0.76, 0.20, 0.14])
        ax.set_facecolor(color); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.text(0.5, 0.55, val,   transform=ax.transAxes, ha="center",
                va="center", fontsize=22, fontweight="bold", color="white")
        ax.text(0.5, 0.18, label, transform=ax.transAxes, ha="center",
                va="center", fontsize=11, color="white", alpha=0.9)

    # ── Genre donut ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    genre_counts = df.genre.value_counts()
    wedges, texts, autotexts = ax2.pie(
        genre_counts.values, labels=None,
        autopct="%1.0f%%", startangle=90,
        colors=sns.color_palette("Set3", len(genre_counts)),
        pctdistance=0.8, wedgeprops={"width":0.55}
    )
    for at in autotexts: at.set_fontsize(7)
    ax2.legend(genre_counts.index, loc="center", fontsize=7,
               bbox_to_anchor=(0.5, -0.08), ncol=2)
    ax2.set_title("Books by Genre", fontweight="bold")

    # ── Rating bar ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    rc = df.star_rating.value_counts().sort_index()
    bars = ax3.bar(rc.index, rc.values,
                   color=sns.color_palette("YlOrRd", 5), edgecolor="white", width=0.65)
    for bar in bars:
        ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                 str(int(bar.get_height())), ha="center", fontsize=9)
    ax3.set_xlabel("Stars"); ax3.set_ylabel("Count")
    ax3.set_title("Rating Distribution", fontweight="bold")
    ax3.set_xticks([1,2,3,4,5])

    # ── Price KDE by stock ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    for label, grp, color in [("In Stock", df[df.in_stock==1], GREEN),
                               ("Out of Stock", df[df.in_stock==0], RED)]:
        grp.price_gbp.plot.kde(ax=ax4, label=label, color=color, lw=2)
    ax4.set_xlabel("Price (£)"); ax4.set_title("Price KDE: Stock Status", fontweight="bold")
    ax4.legend()

    # ── Genre avg price + rating bubble ───────────────────
    ax5 = fig.add_subplot(gs[2, :])
    gs2 = df.groupby("genre").agg(
        avg_price  = ("price_gbp","mean"),
        avg_rating = ("star_rating","mean"),
        count      = ("title","count"),
    ).reset_index()
    scatter = ax5.scatter(
        gs2.avg_price, gs2.avg_rating,
        s=gs2["count"] * 4, alpha=0.7,
        c=gs2.avg_price, cmap="coolwarm", edgecolors="white", lw=0.5
    )
    for _, row in gs2.iterrows():
        ax5.annotate(row.genre, (row.avg_price, row.avg_rating),
                     textcoords="offset points", xytext=(6, 4), fontsize=9)
    cb = plt.colorbar(scatter, ax=ax5, pad=0.01)
    cb.set_label("Avg Price (£)", fontsize=9)
    ax5.set_xlabel("Average Price (£)"); ax5.set_ylabel("Average Star Rating")
    ax5.set_title("Genre Bubble Chart: Price vs Rating  (bubble size = book count)",
                  fontweight="bold")

    save(fig, "viz6_dashboard.png")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  CodeAlpha — Task 3: Data Visualization")
    print("=" * 60)
    df = pd.read_csv(DATA_PATH, index_col="book_id")
    print(f"  Loaded {len(df):,} rows from {DATA_PATH}\n")

    viz_price_deep_dive(df)
    viz_genre_heatmap(df)
    viz_rating_stock(df)
    viz_top_genres(df)
    viz_correlation(df)
    viz_dashboard(df)

    print("\n" + "=" * 60)
    print("  ✅ All 6 visualisations saved to output/")
    print("=" * 60)
