#Naser Abdullah Alam. (2024). Phishing Email Dataset [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DS/5074342

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# =============================================================================
# Config
# =============================================================================
RANDOM_STATE = 42
FIG_DIR = Path("figures_ceas08")
FIG_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")


# =============================================================================
# Loading
# =============================================================================
def smart_read_email_dataset(path: str | Path) -> pd.DataFrame:
    """
    Robust reader for Kaggle CEAS_08.csv.

    Why special handling?
    - 'body' often contains embedded newlines inside quoted strings
    - some rows may be slightly malformed
    """
    path = Path(path)

    # 1) Fast attempt (C engine). Often works.
    try:
        df = pd.read_csv(
            path,
            sep=",",
            quotechar='"',
            engine="c",
            low_memory=False,
        )
    except Exception:
        # 2) More tolerant fallback (python engine)
        df = pd.read_csv(
            path,
            sep=",",
            quotechar='"',
            engine="python",
            on_bad_lines="skip",
        )

    # Normalize col names
    df.columns = [c.strip().lower() for c in df.columns]

    expected = {"sender", "receiver", "date", "subject", "body", "label", "urls"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}\n"
            f"Found columns: {list(df.columns)}\n"
            "If Kaggle version differs, update expected column names accordingly."
        )

    return df


# =============================================================================
# Feature Engineering / Cleaning
# =============================================================================
def safe_to_datetime(series: pd.Series) -> pd.Series:
    # Example: "Tue, 05 Aug 2008 16:31:02 -0700"
    return pd.to_datetime(series, errors="coerce", utc=True)


def normalize_text(s: str) -> str:
    """
    Simple text normalization for visualization/word clouds.
    - lowercase
    - strip URLs + emails
    - remove punctuation
    - collapse whitespace
    """
    if not isinstance(s, str):
        return ""

    s = s.lower()

    # URLs
    s = re.sub(r"http[s]?://\S+|www\.\S+", " ", s)

    # email addresses
    s = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", s)

    # keep alnum + spaces
    s = re.sub(r"[^a-z0-9\s]+", " ", s)

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_email_from_sender(sender: str) -> str:
    """
    sender examples:
    - "Young Esposito <Young@iworld.de>"
    - "Mok <ipline's1983@icable.ph>"
    - "Daily Top 10 <Karmandeep-opengevl@universalnet.psi.br>"
    - sometimes just an email, sometimes just a name
    """
    if not isinstance(sender, str):
        return ""

    # Common pattern: <email@domain>
    m = re.search(r"<\s*([^<>@\s]+@[^<>@\s]+)\s*>", sender)
    if m:
        return m.group(1).strip().lower()

    # Otherwise, any email-like token in the string
    m2 = re.search(r"\b([\w\.-]+@[\w\.-]+\.\w+)\b", sender)
    if m2:
        return m2.group(1).strip().lower()

    return ""


def extract_domain(email: str) -> str:
    if not isinstance(email, str) or "@" not in email:
        return ""
    return email.split("@", 1)[1].lower()


def build_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Dates
    df["date_parsed"] = safe_to_datetime(df["date"])

    # Fill NA for text fields
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")

    # Raw and clean text
    df["text_raw"] = (df["subject"].astype(str) + " " + df["body"].astype(str)).str.strip()
    df["text_clean"] = df["text_raw"].map(normalize_text)

    # Simple lengths
    df["subject_len"] = df["subject"].astype(str).str.len()
    df["body_len"] = df["body"].astype(str).str.len()
    df["text_len"] = df["text_raw"].astype(str).str.len()

    # URLs
    df["urls"] = pd.to_numeric(df["urls"], errors="coerce")
    df["has_url_in_text"] = df["text_raw"].str.contains(
        r"http[s]?://|www\.", regex=True, na=False
    ).astype(int)

    # Time features
    df["hour_utc"] = df["date_parsed"].dt.hour
    df["weekday_utc"] = df["date_parsed"].dt.day_name()

    # Label
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Int64")

    # Sender email + domain
    df["sender_email"] = df["sender"].map(extract_email_from_sender)
    df["sender_domain"] = df["sender_email"].map(extract_domain)

    # Receiver domain (sometimes helpful)
    df["receiver"] = df["receiver"].fillna("")
    df["receiver_domain"] = df["receiver"].astype(str).str.extract(r"@(.+)$")[0].fillna("").str.lower()

    return df


# =============================================================================
# Plot helpers
# =============================================================================
def savefig(name: str) -> None:
    out = FIG_DIR / name
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    print(f"[saved] {out}")
    plt.close()


# =============================================================================
# EDA Plots (baseline)
# =============================================================================
def plot_label_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(data=df, x="label")
    ax.set_title("Email Class Distribution (label)")
    ax.set_xlabel("Label (0=legit, 1=phishing/spam)")
    ax.set_ylabel("Count")
    savefig("01_label_distribution.png")


def plot_urls_vs_label(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    sns.countplot(data=df, x="urls", hue="label", ax=axes[0])
    axes[0].set_title("Provided 'urls' Column vs Label")
    axes[0].set_xlabel("urls (as provided)")
    axes[0].set_ylabel("count")

    sns.countplot(data=df, x="has_url_in_text", hue="label", ax=axes[1])
    axes[1].set_title("Detected URL in Subject/Body vs Label")
    axes[1].set_xlabel("has_url_in_text")
    axes[1].set_ylabel("count")

    savefig("02_urls_vs_label.png")


def plot_text_length_distributions(df: pd.DataFrame) -> None:
    # clip long tail for nicer histogram view
    text_len = df["text_len"].dropna()
    clip_max = np.percentile(text_len, 99) if len(text_len) else 0

    plt.figure(figsize=(8, 4))
    sns.histplot(
        data=df,
        x=np.clip(df["text_len"], 0, clip_max),
        hue="label",
        bins=50,
        kde=True,
    )
    plt.title("Text Length Distribution (Subject+Body) by Label (clipped at 99th pct)")
    plt.xlabel("Text length (characters)")
    plt.ylabel("count")
    savefig("03_text_length_hist.png")

    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x="label", y="text_len")
    plt.yscale("log")
    plt.title("Text Length by Label (log scale)")
    plt.xlabel("Label")
    plt.ylabel("Text length (characters, log)")
    savefig("04_text_length_box_log.png")


def plot_time_patterns(df: pd.DataFrame) -> None:
    if df["date_parsed"].notna().sum() < 10:
        print("[skip] Not enough parsable dates for time plots.")
        return

    plt.figure(figsize=(8, 4))
    sns.countplot(
        data=df.dropna(subset=["weekday_utc"]),
        x="weekday_utc",
        hue="label",
        order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    )
    plt.title("Emails by Weekday (UTC) and Label")
    plt.xlabel("Weekday (UTC)")
    plt.ylabel("count")
    plt.xticks(rotation=25)
    savefig("05_weekday_vs_label.png")

    plt.figure(figsize=(8, 4))
    sns.histplot(
        data=df.dropna(subset=["hour_utc"]),
        x="hour_utc",
        hue="label",
        bins=24,
        multiple="stack",
    )
    plt.title("Emails by Hour of Day (UTC) and Label")
    plt.xlabel("Hour (UTC)")
    plt.ylabel("count")
    savefig("06_hour_vs_label.png")


# =============================================================================
# Additional Plots
# =============================================================================
def plot_subject_len_kde(df: pd.DataFrame) -> None:
    """
    KDE of subject length by label.
    Some datasets have lots of very short subjects; clipping helps readability.
    """
    subj = df["subject_len"].dropna()
    if subj.empty:
        print("[skip] No subject length data.")
        return

    clip_max = np.percentile(subj, 99)
    tmp = df.copy()
    tmp["subject_len_clip"] = np.clip(tmp["subject_len"], 0, clip_max)

    plt.figure(figsize=(8, 4))
    sns.kdeplot(
        data=tmp.dropna(subset=["subject_len_clip", "label"]),
        x="subject_len_clip",
        hue="label",
        common_norm=False,
        fill=True,
    )
    plt.title("Subject Length KDE by Label (clipped at 99th pct)")
    plt.xlabel("Subject length (characters)")
    plt.ylabel("density")
    savefig("07_subject_length_kde.png")


def plot_subject_vs_body_scatter(df: pd.DataFrame) -> None:
    """
    Relationship between subject and body length (scatter).
    We sample for speed if dataset is huge.
    """
    tmp = df.dropna(subset=["subject_len", "body_len", "label"]).copy()

    if len(tmp) == 0:
        print("[skip] No data for subject/body scatter.")
        return

    # Sample for speed / plot clarity
    if len(tmp) > 5000:
        tmp = tmp.sample(5000, random_state=RANDOM_STATE)

    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=tmp, x="subject_len", y="body_len", hue="label", alpha=0.4)
    plt.yscale("log")
    plt.title("Subject Length vs Body Length (body log-scale)")
    plt.xlabel("Subject length (characters)")
    plt.ylabel("Body length (characters, log)")
    savefig("08_subject_vs_body_scatter.png")


def plot_sender_domain_frequency(df: pd.DataFrame, top_n: int = 20) -> None:
    """
    Top sender domains overall + by label.
    """
    tmp = df.copy()
    tmp = tmp[tmp["sender_domain"].astype(str).str.len() > 0]

    if tmp.empty:
        print("[skip] No sender domains extracted (sender_email may be missing).")
        return

    # Top domains overall
    top_domains = tmp["sender_domain"].value_counts().head(top_n).index.tolist()
    tmp_top = tmp[tmp["sender_domain"].isin(top_domains)]

    plt.figure(figsize=(10, 6))
    sns.countplot(
        data=tmp_top,
        y="sender_domain",
        order=tmp_top["sender_domain"].value_counts().index,
        hue="label",
    )
    plt.title(f"Top {top_n} Sender Domains (by Label)")
    plt.xlabel("count")
    plt.ylabel("sender_domain")
    savefig("09_sender_domain_topN_by_label.png")


def plot_receiver_domain_frequency(df: pd.DataFrame, top_n: int = 15) -> None:
    """
    Useful sanity plot: top receiver domains (often includes dataset domains).
    """
    tmp = df.copy()
    tmp = tmp[tmp["receiver_domain"].astype(str).str.len() > 0]

    if tmp.empty:
        print("[skip] No receiver domains extracted.")
        return

    top_domains = tmp["receiver_domain"].value_counts().head(top_n).index.tolist()
    tmp_top = tmp[tmp["receiver_domain"].isin(top_domains)]

    plt.figure(figsize=(10, 6))
    sns.countplot(
        data=tmp_top,
        y="receiver_domain",
        order=tmp_top["receiver_domain"].value_counts().index,
        hue="label",
    )
    plt.title(f"Top {top_n} Receiver Domains (by Label)")
    plt.xlabel("count")
    plt.ylabel("receiver_domain")
    savefig("10_receiver_domain_topN_by_label.png")


def plot_url_rate_by_sender_domain(df: pd.DataFrame, top_n: int = 20) -> None:
    """
    For top sender domains, show fraction of emails that contain URLs (detected in text).
    """
    tmp = df.copy()
    tmp = tmp[tmp["sender_domain"].astype(str).str.len() > 0]
    if tmp.empty:
        print("[skip] No sender domains for URL rate plot.")
        return

    top_domains = tmp["sender_domain"].value_counts().head(top_n).index.tolist()
    tmp = tmp[tmp["sender_domain"].isin(top_domains)]

    grp = tmp.groupby("sender_domain", as_index=False)["has_url_in_text"].mean()
    grp = grp.sort_values("has_url_in_text", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=grp, y="sender_domain", x="has_url_in_text")
    plt.title(f"URL Presence Rate by Sender Domain (Top {top_n})")
    plt.xlabel("Mean(has_url_in_text)")
    plt.ylabel("sender_domain")
    savefig("11_url_rate_by_sender_domain.png")


# =============================================================================
# Wordclouds + Token Counts
# =============================================================================
def make_wordcloud(text: str, title: str, outfile: str) -> None:
    if not text.strip():
        print(f"[skip] Empty text for wordcloud: {title}")
        return

    wc = WordCloud(
        width=1200,
        height=600,
        background_color="white",
        stopwords=set(ENGLISH_STOP_WORDS),
        max_words=200,
    ).generate(text)

    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    savefig(outfile)


def wordclouds_by_label(df: pd.DataFrame) -> None:
    legit = " ".join(df.loc[df["label"] == 0, "text_clean"].dropna().astype(str).tolist())
    phish = " ".join(df.loc[df["label"] == 1, "text_clean"].dropna().astype(str).tolist())

    make_wordcloud(legit, "WordCloud: Legitimate Emails (label=0)", "12_wordcloud_legit.png")
    make_wordcloud(phish, "WordCloud: Phishing/Spam Emails (label=1)", "13_wordcloud_phishing.png")


def top_terms_quick(df: pd.DataFrame, n: int = 20) -> None:
    """
    Quick token frequency comparison per label (no heavy NLP).
    """
    def token_counts(texts: pd.Series) -> pd.Series:
        words = " ".join(texts.dropna().astype(str)).split()
        words = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]
        return pd.Series(words).value_counts()

    legit_counts = token_counts(df.loc[df["label"] == 0, "text_clean"])
    phish_counts = token_counts(df.loc[df["label"] == 1, "text_clean"])

    print("\nTop terms (legit):")
    print(legit_counts.head(n))

    print("\nTop terms (phishing/spam):")
    print(phish_counts.head(n))


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    # Put CEAS_08.csv in the same folder as this script OR provide full path here.
    DATA_PATH = "CEAS_08.csv"

    df = smart_read_email_dataset(DATA_PATH)
    df = build_text_fields(df)

    # -------------------------
    # Basic exploration
    # -------------------------
    print("\n=== Dataset loaded ===")
    print("shape:", df.shape)

    print("\n=== Head (3 rows) ===")
    print(df.head(3))

    print("\n=== Info ===")
    print(df.info())

    print("\n=== Missing values (top 15) ===")
    print(df.isna().sum().sort_values(ascending=False).head(15))

    print("\n=== Label counts ===")
    print(df["label"].value_counts(dropna=False))

    # -------------------------
    # Plots (baseline)
    # -------------------------
    plot_label_distribution(df)
    plot_urls_vs_label(df)
    plot_text_length_distributions(df)
    plot_time_patterns(df)

    # -------------------------
    # Additional plots
    # -------------------------
    plot_subject_len_kde(df)
    plot_subject_vs_body_scatter(df)
    plot_sender_domain_frequency(df, top_n=20)
    plot_receiver_domain_frequency(df, top_n=15)
    plot_url_rate_by_sender_domain(df, top_n=20)

    # -------------------------
    # Wordclouds + quick terms
    # -------------------------
    wordclouds_by_label(df)
    top_terms_quick(df, n=20)

    # -------------------------
    # Save a cleaned export
    # -------------------------
    out_csv = "CEAS_08_cleaned.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n[saved] {out_csv}")
    print(f"[done] Figures saved in: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
