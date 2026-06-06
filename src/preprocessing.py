import re
import pandas as pd


def clean_text(text: str) -> str:
    """
    Cleans raw review text by:
    - Lowercasing
    - Removing HTML tags (if any)
    - Normalizing whitespaces
    - Stripping trailing/leading space
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]*>", " ", text)

    # Normalize whitespaces (tabs, newlines, multiple spaces to single space)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_features(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Extracts structural and metadata features from review text using Pandas .assign()
    and vectorized operations:
    - char_count: Total characters
    - word_count: Total words
    - avg_word_length: Average length of words
    - cap_ratio: Ratio of uppercase characters to total characters
    - exclamation_ratio: Ratio of exclamation marks to total characters
    - question_ratio: Ratio of question marks to total characters
    """
    # Cast to string series once for vector safety
    text_series = df[text_col].astype(str)

    df_features = df.assign(
        char_count=text_series.str.len(),
        word_count=text_series.str.split().str.len(),
        avg_word_length=lambda d: d["char_count"] / d["word_count"].replace(0, 1),
        cap_ratio=lambda d: text_series.str.findall(r"[A-Z]").str.len() / d["char_count"].replace(0, 1),
        exclamation_ratio=lambda d: text_series.str.count("!") / d["char_count"].replace(0, 1),
        question_ratio=lambda d: text_series.str.count(r"\?") / d["char_count"].replace(0, 1),
    )
    return df_features


def preprocess_pipeline(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Main entry point for preprocessing using Pandas method chaining:
    1. Extracts structural metadata features from raw text using .pipe()
    2. Cleans the text column using .assign()
    """
    df_preprocessed = df.pipe(extract_features, text_col=text_col).assign(
        clean_text=lambda d: d[text_col].apply(clean_text)
    )
    return df_preprocessed
