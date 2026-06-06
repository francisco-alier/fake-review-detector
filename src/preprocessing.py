import re
import string
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
    Extracts structural and metadata features from review text:
    - char_count: Total characters
    - word_count: Total words
    - avg_word_length: Average length of words
    - cap_ratio: Ratio of uppercase characters to total characters (captures shouting/dramatic style)
    - exclamation_ratio: Ratio of exclamation marks to total characters
    - question_ratio: Ratio of question marks to total characters
    """
    # Create a copy to prevent SettingWithCopyWarning
    df = df.copy()

    # Apply character count
    df["char_count"] = df[text_col].apply(lambda x: len(str(x)))

    # Apply word count
    df["word_count"] = df[text_col].apply(lambda x: len(str(x).split()))

    # Avg word length
    df["avg_word_length"] = df.apply(
        lambda row: (row["char_count"] / row["word_count"]) if row["word_count"] > 0 else 0,
        axis=1,
    )

    # Capital ratio (on raw text before cleaning/lowercasing)
    df["cap_ratio"] = df[text_col].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
    )

    # Exclamation mark ratio
    df["exclamation_ratio"] = df[text_col].apply(
        lambda x: str(x).count("!") / max(len(str(x)), 1)
    )

    # Question mark ratio
    df["question_ratio"] = df[text_col].apply(
        lambda x: str(x).count("?") / max(len(str(x)), 1)
    )

    return df


def preprocess_pipeline(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Main entry point for preprocessing:
    1. Extracts structural metadata features from raw text.
    2. Cleans the text column (lowercases, removes HTML, normalizes spaces).
    """
    # Extract features first (while case/punctuation are still intact)
    df_featured = extract_features(df, text_col=text_col)

    # Clean the text column
    df_featured["clean_text"] = df_featured[text_col].apply(clean_text)

    return df_featured
