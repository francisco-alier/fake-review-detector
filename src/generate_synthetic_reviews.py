import json
import logging
import os
import random
import time
from pathlib import Path
import pandas as pd
import requests

from src.utils import load_config

logger = logging.getLogger("generator")

# Pre-defined templates for offline/mock generation
MOCK_TEMPLATES = {
    "positive": [
        "Absolutely loved the food at {restaurant}! The {dish} was cooked to perfection and the service was top-notch. Highly recommend!",
        "Best dining experience in a long time. {restaurant} exceeded all expectations. We ordered the {dish} and it was incredible.",
        "A hidden gem! The ambiance at {restaurant} was lovely, and the staff was extremely friendly. The {dish} is a must-try.",
        "Delicious food and great atmosphere. We will definitely be back to {restaurant}. The {dish} was spectacular!",
        "Superb service and outstanding flavors. {restaurant} knows how to do hospitality. Best {dish} in town by far."
    ],
    "negative": [
        "Horrible service. We waited over an hour for our {dish} at {restaurant} and it arrived cold. Never going back.",
        "Terrible experience. The staff at {restaurant} was rude and the {dish} tasted completely bland. Save your money.",
        "Disappointed. I had high hopes for {restaurant} but the {dish} was greasy and overpriced. The ambiance was too noisy.",
        "Avoid this place! {restaurant} is highly overrated. The service was slow and the {dish} was undercooked.",
        "Worst meal I've had all year. {restaurant} was dirty, the waiter ignored us, and the {dish} made me feel sick."
    ]
}

RESTAURANTS = [
    ("Bella Italia", "lasagna"),
    ("Sakura Sushi", "salmon sashimi"),
    ("The Burger Joint", "truffle burger"),
    ("Spice & Curry", "chicken tikka masala"),
    ("Le Bistro", "steak frites"),
    ("Taco Loco", "barbacoa tacos")
]


def generate_batch_with_gemini(restaurant_name: str, dish: str, is_positive: bool, num_in_batch: int, api_key: str) -> list:
    """Calls Gemini API to generate a batch of reviews."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    sentiment = "positive and glowing" if is_positive else "negative, critical and sabotage-style"
    prompt = (
        f"Generate a list of exactly {num_in_batch} realistic online customer reviews for a restaurant named '{restaurant_name}'.\n"
        f"The reviews must be {sentiment}, and each review must mention '{dish}' naturally.\n"
        f"Make them look like real, slightly rushed customer reviews of varying lengths (1-3 sentences).\n"
        f"Return the output strictly as a JSON array of strings. Do not include markdown code block formatting (no backticks)."
    )

    payload = {
        "contents": [{
            "parts": [{
                "text": prompt
            }]
        }],
        "generationConfig": {
            "temperature": 0.9,
            "maxOutputTokens": 800,
            "responseMimeType": "application/json"
        }
    }
    
    response = requests.post(url, json=payload, timeout=20)
    if response.status_code == 200:
        response_json = response.json()
        try:
            raw_text = response_json["candidates"][0]["content"]["parts"][0]["text"].strip()
            parsed_list = json.loads(raw_text)
            if isinstance(parsed_list, list):
                cleaned_list = [str(item).strip() for item in parsed_list]
                return cleaned_list
            raise ValueError("Parsed JSON is not a list")
        except (KeyError, IndexError, json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse Gemini batch response: {e}. Raw response: {response_json}")
    else:
        raise Exception(f"Gemini API Error (Status {response.status_code}): {response.text}")


def generate_batch_with_ollama(restaurant_name: str, dish: str, is_positive: bool, num_in_batch: int, model: str, host: str) -> list:
    """Calls a local Ollama server to generate a batch of reviews in JSON format."""
    url = f"{host}/api/generate"
    
    sentiment = "positive and glowing" if is_positive else "negative, critical and sabotage-style"
    prompt = (
        f"Generate a list of exactly {num_in_batch} realistic online customer reviews for a restaurant named '{restaurant_name}'.\n"
        f"The reviews must be {sentiment}, and each review must mention '{dish}' naturally.\n"
        f"Make them look like real, slightly rushed customer reviews of varying lengths (1-3 sentences).\n"
        f"Return the output strictly as a JSON array of strings. For example: [\"Review 1...\", \"Review 2...\"]. Do not include any other text."
    )
    
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0.9
        }
    }
    
    response = requests.post(url, json=payload, timeout=30)
    if response.status_code == 200:
        response_json = response.json()
        try:
            raw_text = response_json["response"].strip()
            parsed_list = json.loads(raw_text)
            if isinstance(parsed_list, list):
                cleaned_list = [str(item).strip() for item in parsed_list]
                return cleaned_list
            raise ValueError("Parsed JSON is not a list")
        except (KeyError, json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Failed to parse Ollama batch response: {e}. Raw response: {response_json}")
    else:
        raise Exception(f"Ollama API Error (Status {response.status_code}): {response.text}")


def save_human_readable_reviews(df_reviews: pd.DataFrame):
    """Saves the generated reviews to a human-readable text file for manual auditing."""
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_path = logs_dir / "generated_reviews.txt"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("GENERATED FAKE REVIEWS AUDIT LOG\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, row in df_reviews.iterrows():
            f.write(f"Review #{idx + 1}\n")
            f.write(f"Restaurant: {row.get('restaurant_name', 'N/A')}\n")
            f.write(f"Rating:     {row['rating']} stars\n")
            f.write(f"Origin:     {row['origin']}\n")
            f.write(f"Text:\n{row['text']}\n")
            f.write("-" * 80 + "\n\n")
            
    logger.info(f"Saved human-readable review audit log to: {output_path.resolve()}")


def generate_reviews(num_reviews=500) -> pd.DataFrame:
    """
    Generates synthetic reviews. Dynamically reads configuration to decide whether
    to use Gemini, Ollama, or fallback to mock templates.
    """
    config = load_config()
    gen_params = config.get("generator_params", {})
    backend = gen_params.get("backend", "mock").lower()
    
    reviews_list = []
    batch_size = 10  # Number of reviews to generate in a single call
    
    # Check if Gemini key is available if backend is gemini
    api_key = os.getenv("GEMINI_API_KEY")
    if backend == "gemini" and (api_key is None or api_key == ""):
        logger.warning("GEMINI_API_KEY not found in environment. Defaulting backend to mock.")
        backend = "mock"

    if backend == "gemini":
        logger.info(f"Generating {num_reviews} reviews via Gemini API (Batch size: {batch_size})...")
        logger.info("Rate limiting enabled: 4.5 seconds delay between requests.")
    elif backend == "ollama":
        model = gen_params.get("ollama_model", "llama3")
        host = gen_params.get("ollama_host", "http://localhost:11434")
        logger.info(f"Generating {num_reviews} reviews via local Ollama (Model: {model}, Host: {host}, Batch size: {batch_size})...")
    else:
        logger.info(f"Generating {num_reviews} mock reviews offline using templates.")

    total_batches = (num_reviews + batch_size - 1) // batch_size
    
    for b in range(total_batches):
        restaurant_name, dish = random.choice(RESTAURANTS)
        is_positive = (b % 2 == 0)
        rating = 5.0 if is_positive else 1.0
        label = 1  # 1 = computer-generated/fake
        
        current_batch_size = min(batch_size, num_reviews - len(reviews_list))
        batch_reviews = []
        origin = backend
        
        if backend == "gemini":
            try:
                batch_reviews = generate_batch_with_gemini(
                    restaurant_name=restaurant_name,
                    dish=dish,
                    is_positive=is_positive,
                    num_in_batch=current_batch_size,
                    api_key=api_key
                )
                time.sleep(4.5)
            except Exception as e:
                logger.error(f"Gemini API Error on batch {b + 1}: {e}. Falling back to templates...")
                origin = "Mock"
        elif backend == "ollama":
            try:
                batch_reviews = generate_batch_with_ollama(
                    restaurant_name=restaurant_name,
                    dish=dish,
                    is_positive=is_positive,
                    num_in_batch=current_batch_size,
                    model=model,
                    host=host
                )
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Ollama Error on batch {b + 1}: {e}. Make sure Ollama is running and model '{model}' is pulled. Falling back to templates...")
                origin = "Mock"
                
        # Handle template fallback or direct mock backend
        if len(batch_reviews) == 0 or origin == "Mock" or backend == "mock":
            templates = MOCK_TEMPLATES["positive"] if is_positive else MOCK_TEMPLATES["negative"]
            batch_reviews = [
                random.choice(templates).format(restaurant=restaurant_name, dish=dish)
                for _ in range(current_batch_size)
            ]
            origin = "Mock"
            
        for text in batch_reviews:
            reviews_list.append({
                "category": "Restaurant",
                "restaurant_name": restaurant_name,
                "rating": rating,
                "text": text,
                "label": label,
                "origin": origin
            })
            
        logger.info(f"Generated {len(reviews_list)}/{num_reviews} reviews.")

    df_synthetic = pd.DataFrame(reviews_list)
    
    # Save a readable text file for manual auditing
    save_human_readable_reviews(df_synthetic)
    
    return df_synthetic


def main():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = raw_dir / "synthetic_reviews.parquet"
    
    config = load_config()
    num_revs = config.get("data_params", {}).get("num_synthetic_reviews", 100)
    
    df_gen = generate_reviews(num_reviews=num_revs)
    df_gen.to_parquet(output_path, index=False)
    logger.info(f"Saved synthetic parquet to: {output_path.resolve()}")


if __name__ == "__main__":
    from src.utils import setup_logging
    setup_logging("generation.log")
    main()
