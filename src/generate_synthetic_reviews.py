import json
import os
import random
import time
from pathlib import Path
import pandas as pd
import requests

# Pre-defined templates for offline/mock generation if Gemini API key is not provided or fails
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
    """
    Calls Gemini API to generate a batch of reviews in a single request.
    Requests JSON output format for reliable parsing.
    """
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


def generate_reviews(num_reviews=500) -> pd.DataFrame:
    """
    Generates synthetic reviews. Uses batch generation and rate limiting (4.5s delay)
    to respect the Gemini API free tier limits (15 Requests Per Minute, 1500 Requests Per Day).
    """
    api_key = os.getenv("GEMINI_API_KEY")
    use_api = api_key is not None and api_key != ""
    
    reviews_list = []
    batch_size = 10  # Number of reviews to generate in a single API call
    
    if use_api:
        print(f"Using Gemini API to generate {num_reviews} reviews in batches of {batch_size}...")
        print("Rate limiting enabled: 4.5 seconds delay between API calls to stay under 15 RPM.")
    else:
        print("GEMINI_API_KEY not found in environment variables. Generating mock reviews offline.")

    # Calculate total batches needed
    total_batches = (num_reviews + batch_size - 1) // batch_size
    
    for b in range(total_batches):
        restaurant_name, dish = random.choice(RESTAURANTS)
        is_positive = (b % 2 == 0)
        rating = 5.0 if is_positive else 1.0
        label = 1  # 1 = computer-generated/fake
        
        # Calculate size for this specific batch (handles remainder)
        current_batch_size = min(batch_size, num_reviews - len(reviews_list))
        
        batch_reviews = []
        origin = "Gemini" if use_api else "Mock"
        
        if use_api:
            try:
                # Call batch generation
                batch_reviews = generate_batch_with_gemini(
                    restaurant_name=restaurant_name,
                    dish=dish,
                    is_positive=is_positive,
                    num_in_batch=current_batch_size,
                    api_key=api_key
                )
                time.sleep(4.5)
            except Exception as e:
                print(f"\nAPI Error on batch {b + 1}: {e}. Falling back to templates for this batch...")
                templates = MOCK_TEMPLATES["positive"] if is_positive else MOCK_TEMPLATES["negative"]
                batch_reviews = [
                    random.choice(templates).format(restaurant=restaurant_name, dish=dish)
                    for _ in range(current_batch_size)
                ]
                origin = "Mock"  # Mark this batch as mock origin because the API call failed
        else:
            # Offline mock generation
            templates = MOCK_TEMPLATES["positive"] if is_positive else MOCK_TEMPLATES["negative"]
            batch_reviews = [
                random.choice(templates).format(restaurant=restaurant_name, dish=dish)
                for _ in range(current_batch_size)
            ]
            
        for text in batch_reviews:
            reviews_list.append({
                "category": "Restaurant",
                "rating": rating,
                "text": text,
                "label": label,
                "origin": origin
            })
            
        print(f"Generated {len(reviews_list)}/{num_reviews} reviews...", end="\r")

    print(f"\nGeneration complete! Total reviews created: {len(reviews_list)}")
    df_synthetic = pd.DataFrame(reviews_list)
    return df_synthetic


def main():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = raw_dir / "synthetic_reviews.parquet"
    
    # Generate 100 reviews by default for testing
    df_gen = generate_reviews(num_reviews=100)
    
    df_gen.to_parquet(output_path, index=False)
    print(f"Saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
