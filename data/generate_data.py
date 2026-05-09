#!/usr/bin/env python
# ============================================================
#  data/generate_data.py
#  Generates a balanced synthetic review dataset (CSV).
#  Run once before training if you don't have real data.
# ============================================================

import csv
import os
import random

random.seed(42)

POSITIVE = [
    "This product is absolutely amazing, I love every bit of it!",
    "Exceeded all my expectations. Highly recommend to everyone.",
    "Best purchase I have made this year. Fantastic quality!",
    "Outstanding performance and great value for the price.",
    "Five stars! Will definitely buy again from this seller.",
    "Superb craftsmanship and delivery was incredibly fast.",
    "I am so happy with this purchase. Works perfectly.",
    "Incredible product – worth every penny spent on it.",
    "Really impressed with the build quality and packaging.",
    "Absolutely love it! Does exactly what it promises.",
    "Top-notch quality and customer service was excellent.",
    "Perfect gift idea. My family loved it very much.",
    "Very satisfied with the product. No complaints at all.",
    "Great product, arrived quickly and in perfect condition.",
    "Brilliant! Way better than I expected for the price.",
    "This is the best gadget I have ever owned. Stunning!",
    "Smooth, responsive, and looks gorgeous. 10 out of 10.",
    "Arrived earlier than expected and works beautifully.",
    "Highly recommend! The product quality is top tier.",
    "So happy with this. The photos don't do it justice.",
    "Wonderful experience from ordering to delivery.",
    "Exceptional value. Cannot believe how good this is.",
    "Would definitely recommend this to all my friends.",
    "Pleasantly surprised by how good the quality is.",
    "Works like a charm. Very happy with my purchase.",
    "Perfect in every way. Shipping was super fast too.",
    "Excellent product and the service was outstanding.",
    "Love the design and it works even better than described.",
    "Flawless purchase. Very impressed with everything.",
    "Amazing quality at this price point. A must-buy!",
]

NEUTRAL = [
    "The product is okay. Does the job but nothing special.",
    "Average quality for the price. Not bad, not great.",
    "It arrived on time and works as described. Fine.",
    "Decent product. Does what it says on the box.",
    "It is fine I guess. Nothing to get excited about.",
    "Mediocre. I have seen better but also worse.",
    "Serviceable. Gets the job done without any extras.",
    "It is an average product. No major issues so far.",
    "Acceptable for everyday use. Won't blow your mind.",
    "Fairly standard item. Matches the description.",
    "Not bad. Meets the basic requirements I had.",
    "It works. That is about all I can say for it.",
    "Middle of the road. Has pros and cons equally.",
    "Generic product. Exactly what you would expect.",
    "OK product. Nothing remarkable or disappointing.",
    "I have mixed feelings. Some good aspects, some meh.",
    "Standard quality. Arrived on time in good condition.",
    "Passable. Would not go out of my way to recommend.",
    "Average build quality. Functional but not impressive.",
    "Ordinary item. Does the basics well, nothing more.",
    "So-so. It gets the job done but could be better.",
    "It is alright. Nothing special stands out to me.",
    "Fair product for the cost. Nothing fancy at all.",
    "OK I suppose. Functionality is there, style is not.",
    "Basic and functional. No real complaints though.",
    "Normal product experience. No surprises either way.",
    "It does what it needs to. Average in every respect.",
    "Fine for casual use. Don't expect premium quality.",
    "Indifferent about this purchase. Just an okay item.",
    "It is what it is. Meets minimum expectations.",
]

NEGATIVE = [
    "Terrible product! Broke within two days of use.",
    "Absolute waste of money. Do not buy this product.",
    "Very disappointed. The quality is shockingly poor.",
    "Worst purchase I have ever made. Stay away!",
    "Product arrived damaged and support was useless.",
    "Complete rubbish. Stopped working after one week.",
    "Misleading photos. The actual product is awful.",
    "Never buying from this seller again. Horrible!",
    "The item is nothing like the description. Scam!",
    "Extremely poor quality. Fell apart immediately.",
    "Do NOT waste your money on this. It is terrible.",
    "One star is too many. Absolute garbage product.",
    "Deeply unsatisfied. This should not be sold.",
    "Regret buying this. Totally useless product.",
    "Appalling quality control. Every part was broken.",
    "This rubbish broke the very first time I used it.",
    "Fraud! Item was completely different from images.",
    "Horrible experience from start to finish. Avoid!",
    "The worst quality I have ever seen. Disgusting.",
    "Save your money. This product is a total disaster.",
    "Dreadful build quality. Cannot recommend at all.",
    "Faulty right out of the box. Refund process awful.",
    "Useless and cheap. Not worth a single rupee.",
    "Total disappointment. Expected much much better.",
    "Shocking. I cannot believe they sell this junk.",
    "Atrocious quality. Broke on the very first use.",
    "Extremely frustrating experience. Will not return.",
    "Garbage product. The company should be ashamed.",
    "Bought as a gift – embarrassed to have given it.",
    "Zero quality control. This is borderline illegal.",
]

SOCIAL_POSITIVE = [
    "Just tried this and it's honestly fire 🔥 love it!",
    "Can't stop recommending this to everyone I know lol",
    "Obsessed with this product rn. Game changer fr.",
    "Not sponsored but this is genuinely incredible.",
    "Y'all need to try this ASAP it's so good!!",
]

SOCIAL_NEGATIVE = [
    "This product is literally the worst thing I ever bought smh",
    "Bruh this broke in like 2 days wtf",
    "Would give 0 stars if I could this is trash",
    "Don't @ me but this product is a scam fr fr",
    "Spent way too much on this garbage ngl",
]

SOCIAL_NEUTRAL = [
    "I guess it's fine? Idk not really my vibe tbh",
    "It's whatever. Gets the job done I suppose lol",
    "Meh. Not good not bad. Just... exists.",
    "Honestly don't have strong feelings about this",
    "It works I guess. Nothing crazy to report here",
]


def build_dataset() -> list[dict]:
    rows = []
    for text in POSITIVE + SOCIAL_POSITIVE:
        rows.append({"text": text, "label": 2})
    for text in NEUTRAL + SOCIAL_NEUTRAL:
        rows.append({"text": text, "label": 1})
    for text in NEGATIVE + SOCIAL_NEGATIVE:
        rows.append({"text": text, "label": 0})

    # Augment with minor variations (swap words, add filler)
    augmented = []
    fillers = ["Honestly, ", "FYI – ", "Update: ", "Review: ", "Note: "]
    for row in rows:
        if random.random() < 0.3:
            augmented.append({
                "text":  random.choice(fillers) + row["text"],
                "label": row["label"],
            })

    rows += augmented
    random.shuffle(rows)
    return rows


def save_dataset(path: str = None) -> str:
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "reviews.csv")

    rows = build_dataset()
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(rows)

    label_counts = {0: 0, 1: 0, 2: 0}
    for r in rows:
        label_counts[r["label"]] += 1

    print(f"✅  Dataset saved → {path}")
    print(f"   Total samples : {len(rows)}")
    print(f"   Positive (2)  : {label_counts[2]}")
    print(f"   Neutral  (1)  : {label_counts[1]}")
    print(f"   Negative (0)  : {label_counts[0]}")
    return path


if __name__ == "__main__":
    save_dataset()
