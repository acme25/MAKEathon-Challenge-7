# python -m apps.test_semantic

from backend.services.semantic_mapper import best_categories_for_term

terms = ["Kletterpark", "Spielplatz", "Skatepark", "Rasenfläche", "Teich", "Weg"]

for t in terms:
    matches = best_categories_for_term(t, top_k=3)
    print(f"\n🔎 '{t}' →")
    for name, score in matches:
        print(f"   {name:30s}  ({score:.3f})")
