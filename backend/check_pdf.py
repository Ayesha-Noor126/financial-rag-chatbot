import fitz

pdf_path = r"storage\uploads\8718ec25-8fbf-48d2-bab4-6c5346353b39.pdf"

doc = fitz.open(pdf_path)

for i, page in enumerate(doc):
    text = page.get_text("text")

    print("=" * 80)
    print(f"Page {i+1}")
    print(f"Characters: {len(text)}")

    if (
        "revenue" in text.lower()
        or "turnover" in text.lower()
        or "net sales" in text.lower()
    ):
        print(">>> Revenue-related page found <<<")
        print(text[:1500])

    print()