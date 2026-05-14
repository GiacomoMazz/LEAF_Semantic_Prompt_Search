# make sure primary text field is explicitly "content"

from config import TEXT_TEMPLATE, METADATA_FIELDS

# handling the tag array situation

def format_tags(tags) -> str:
    if tags is None:
        return ""
    
    if isinstance(tags, list):
        return ", ".join(str(tag) for tag in tags)
    
    return str(tags)

def build_search_text(record: dict) -> str:
    return TEXT_TEMPLATE.format(
        title = record.get("title", ""),
        category = record.get("category", ""),
        subcategory = record.get("subcategory", ""),
        tags = format_tags(record.get("tags", [])),
        difficulty = record.get("difficulty", ""),
        content = record.get("content", ""),
    ).strip()

def format_metadata_value(value):
    if value is None:
        return ""
    
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    
    return str(value)

def build_metadata(record: dict) -> dict:
    metadata = {}

    for field in METADATA_FIELDS:
        metadata[field] = format_metadata_value(record.get(field, ""))
    
    return metadata

def preprocess_record(record: dict) -> dict:
    return {
        "id": str(record.get("id")),
        "text": build_search_text(record),
        "metadata": build_metadata(record)
    }

def preprocess_records(records: list[dict]) -> list[dict]:
    result = []

    for record in records:

        if record.get("id") is not None and record.get("content"):
            result.append(preprocess_record(record))

    return result



if __name__ == "__main__":
    from data_loader import load_raw_data

    raw_records = load_raw_data()

    processed_records = preprocess_records(raw_records)

    print(f"Loaded records: {len(raw_records)}")

    print(f"Processed records: {len(processed_records)}")

    print("\nFirst processed text:\n")

    print(processed_records[0]["text"])

    print("\nFirst processed metadata:\n")

    print(processed_records[0]["metadata"])

