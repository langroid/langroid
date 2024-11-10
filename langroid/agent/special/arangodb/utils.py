from typing import Any, Dict, List


def count_fields(schema: Dict[str, List[Dict[str, Any]]]) -> int:
    total = 0
    for coll in schema["Collection Schema"]:
        # Count all keys in each collection's dict
        total += len(coll)
        # Also count properties if they exist
        props = coll.get(f"{coll['collection_type']}_properties", [])
        total += len(props)
    return total


def trim_schema(
    schema: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Keep only edge connection info, remove properties and examples"""
    trimmed: Dict[str, List[Dict[str, Any]]] = {
        "Graph Schema": schema["Graph Schema"],
        "Collection Schema": [],
    }
    for coll in schema["Collection Schema"]:
        col_info: Dict[str, Any] = {
            "collection_name": coll["collection_name"],
            "collection_type": coll["collection_type"],
        }
        if coll["collection_type"] == "edge":
            # preserve from/to info if present
            if f"example_{coll['collection_type']}" in coll:
                example = coll[f"example_{coll['collection_type']}"]
                if example and "_from" in example:
                    col_info["from_collection"] = example["_from"].split("/")[0]
                    col_info["to_collection"] = example["_to"].split("/")[0]
        trimmed["Collection Schema"].append(col_info)
    return trimmed
