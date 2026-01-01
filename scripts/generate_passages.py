#!/usr/bin/env python3
"""
Generate individual passage markdown files from passages.json.

This script:
1. Reads passages.json (your Twisted Tongues export)
2. Creates a markdown file for each passage in content/passages/
3. Outputs a cleaned passages.json with internal links stripped

Usage:
    python scripts/generate_passages.py

    # With custom paths:
    python scripts/generate_passages.py --input export.json --output content/passages
"""

import json
import argparse
import re
from pathlib import Path
from copy import deepcopy


def slugify(text):
    """Convert text to URL-friendly slug."""
    slug = text.lower()
    slug = re.sub(r'[\s_]+', '-', slug)
    slug = re.sub(r'[^a-z0-9-]', '', slug)
    slug = re.sub(r'-+', '-', slug)
    slug = slug.strip('-')
    return slug or 'untitled'


def strip_links(passages_data):
    """Remove link fields from passages data (they're internal to Twisted Tongues)."""
    cleaned = deepcopy(passages_data)

    for passage in cleaned.get('passages', []):
        # Remove passage-level link
        passage.pop('link', None)

        # Remove sentence-level links
        for sentence in passage.get('sentences', []):
            sentence.pop('link', None)

    return cleaned


def main():
    parser = argparse.ArgumentParser(
        description="Generate passage markdown files from passages.json"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/passages.json",
        help="Path to input passages.json (default: data/passages.json)"
    )
    parser.add_argument(
        "--output", "-o",
        default="content/passages",
        help="Output directory for markdown files (default: content/passages)"
    )
    parser.add_argument(
        "--clean", "-c",
        action="store_true",
        help="Remove existing passage files before generating"
    )
    parser.add_argument(
        "--no-strip-links",
        action="store_true",
        help="Don't strip internal links from the output JSON"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return 1

    # Load passages
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    passages = data.get('passages', [])

    if not passages:
        print("Warning: No passages found in input file")
        return 0

    # Prepare output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Clean existing files if requested
    if args.clean:
        for existing in output_path.glob("*.md"):
            if existing.name != "_index.md":
                existing.unlink()
                print(f"Removed: {existing.name}")

    # Track slugs to avoid duplicates
    used_slugs = set()

    # Generate markdown files
    for index, passage in enumerate(passages):
        name = passage.get('name', f'Passage {index + 1}')
        description = passage.get('description', '')

        # Generate unique slug
        base_slug = slugify(name)
        slug = base_slug
        counter = 2
        while slug in used_slugs:
            slug = f"{base_slug}-{counter}"
            counter += 1
        used_slugs.add(slug)

        # Count valid sentences
        sentence_count = sum(
            1 for s in passage.get('sentences', [])
            if s.get('word_tracks') and
               s['word_tracks'] and
               s['word_tracks'][0].get('words') and
               len(s['word_tracks'][0]['words']) > 0
        )

        # Create markdown content
        content = f'''+++
title = "{name.replace('"', '\\"')}"
template = "passage.html"
description = "{description.replace('"', '\\"')}"

[extra]
passage_index = {index}
sentence_count = {sentence_count}
+++
'''

        file_path = output_path / f"{slug}.md"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"Created: {slug}.md ({sentence_count} sentences)")

    # Output cleaned JSON (strip internal links)
    if not args.no_strip_links:
        cleaned_data = strip_links(data)

        # Write back to the same location (overwrite with cleaned version)
        with open(input_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)

        print(f"\nCleaned {input_path} (removed internal links)")

    print(f"\nGenerated {len(passages)} passage files in {output_path}")
    print("\nNext: Run 'zola build' to generate the site")

    return 0


if __name__ == "__main__":
    exit(main())
