#!/usr/bin/env python3
"""
Generate an audio.json manifest template from passages.json.

This creates a JSON file with all passage names as keys and null values,
which you can then edit to add audio file paths.

Usage:
    python scripts/generate_audio_manifest.py

    # Or with custom paths:
    python scripts/generate_audio_manifest.py --input data.json --output audio.json
"""

import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Generate audio.json template from passages.json"
    )
    parser.add_argument(
        "--input", "-i",
        default="data/passages.json",
        help="Path to passages.json (default: data/passages.json)"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/audio.json",
        help="Path to output audio.json (default: data/audio.json)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing audio.json without prompting"
    )

    args = parser.parse_args()

    # Check input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return 1

    # Check if output already exists
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        response = input(f"{output_path} already exists. Overwrite? [y/N] ")
        if response.lower() != 'y':
            print("Aborted.")
            return 0

    # Load passages
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    passages = data.get('passages', [])

    if not passages:
        print("Warning: No passages found in input file")

    # Create manifest with null values
    manifest = {}
    for passage in passages:
        name = passage.get('name', '')
        if name:
            manifest[name] = None

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Created {output_path} with {len(manifest)} entries")
    print("\nNext steps:")
    print("1. Add your audio files to static/audio/")
    print("2. Edit audio.json to map passage names to file paths")
    print('   Example: "Passage Name": "audio/recording.mp3"')

    return 0


if __name__ == "__main__":
    exit(main())
