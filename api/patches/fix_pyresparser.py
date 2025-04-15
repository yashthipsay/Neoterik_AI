import sys
import os
from pathlib import Path

# Find the site-packages directory for your virtualenv
site_packages = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"

# Path to the utils.py file in pyresparser
utils_file = site_packages / "pyresparser" / "utils.py"

if not utils_file.exists():
    print(f"Could not find {utils_file}")
    sys.exit(1)

# Read the current content
with open(utils_file, 'r') as f:
    content = f.read()

# Replace the problematic line with a more appropriate fix for spaCy 3.x
if "matcher.add('NAME', None, *pattern)" in content:
    patched_content = content.replace(
        "matcher.add('NAME', None, *pattern)",
        """
        # Convert patterns to spaCy 3.x format
        spacy_patterns = []
        for token in pattern:
            if isinstance(token, dict):
                spacy_patterns.append(token)
            elif isinstance(token, str):
                spacy_patterns.append({"LOWER": token.lower()})
            else:
                # Handle other cases or just skip
                continue
        matcher.add('NAME', [spacy_patterns])
        """
    )
    
    # Write back the patched file
    with open(utils_file, 'w') as f:
        f.write(patched_content)
    
    print(f"Successfully patched {utils_file}")
else:
    print("Could not find the line to patch. The file may already be patched or has a different structure.")