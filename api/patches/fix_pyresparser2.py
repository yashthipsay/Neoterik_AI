import sys
import os
import re
import shutil
from pathlib import Path

# Find the site-packages directory for your virtualenv
site_packages = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"

# Path to the utils.py file in pyresparser
utils_file = site_packages / "pyresparser" / "utils.py"

if not utils_file.exists():
    print(f"Could not find {utils_file}")
    sys.exit(1)

# Backup the original file
backup_file = utils_file.with_suffix('.py.bak')
shutil.copy2(utils_file, backup_file)
print(f"Created backup at {backup_file}")

# Read the current content
with open(utils_file, 'r') as f:
    content = f.read()

# Use regex to locate the matcher.add line for rule 'NAME'
pattern_regex = r"matcher\.add\s*\(\s*[\'\"]NAME[\'\"]\s*,\s*None\s*,\s*\*pattern\s*\)"
replacement_code = r"""
# Convert pattern(s) to spaCy 3.x format
if not isinstance(pattern, list):
    pattern = [pattern]
spacy_patterns = []
for token in pattern:
    if isinstance(token, dict):
        spacy_patterns.append(token)
    elif isinstance(token, str):
        spacy_patterns.append({"LOWER": token.lower()})
    else:
        # Skip tokens that are not a dict or string
        continue
# Ensure we have at least one valid token before adding
if spacy_patterns:
    matcher.add("NAME", [spacy_patterns])
else:
    raise ValueError("No valid token pattern found for matcher rule 'NAME'")
"""
patched_content = re.sub(pattern_regex, replacement_code, content)

if patched_content != content:
    with open(utils_file, 'w') as f:
        f.write(patched_content)
    print("\nSuccessfully patched the file!")
else:
    # Try alternative pattern matching to print details
    alt_pattern = r"matcher\.add\s*\(\s*[\'\"]NAME[\'\"]\s*,.+\)"
    print("\nTrying to find alternative pattern...")
    alt_matches = re.findall(alt_pattern, content)
    for i, match in enumerate(alt_matches):
        print(f"{i+1}: {match}")
    
    print("\nNo changes were made. The file might already be patched or has a different structure.")