import json
import re

with open("ceph_exporter_20250611_110357.json", "r") as f:
    raw = f.read()

# Extract config block
config_match = re.search(r'{\s*"config":.*?"start_time":\s*".*?"\s*}', raw, re.DOTALL)
config_block = json.loads(config_match.group(0))

# Extract all metric blocks
metric_matches = re.findall(r'{\s*"timestamp":\s*".*?"\s*,\s*"metrics":\s*{.*?}\s*}', raw, re.DOTALL)
metric_blocks = [json.loads(m) for m in metric_matches]

# Combine into proper JSON
cleaned = {
    **config_block,
    "data": metric_blocks
}

with open("cleaned_ceph_data.json", "w") as f:
    json.dump(cleaned, f, indent=2)
