import subprocess
import json

accounts = [
    "mrbeast",
    "khaby00",
    "therock",
    "natgeo",
    "nasa",
    "espn",
    "barstoolsports",
    "9gag",
    "pubity",
    "complex"
]

reels = []

for account in accounts:
    print(f"Fetching reels from @{account}...")
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "-J",
        f"https://www.instagram.com/{account}/reels/"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  Failed for @{account}: {result.stderr[:200]}")
        continue

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"  Failed to parse JSON for @{account}")
        continue

    count = 0
    for entry in data.get("entries", []):
        if entry.get("url"):
            reels.append(f"https://www.instagram.com/reel/{entry['id']}/")
            count += 1
    print(f"  Found {count} reels")

# limit to 70
reels = reels[:70]

with open("reels.txt", "w") as f:
    for r in reels:
        f.write(r + "\n")

print(f"\nSaved {len(reels)} reels to reels.txt")
