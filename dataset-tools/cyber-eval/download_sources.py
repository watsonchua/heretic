"""Download the pinned public inputs; no credentials or decision model required."""

import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "sources"
SOURCES.mkdir(parents=True, exist_ok=True)

REPOS = [
    (
        "mlabonne/harmless_alpaca",
        "02c6a92cfcf11bb0c387334f8146d149d65b587f",
        "harmless_alpaca",
        ["README.md", "data/*.parquet"],
    ),
    (
        "AlignmentResearch/ClearHarm",
        "79464fb6b3c2a8ee925184f394f9636600349f88",
        "clearharm",
        ["README.md", "data/*.parquet"],
    ),
    (
        "purpcode/CyberSecEval-MITRE",
        "70213a497f4e7d5ee5b0089eac5b8d3244eba2af",
        "cyberseceval-mitre",
        ["README.md", "data/*.parquet"],
    ),
    (
        "purpcode/CyberSecEval-FRR",
        "73294ca390f12ebb075d763550b942875dee5fc2",
        "cyberseceval-frr",
        ["README.md", "data/*.parquet"],
    ),
    # Phase 2. Revisions read from the HF API on 2026-10-15. Only ungated repos with a clear
    # licence are listed; gated ones (AdvBench, XSTest upstream, SORRY-Bench, WildGuardMix)
    # need an accepted agreement and a token, so they stay out of the reproducible path.
    ("JailbreakBench/JBB-Behaviors", "886acc352a31", "jbb-behaviors", ["README.md", "data/*"]),
    ("walledai/MaliciousInstruct", "4669349f48fe", "malicious-instruct", ["README.md", "data/*"]),
    ("natolambert/xstest-v2-copy", "b71afe2a6d10", "xstest", ["README.md", "*.csv", "data/*"]),
    ("bench-llm/or-bench", "e36d8b80e818", "or-bench", ["README.md", "*/*.parquet", "data/*"]),
    (
        "AlicanKiraz0/Cybersecurity-Dataset-Fenrir-v2.1",
        "fd7967ddda76",
        "fenrir",
        ["README.md", "*.json*"],
    ),
]
for repo, revision, directory, patterns in REPOS:
    subprocess.run(
        [
            str(Path(sys.executable).with_name("hf")),
            "download",
            repo,
            "--repo-type",
            "dataset",
            "--revision",
            revision,
            "--local-dir",
            str(SOURCES / directory),
            *[arg for pattern in patterns for arg in ("--include", pattern)],
        ],
        check=True,
    )

GITHUB = [
    (
        "alexandrasouly/strongreject",
        "f7cad6c17e624e21d8df2278e918ae1dddb4cb56",
        "strongreject_dataset/strongreject_dataset.csv",
        "strongreject.csv",
    ),
    (
        "alexandrasouly/strongreject",
        "f7cad6c17e624e21d8df2278e918ae1dddb4cb56",
        "LICENSE",
        "StrongREJECT-LICENSE",
    ),
    (
        "centerforaisafety/HarmBench",
        "8e1604d1171fe8a48d8febecd22f600e462bdcdd",
        "data/behavior_datasets/harmbench_behaviors_text_all.csv",
        "harmbench.csv",
    ),
    (
        "centerforaisafety/HarmBench",
        "8e1604d1171fe8a48d8febecd22f600e462bdcdd",
        "LICENSE",
        "HarmBench-LICENSE",
    ),
]
for repo, revision, remote, local in GITHUB:
    with urlopen(
        f"https://raw.githubusercontent.com/{repo}/{revision}/{remote}", timeout=60
    ) as response:
        (SOURCES / local).write_bytes(response.read())
print("Pinned sources saved to", SOURCES)
