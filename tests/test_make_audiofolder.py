import csv
import json
import struct
import subprocess
import sys
import wave
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_audiofolder.py"

ROWS = [
    ("wavs/a_0.wav", "train", "a", "sɛ wo kɔ, ɛyɛ dɛ.", 1.0),
    ("wavs/b_0.wav", "validation", "b", "yie!", 1.0),
    ("wavs/c_0.wav", "train", "c", ".", 1.0),  # only punctuation: must be skipped
    ("wavs/d_0.wav", "test", "d", "Ɔkɔ fie", 1.0),
]


def make_prepared(tmp_path, name="metadata.tsv"):
    data = tmp_path / "prepared"
    (data / "wavs").mkdir(parents=True)
    for r in ROWS:
        with wave.open(str(data / r[0]), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(struct.pack("<h", 0) * 1600)
    with open(data / name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        w.writerow(["file", "split", "source_id", "text", "seconds"])
        w.writerows(ROWS)
    return data


def run(data, out, *flags):
    return subprocess.run([sys.executable, str(SCRIPT), "--data", str(data), "--out", str(out), *flags],
                          capture_output=True, text=True)


def read_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]


def test_writes_jsonl_with_string_file_names_and_skips_punctuation_only_text(tmp_path):
    data, out = make_prepared(tmp_path), tmp_path / "ds"
    p = run(data, out)
    assert p.returncode == 0, p.stderr
    assert not list(out.rglob("metadata.csv"))  # CSV goes through pandas, which the loader mishandles
    train = read_jsonl(out / "train" / "metadata.jsonl")
    assert train == [{"file_name": "a_0.wav", "text": "sɛ wo kɔ, ɛyɛ dɛ."}]
    assert all(isinstance(r["file_name"], str) for r in train)
    assert read_jsonl(out / "validation" / "metadata.jsonl") == [{"file_name": "b_0.wav", "text": "yie!"}]
    assert (out / "train" / "a_0.wav").exists() and not (out / "train" / "c_0.wav").exists()


def test_strip_punct_option(tmp_path):
    data, out = make_prepared(tmp_path), tmp_path / "ds"
    assert run(data, out, "--strip-punct").returncode == 0
    assert read_jsonl(out / "train" / "metadata.jsonl")[0]["text"] == "sɛ wo kɔ ɛyɛ dɛ"


def test_prefers_the_filtered_metadata(tmp_path):
    data = make_prepared(tmp_path)
    (data / "metadata_filtered.tsv").write_text(
        "file\tsplit\tsource_id\ttext\tseconds\nwavs/d_0.wav\ttest\td\tƆkɔ fie\t1.0\n", encoding="utf-8")
    out = tmp_path / "ds"
    assert run(data, out).returncode == 0
    assert not (out / "train").exists() and read_jsonl(out / "test" / "metadata.jsonl")[0]["file_name"] == "d_0.wav"
