import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr

OUTPUT_DIR = Path("output")
REFERENCE = "reference_clip3.wav"


# ponytail: runs narrate_expressive.py as a subprocess, so the model reloads on every request.
# Keep the model in memory here if load time becomes a problem.
def narrate(text):
    if not text.strip():
        raise gr.Error("Metin boş.")
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / f"{datetime.now():%Y%m%d_%H%M%S}.wav"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as f:
        f.write(text)
    proc = subprocess.Popen(
        [sys.executable, "-u", "narrate_expressive.py", f.name, REFERENCE, "-o", str(out),
         "--exaggeration", "0.6", "--sentence_pause", "0.2", "--paragraph_pause", "1.2",
         "--fade", "0.5", "--loudness", "-16"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    lines = []
    for line in proc.stdout:
        lines.append(line.rstrip())
        yield None, "\n".join(lines[-200:])
    proc.wait()
    Path(f.name).unlink()
    if proc.returncode != 0:
        raise gr.Error("\n".join(lines[-15:]))
    yield str(out), "\n".join(lines[-200:])


with gr.Blocks(title="Narrator") as demo:
    text = gr.Textbox(label="Metin", lines=15, placeholder="Metni buraya yapıştır...")
    btn = gr.Button("Seslendir", variant="primary")
    audio = gr.Audio(label="Çıktı")
    log = gr.Textbox(label="Log", lines=12, max_lines=12, autoscroll=True)
    btn.click(narrate, text, [audio, log])

if __name__ == "__main__":
    demo.launch()
