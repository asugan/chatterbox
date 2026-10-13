import argparse
import gc
import re
import torch
import torchaudio as ta
from chatterbox.tts_turbo import ChatterboxTurboTTS


def split_text(text, max_chars=250):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) > max_chars:
            if current.strip():
                chunks.append(current.strip())
            current = s
        else:
            current += " " + s
    if current.strip():
        chunks.append(current.strip())
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Chatterbox Turbo TTS Narrator")
    parser.add_argument("text_file", help="Path to .txt file")
    parser.add_argument("reference_audio", help="Path to reference voice audio (min 5s)")
    parser.add_argument("-o", "--output", default="output.wav", help="Output file (default: output.wav)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--chunk_size", type=int, default=250, help="Max characters per chunk")
    args = parser.parse_args()

    with open(args.text_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    print(f"Text loaded: {len(text)} characters")

    chunks = split_text(text, args.chunk_size)
    print(f"Split into {len(chunks)} chunks")

    print("Loading model...")
    model = ChatterboxTurboTTS.from_pretrained(device="cuda")

    all_wavs = []
    for i, chunk in enumerate(chunks):
        torch.cuda.empty_cache()
        gc.collect()
        print(f"[{i+1}/{len(chunks)}] Generating: {chunk[:60]}...")
        wav = model.generate(
            text=chunk,
            audio_prompt_path=args.reference_audio,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )
        all_wavs.append(wav.cpu())
        del wav

    torch.cuda.empty_cache()
    final_wav = torch.cat(all_wavs, dim=1)
    ta.save(args.output, final_wav, model.sr)
    print(f"Saved: {args.output} ({final_wav.shape[1] / model.sr:.1f}s)")


if __name__ == "__main__":
    main()
