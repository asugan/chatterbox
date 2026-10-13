import argparse
import gc
import numpy as np
import torch
import torchaudio as ta
import spacy
import pyloudnorm as pyln
from chatterbox.tts import ChatterboxTTS


def make_silence(duration_s, sample_rate):
    return torch.zeros(1, int(duration_s * sample_rate))


def apply_fade(wav, sr, fade_duration):
    num_samples = wav.shape[1]
    fade_samples = min(int(fade_duration * sr), num_samples // 2)
    fade_in = torch.linspace(0.0, 1.0, fade_samples)
    fade_out = torch.linspace(1.0, 0.0, fade_samples)
    wav[0, :fade_samples] *= fade_in
    wav[0, -fade_samples:] *= fade_out
    return wav


def normalize_loudness(wav, sr, target_lufs):
    audio_np = wav.squeeze(0).numpy()
    meter = pyln.Meter(sr)
    current_lufs = meter.integrated_loudness(audio_np)
    if np.isinf(current_lufs):
        return wav
    audio_np = pyln.normalize.loudness(audio_np, current_lufs, target_lufs)
    return torch.from_numpy(audio_np).unsqueeze(0)


def split_paragraphs(text):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]
    return paragraphs


def split_sentences(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def main():
    parser = argparse.ArgumentParser(description="Chatterbox Expressive TTS Narrator")
    parser.add_argument("text_file", help="Path to .txt file")
    parser.add_argument("reference_audio", help="Path to reference voice audio (min 5s)")
    parser.add_argument("-o", "--output", default="output.wav", help="Output file (default: output.wav)")
    parser.add_argument("--exaggeration", type=float, default=0.85, help="Expressiveness (0.5=subtle, 0.7=natural, 1.0+=dramatic)")
    parser.add_argument("--cfg_weight", type=float, default=0.5, help="Classifier-free guidance weight")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--sentence_pause", type=float, default=0.4, help="Silence between sentences (seconds)")
    parser.add_argument("--paragraph_pause", type=float, default=1.2, help="Silence between paragraphs (seconds)")
    parser.add_argument("--fade", type=float, default=0.5, help="Fade in/out duration (seconds, 0 to disable)")
    parser.add_argument("--loudness", type=float, default=-16.0, help="Target loudness in LUFS (-16=broadcast, 0 to disable)")
    args = parser.parse_args()

    with open(args.text_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    print(f"Text loaded: {len(text)} characters")

    paragraphs = split_paragraphs(text)
    print(f"Paragraphs: {len(paragraphs)}")
    print(f"Exaggeration: {args.exaggeration}, CFG: {args.cfg_weight}")
    print(f"Sentence pause: {args.sentence_pause}s, Paragraph pause: {args.paragraph_pause}s")
    print(f"Fade: {args.fade}s, Target loudness: {args.loudness} LUFS")

    print("Loading model...")
    model = ChatterboxTTS.from_pretrained(device="cuda")

    sentence_silence = make_silence(args.sentence_pause, model.sr)
    paragraph_silence = make_silence(args.paragraph_pause, model.sr)

    all_wavs = []
    total_sentences = 0
    for p_idx, paragraph in enumerate(paragraphs):
        sentences = split_sentences(paragraph)
        total_sentences += len(sentences)

        for s_idx, sentence in enumerate(sentences):
            torch.cuda.empty_cache()
            gc.collect()
            print(f"[P{p_idx+1}/{len(paragraphs)}][S{s_idx+1}/{len(sentences)}] {sentence[:60]}...")
            wav = model.generate(
                text=sentence,
                audio_prompt_path=args.reference_audio,
                exaggeration=args.exaggeration,
                cfg_weight=args.cfg_weight,
                temperature=args.temperature,
            )
            all_wavs.append(wav.cpu())
            del wav

            if s_idx < len(sentences) - 1:
                all_wavs.append(sentence_silence)

        if p_idx < len(paragraphs) - 1:
            all_wavs.append(paragraph_silence)

    print(f"Total sentences generated: {total_sentences}")

    print("Concatenating...")
    final_wav = torch.cat(all_wavs, dim=1)

    if args.fade > 0:
        print(f"Applying fade in/out ({args.fade}s)...")
        final_wav = apply_fade(final_wav, model.sr, args.fade)

    if args.loudness != 0:
        print(f"Normalizing loudness to {args.loudness} LUFS...")
        final_wav = normalize_loudness(final_wav, model.sr, args.loudness)

    ta.save(args.output, final_wav, model.sr)
    duration = final_wav.shape[1] / model.sr
    print(f"Saved: {args.output} ({duration:.1f}s)")


if __name__ == "__main__":
    main()
