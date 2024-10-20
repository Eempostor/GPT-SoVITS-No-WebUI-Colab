import os
import gc
import traceback
import torch

import whisperx
import soundfile as sf

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def slice(input_audio, audio_output_folder, transcript_output_folder, model_size, language, precision):
    audio_data, sr = sf.read(input_audio)

    output = []
    os.makedirs(audio_output_folder, exist_ok=True)
    os.makedirs(transcript_output_folder, exist_ok=True)
    output_base_name = os.path.splitext(os.path.basename(input_audio))[0]

    try:
        # whisperx is good i think
        print(f"Loading whisperx {model_size} model...")
        whisperx_model = whisperx.load_model(
            whisper_arch=model_size,
            device=device,
            compute_type="float16" if precision == "fp16" else "float32"
        )

        print("Trancribing the audio...")
        result = whisperx_model.transcribe(
            audio=input_audio,
            batch_size=8,
            language=language,
            print_progress=True
        )
        skipped_segments = 0

        print("Slicing audio based on transcript...")
        for i, segment in enumerate(result["segments"]):
            output_audio_path = os.path.join(audio_output_folder, f"{output_base_name}_{i}.wav")
            start_time = int(segment["start"]*sr)
            end_time = int(segment["end"]*sr)

            if not 54 > (segment["end"]-segment["start"]) > 0.6:
                print(f"Skipped segment {i} due to being under or over the required length")
                skipped_segments += 1
                continue

            new_audio_data = audio_data[start_time:end_time]
            sf.write(
                output_audio_path,
                new_audio_data,
                sr,
                format="wav"
            )

            output.append(f"{output_audio_path}|{output_base_name}|{language.upper()}|{segment['text']}")
            print(f"{i}. [{segment['start']:.2f}s -> {segment['end']:.2f}s]{segment['text']}")
    except:
        return print(traceback.format_exc())

    print(f"Total skipped segments: {skipped_segments}")
    output_transcript_path = os.path.join(transcript_output_folder, f"{output_base_name}.list")
    with open(output_transcript_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output))
    return audio_output_folder, output_transcript_path