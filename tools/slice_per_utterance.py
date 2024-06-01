import os
import traceback
import torch

from transformers import pipeline
import soundfile as sf

device = 'cuda' if torch.cuda.is_available() else 'cpu'
to_lang_name = {
    "en": "english",
    "ja": "japanese",
    "zh":"chinese"
}
    
def slice(input_audio, audio_output_folder, transcript_output_folder, model_size, language, precision):
    audio_data, sr = sf.read(input_audio)
    
    output = []
    os.makedirs(audio_output_folder, exist_ok=True)
    os.makedirs(transcript_output_folder, exist_ok=True)
    output_base_name = os.path.splitext(os.path.basename(input_audio))[0]
    
    try:
        # hf whisper does better for longer audios
        print(f"Loading whisper {model_size} model...")
        whisper_model = pipeline(
            "automatic-speech-recognition",
            model=f"openai/whisper-{model_size}",
            chunk_length_s=30,
            return_timestamps=True,
            device=device,
        )
        
        print("Trancribing the audio...")
        segments = whisper_model(input_audio, return_timestamps=True, generate_kwargs={"language": to_lang_name[language]})
        skipped_segments = 0
        
        print("Slicing audio based on transcript...")
        for i, segment in enumerate(segments["chunks"]):
            output_audio_path = os.path.join(audio_output_folder, f"{output_base_name}_{i}.wav")
            start_time = int(segment["timestamp"][0]*sr)
            end_time = int(segment["timestamp"][1]*sr)
            
            if not 54 > (segment["timestamp"][1]-segment["timestamp"][0]) > 0.6:
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
            print(f"{i}. [{segment['timestamp'][0]:.2f}s -> {segment['timestamp'][1]:.2f}s]{segment['text']}")
    except:
        return print(traceback.format_exc())
    
    print(f"Total skipped segments: {skipped_segments}")
    output_transcript_path = os.path.join(transcript_output_folder, f"{output_base_name}.list")
    with open(output_transcript_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output))
    return audio_output_folder, output_transcript_path