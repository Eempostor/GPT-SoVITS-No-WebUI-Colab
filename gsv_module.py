from tools import my_utils

# time decorator
def time_it(func):
    def wrapper(*args, **kwargs):
        t0 = ttime()
        opt = func(*args, **kwargs)
        t1 = ttime()
        return opt, t1-t0
    return wrapper

def run_cmd(cmd):
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
    for line in p.stdout:
        print(line, end='')
    p.wait()

# tools/my_utils.py 38-72
def check_for_existence(file_list: list = None, is_train=False, is_dataset_processing=False):
    files_status = []
    if is_train and file_list:
        file_list.append(os.path.join(file_list[0], '2-name2text.txt'))
        file_list.append(os.path.join(file_list[0], '3-bert'))
        file_list.append(os.path.join(file_list[0], '4-cnhubert'))
        file_list.append(os.path.join(file_list[0], '5-wav32k'))
        file_list.append(os.path.join(file_list[0], '6-name2semantic.tsv'))
    for file in file_list:
        files_status.append(os.path.exists(file))
    if sum(files_status) != len(files_status):
        if is_train:
            bad_files = []
            for file, status in zip(file_list, files_status):
                if not status:
                    bad_files.append(file)
            raise FileNotFoundError(f"The following files or folders do not exist: {bad_files}.")
        elif is_dataset_processing:
            if files_status[0]:
                return True
            elif not files_status[0]:
                raise FileNotFoundError(f'"{file_list[0]}" does not exist.')
            elif not files_status[1] and file_list[1]:
                raise FileNotFoundError(f'"{file_list[1]}" does not exist.')
        else:
            if file_list[0]:
                raise FileNotFoundError(f'"{file_list[0]}" does not exist.')
            else:
                raise ValueError("The path cannot be empty.")
    return True
    
# tools/my_utils.py 74-115
def check_details(path_list=None, is_train=False, is_dataset_processing=False):
    if is_dataset_processing:
        list_path, audio_path = path_list
        if not list_path.endswith('.list'):
            raise ValueError("Transcription file is not a .list file.")
        if not os.path.isdir(audio_path):
            raise ValueError("Dataset path is not a directory.")
        with open(list_path, "r", encoding="utf8") as f:
            line = f.readline().strip("\n").split("\n")
        wav_name, _, __, ___ = line[0].split("|")
        wav_name = my_utils.clean_path(wav_name)
        wav_name = os.path.basename(wav_name)
        wav_path = "%s/%s"%(audio_path, wav_name)
        
        if not os.path.exists(wav_path):
            raise FileNotFoundError("Transcription file contains invalid audio paths.")
    if is_train:
        path_list.append(os.path.join(path_list[0],'2-name2text.txt'))
        path_list.append(os.path.join(path_list[0],'4-cnhubert'))
        path_list.append(os.path.join(path_list[0],'5-wav32k'))
        path_list.append(os.path.join(path_list[0],'6-name2semantic.tsv'))
        phone_path, hubert_path, wav_path, semantic_path = path_list[1:]
        with open(phone_path,'r',encoding='utf-8') as f:
            if not f.read(1):
                raise FileNotFoundError("Phoneme dataset not found.")
        if not os.listdir(hubert_path):
            raise FileNotFoundError("Hubert dataset not found.")
        if not os.listdir(wav_path):
            raise FileNotFoundError("Audio dataset not found.")
        df = pd.read_csv(
            semantic_path, delimiter="\t", encoding="utf-8"
        )
        if not df:
            raise FileNotFoundError("Semantic dataset not found.")
    
from tools.slice_audio import slice as silence_slice
# webui.py 405-430
@time_it
def start_slice(inp, opt_root, threshold, min_length, min_interval, hop_size, max_sil_kept, _max,alpha, n_parts):
    inp = my_utils.clean_path(inp)
    opt_root = my_utils.clean_path(opt_root)
    check_for_existance([inp])
    if not os.path.exists(inp):
        raise FileNotFoundError("The input path does not exist.")
    if os.path.isfile(inp):
        n_parts=1
    elif os.path.isdir(inp):
        pass
    else:
        raise FileNotFoundError("The input path exists but is neither a file nor a folder.")
    for i_part in range(n_parts):
        silence_slice(
            inp=inp,
            opt_root=opt_root,
            threshold=threshold,
            min_length=min_length,
            min_interval=min_interval,
            hop_size=hop_size,
            max_sil_kept=max_sil_kept,
            _max=_max,
            alpha=alpha,
            i_part=i_part,
            all_part=n_parts
        )
    return opt_root
        
from tools.asr.fasterwhisper_asr import execute_asr
# webui.py 250-272
@time_it
def start_asr(asr_inp_dir, asr_opt_dir, asr_model, asr_model_size, asr_lang, asr_precision):
    asr_inp_dir = my_utils.clean_path(asr_inp_dir)
    asr_opt_dir = my_utils.clean_path(asr_opt_dir)
    check_for_existance([asr_inp_dir])
    output_file_path = execute_asr(
        input_folder=asr_inp_dir,
        output_folder=asr_opr_dir,
        model_size=asr_model_size,
        language=asr_lang,
        precision=asr_precision
    )
    return output_file_path
    
# additional feature: slice per utterance
@time_it
def start_slice_per_utterance(input_audio, audio_output_folder, transcript_output_folder, model_size, language, precision, whisper_type="whisperx"):
    input_audio = my_utils.clean_path(input_audio)
    audio_output_folder = my_utils.clean_path(audio_output_folder)
    transcript_output_folder = my_utils.clean_path(transcript_output_folder)
    check_for_existance([input_audio])
    if whisper_type == "whisperx":
        from tools.asr.slice_per_utterance_whisperx import slice as utterance_slice
    else:
        from tools.asr.slice_per_utterance_hfwhisper import slice as utterance_slice
    audio_output_folder, output_transcript_path = utterance_slice(
        input_audio=input_audio,
        audio_output_folder=audio_output_folder,
        transcript_output_folder=transcript_output_folder,
        model_size=model_size,
        language=language,
        precision=precision
    )
    return audio_output_folder, output_transcript_path

# webui.py 444-493
@time_it
def get_text(inp_text, inp_wav_dir, exp_name, all_parts, bert_pretrained_dir):
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    if check_for_existance([inp_text,inp_wav_dir], is_dataset_processing=True):
        check_details([inp_text,inp_wav_dir], is_dataset_processing=True)
    
    opt_dir="%s/%s"%(exp_root,exp_name)
    config={
        "inp_text":inp_text,
        "inp_wav_dir":inp_wav_dir,
        "exp_name":exp_name,
        "opt_dir":opt_dir,
        "bert_pretrained_dir":bert_pretrained_dir,
    }
    for i_part in range(all_parts):
        config.update(
            {
                "i_part": str(i_part),
                "all_parts": str(all_parts),
                "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                "is_half": str(is_half)
            }
        )
        os.environ.update(config)
        cmd = '"%s" GPT_SoVITS/prepare_datasets/1-get-text.py'%python_exec
        run_cmd(cmd)
    opt = []
    for i_part in range(all_parts):
        txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
        with open(txt_path, "r", encoding="utf8") as f:
            opt += f.read().strip("\n").split("\n")
        os.remove(txt_path)
    path_text = "%s/2-name2text.txt" % opt_dir
    with open(path_text, "w", encoding="utf8") as f:
        f.write("\n".join(opt) + "\n")
    if opt:
        print("Text processing successful.")
    else:
        print("Text processing failed.")