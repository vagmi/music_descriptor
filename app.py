import gradio as gr
#
from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
import torchaudio
import torchaudio.transforms as T
import logging

import json
import os

import importlib 
modeling_MERT = importlib.import_module("MERT-v1-95M.modeling_MERT")

from Prediction_Head.MTGGenre_head import MLPProberBase 
# input cr: https://huggingface.co/spaces/thealphhamerc/audio-to-text/blob/main/app.py


logger = logging.getLogger("MERT-v1-95M-app")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s;%(levelname)s;%(message)s", "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)



inputs = [
    gr.components.Audio(type="filepath", label="Add music audio file"), 
    gr.inputs.Audio(source="microphone", type="filepath"),
]
live_inputs = [
    gr.Audio(source="microphone",streaming=True, type="filepath"),
]
# outputs = [gr.components.Textbox()]
# outputs = [gr.components.Textbox(), transcription_df]
title = "One Model for All Music Understanding Tasks"
description = "An example of using the [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) model as backbone to conduct multiple music understanding tasks with the universal represenation."
article = "The tasks include EMO, GS, MTGInstrument, MTGGenre, MTGTop50, MTGMood, NSynthI, NSynthP, VocalSetS, VocalSetT. \n\n More models can be referred at the [map organization page](https://huggingface.co/m-a-p)."
audio_examples = [
    # ["input/example-1.wav"],
    # ["input/example-2.wav"],
]

# Load the model and the corresponding preprocessor config
# model = AutoModel.from_pretrained("m-a-p/MERT-v0-public", trust_remote_code=True)
# processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v0-public",trust_remote_code=True)
model = modeling_MERT.MERTModel.from_pretrained("./MERT-v1-95M")
processor = Wav2Vec2FeatureExtractor.from_pretrained("./MERT-v1-95M")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MERT_BEST_LAYER_IDX = {
    'EMO': 5,
    'GS': 8,
    'GTZAN': 7,
    'MTGGenre': 7,
    'MTGInstrument': 'all',
    'MTGMood': 6,
    'MTGTop50': 6,
    'MTT': 'all',
    'NSynthI': 6,
    'NSynthP': 1,
    'VocalSetS': 2,
    'VocalSetT': 9,
} 

MERT_BEST_LAYER_IDX = {
    'EMO': 5,
    'GS': 8,
    'GTZAN': 7,
    'MTGGenre': 7,
    'MTGInstrument': 'all',
    'MTGMood': 6,
    'MTGTop50': 6,
    'MTT': 'all',
    'NSynthI': 6,
    'NSynthP': 1,
    'VocalSetS': 2,
    'VocalSetT': 9,
} 
CLASSIFIERS = {

}

ID2CLASS = {

}

TASKS = ['EMO','GS', 'MTGInstrument', 'MTGGenre', 'MTGTop50', 'MTGMood', 'NSynthI', 'NSynthP', 'VocalSetS', 'VocalSetT']
head_dir = './Prediction_Head/best-layer-MERT-v1-95M'
for task in TASKS:
    print('loading', task)
    with open(os.path.join(head_dir,f'{task}.id2class.json'), 'r') as f:
        ID2CLASS[task]=json.load(f)
    num_class = len(ID2CLASS[task].keys())
    CLASSIFIERS[task] = MLPProberBase(d=768, layer=MERT_BEST_LAYER_IDX[task], num_outputs=num_class)
    CLASSIFIERS[task].load_state_dict(torch.load(f'{head_dir}/{task}.ckpt')['state_dict'])
    CLASSIFIERS[task].to(device)

model.to(device)

def model_infernce(inputs):
    waveform, sample_rate = torchaudio.load(inputs)

    resample_rate = processor.sampling_rate

    # make sure the sample_rate aligned
    if resample_rate != sample_rate:
        print(f'setting rate from {sample_rate} to {resample_rate}')
        resampler = T.Resample(sample_rate, resample_rate)
        waveform = resampler(waveform)
    
    waveform = waveform.view(-1,) # make it (n_sample, )
    model_inputs = processor(waveform, sampling_rate=resample_rate, return_tensors="pt")
    model_inputs.to(device)
    with torch.no_grad():
        model_outputs = model(**model_inputs, output_hidden_states=True)

    # take a look at the output shape, there are 13 layers of representation
    # each layer performs differently in different downstream tasks, you should choose empirically
    all_layer_hidden_states = torch.stack(model_outputs.hidden_states).squeeze()[1:,:,:].unsqueeze(0)
    print(all_layer_hidden_states.shape) # [13 layer, Time steps, 768 feature_dim]
    all_layer_hidden_states = all_layer_hidden_states.mean(dim=2)

    task_output_texts = ""
    for task in TASKS:
        num_class = len(ID2CLASS[task].keys())
        if MERT_BEST_LAYER_IDX[task] == 'all':
            logits = CLASSIFIERS[task](all_layer_hidden_states) # [1, 87]
        else:
            logits = CLASSIFIERS[task](all_layer_hidden_states[:, MERT_BEST_LAYER_IDX[task]])
        print(f'task {task} logits:', logits.shape, 'num class:', num_class)
        
        sorted_idx = torch.argsort(logits, dim = -1, descending=True)[0] # batch =1 
        sorted_prob,_ = torch.sort(nn.functional.softmax(logits[0], dim=-1), dim=-1, descending=True)
        # print(sorted_prob)
        # print(sorted_prob.shape)
        
        top_n_show = 3 if num_class >= 3 else num_class
        task_output_texts = task_output_texts + f"TASK {task} output:\n" + "\n".join([str(ID2CLASS[task][str(sorted_idx[idx].item())])+f', probability: {sorted_prob[idx].item():.2%}' for idx in range(top_n_show)]) + '\n'
        task_output_texts = task_output_texts + '----------------------\n'
        # output_texts = "\n".join([id2cls[str(idx.item())].replace('genre---', '') for idx in sorted_idx[:5]])
    # logger.warning(all_layer_hidden_states.shape)
    
    # return f"device {device}, sample reprensentation:  {str(all_layer_hidden_states[12, 0, :10])}"
    # return f"device: {device}\n" + output_texts
    return task_output_texts
    
def convert_audio(inputs, microphone):
    if (microphone is not None):
        inputs = microphone

    text = model_infernce(inputs)

    return text
    
def live_convert_audio(microphone):
    if (microphone is not None):
        inputs = microphone
    
    text = model_infernce(inputs)

    return text

audio_chunked = gr.Interface(
    fn=convert_audio,
    inputs=inputs,
    outputs=[gr.components.Textbox()],
    allow_flagging="never",
    title=title,
    description=description,
    article=article,
    examples=audio_examples,
)

live_audio_chunked = gr.Interface(
    fn=live_convert_audio,
    inputs=live_inputs,
    outputs=[gr.components.Textbox()],
    allow_flagging="never",
    title=title,
    description=description,
    article=article,
    # examples=audio_examples,
    live=True,
)


demo = gr.Blocks()
with demo:
    gr.TabbedInterface(
        [
            audio_chunked,
            live_audio_chunked,
        ], 
        [
            "Audio File or Recording",
            "Live Streaming Music"
        ]
    )
demo.queue(concurrency_count=1, max_size=5)
demo.launch(show_api=False)