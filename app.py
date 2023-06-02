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
import re

import pandas as pd

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

title = "One Model for All Music Understanding Tasks"
description = "An example of using the [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) model as backbone to conduct multiple music understanding tasks with the universal represenation."
article = "The tasks include EMO, GS, MTGInstrument, MTGGenre, MTGTop50, MTGMood, NSynthI, NSynthP, VocalSetS, VocalSetT. \n\n More models can be referred at the [map organization page](https://huggingface.co/m-a-p)."
audio_examples = [
    # ["input/example-1.wav"],
    # ["input/example-2.wav"],
]

df_init = pd.DataFrame(columns=['Task', 'Top 1', 'Top 2', 'Top 3', 'Top 4', 'Top 5'])
transcription_df = gr.DataFrame(value=df_init, label="Output Dataframe", row_count=(
    0, "dynamic"), max_rows=30, wrap=True, overflow_row_behaviour='paginate')
# outputs = [gr.components.Textbox()]
outputs = transcription_df

df_init_live = pd.DataFrame(columns=['Task', 'Top 1', 'Top 2', 'Top 3', 'Top 4', 'Top 5'])
transcription_df_live = gr.DataFrame(value=df_init_live, label="Output Dataframe", row_count=(
    0, "dynamic"), max_rows=30, wrap=True, overflow_row_behaviour='paginate')
outputs_live = transcription_df_live

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

TASKS = ['GS', 'MTGInstrument', 'MTGGenre', 'MTGTop50', 'MTGMood', 'NSynthI', 'NSynthP', 'VocalSetS', 'VocalSetT','EMO',]
Regression_TASKS = ['EMO']
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
        # print(f'setting rate from {sample_rate} to {resample_rate}')
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
    df = pd.DataFrame(columns=['Task', 'Top 1', 'Top 2', 'Top 3', 'Top 4', 'Top 5'])
    df_objects = []

    for task in TASKS:
        num_class = len(ID2CLASS[task].keys())
        if MERT_BEST_LAYER_IDX[task] == 'all':
            logits = CLASSIFIERS[task](all_layer_hidden_states) # [1, 87]
        else:
            logits = CLASSIFIERS[task](all_layer_hidden_states[:, MERT_BEST_LAYER_IDX[task]])
        # print(f'task {task} logits:', logits.shape, 'num class:', num_class)
        
        sorted_idx = torch.argsort(logits, dim = -1, descending=True)[0] # batch =1 
        sorted_prob,_ = torch.sort(nn.functional.softmax(logits[0], dim=-1), dim=-1, descending=True)
        # print(sorted_prob)
        # print(sorted_prob.shape)
        
        top_n_show = 5 if num_class >= 5 else num_class
        # task_output_texts = task_output_texts + f"TASK {task} output:\n" + "\n".join([str(ID2CLASS[task][str(sorted_idx[idx].item())])+f', probability: {sorted_prob[idx].item():.2%}' for idx in range(top_n_show)]) + '\n'
        # task_output_texts = task_output_texts + '----------------------\n'

        row_elements = [task]
        for idx in range(top_n_show):
            print(ID2CLASS[task])
            # print('id', str(sorted_idx[idx].item()))
            output_class_name = str(ID2CLASS[task][str(sorted_idx[idx].item())])
            output_class_name = re.sub(r'^\w+---', '', output_class_name)
            output_class_name = re.sub(r'^\w+\/\w+---', '', output_class_name)
            # print('output name', output_class_name)
            output_prob = f' {sorted_prob[idx].item():.2%}'
            row_elements.append(output_class_name+output_prob)
        # fill empty elment
        for _ in range(5+1 - len(row_elements)):
            row_elements.append(' ')
        df_objects.append(row_elements)
    df = pd.DataFrame(df_objects, columns=['Task', 'Top 1', 'Top 2', 'Top 3', 'Top 4', 'Top 5'])
    return df
    
def convert_audio(inputs, microphone):
    if (microphone is not None):
        inputs = microphone
    df = model_infernce(inputs)    
    return df

def live_convert_audio(microphone):
    if (microphone is not None):
        inputs = microphone
    df = model_infernce(inputs)    
    return df

audio_chunked = gr.Interface(
    fn=convert_audio,
    inputs=inputs,
    outputs=outputs,
    allow_flagging="never",
    title=title,
    description=description,
    article=article,
    examples=audio_examples,
)

live_audio_chunked = gr.Interface(
    fn=live_convert_audio,
    inputs=live_inputs,
    outputs=outputs_live,
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
# demo.queue(concurrency_count=1, max_size=5)
demo.launch(show_api=False)