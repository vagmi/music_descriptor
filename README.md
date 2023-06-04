---
title: Music Descriptor
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 3.29.0
app_file: app.py
pinned: true
license: cc-by-nc-4.0
---

<!-- Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference -->

# Demo Introduction
This is an example of using the [MERT-v1-95M](https://huggingface.co/m-a-p/MERT-v1-95M) model as backbone to conduct multiple music understanding tasks with the universal represenation.

The tasks include EMO, GS, MTGInstrument, MTGGenre, MTGTop50, MTGMood, NSynthI, NSynthP, VocalSetS, VocalSetT. 

More models can be referred at the [map organization page](https://huggingface.co/m-a-p).

# Known Issues

## Audio Format Support

Theorectically, all the audio formats supported by [torchaudio.load()](https://pytorch.org/audio/stable/torchaudio.html#torchaudio.load) can be used in the demo. Theese should include but not limited to `WAV, AMB, MP3, FLAC`.

## Error Output

Due the **hardware limitation** of the machine hosting our demospecification (2 CPU and 16GB RAM), there might be `Error` output when uploading long audios. 

Unfortunately, we couldn't fix this in a short time since our team are all volunteer researchers.

We recommend to test audios less than 30 seconds or using the live mode if you are trying the [Music Descriptor demo](https://huggingface.co/spaces/m-a-p/Music-Descriptor) hosted online at HuggingFace Space.

This issue is expected to solve in the future by applying more community-support GPU resources or using other audio encoding strategy.

In the current stage, if you want to directly run the demo with longer audios, you could:
* clone this space `git clone https://huggingface.co/spaces/m-a-p/Music-Descriptor` and deploy the demo on your own machine with higher performance following the [official instruction](https://huggingface.co/docs/hub/spaces). The code will automatically use GPU for inference if there is GPU that can be detected by `torch.cuda.is_available()`. 
* develop your own application with the MERT models if you have the experience of machine learning.