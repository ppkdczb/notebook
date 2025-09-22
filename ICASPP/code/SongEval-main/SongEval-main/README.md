# 🎵 SongEval: A Benchmark Dataset for Song Aesthetics Evaluation

[![Hugging Face Dataset](https://img.shields.io/badge/HuggingFace-Dataset-blue)](https://huggingface.co/datasets/ASLP-lab/SongEval)
[![Arxiv Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2505.10793)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)  


This repository provides a **trained aesthetic evaluation toolkit** based on [SongEval](https://huggingface.co/datasets/ASLP-lab/SongEval), the first large-scale, open-source dataset for human-perceived song aesthetics. The toolkit enables **automatic scoring of generated song** across five perceptual aesthetic dimensions aligned with professional musician judgments.

---

## 🌟 Key Features

- 🧠 **Pretrained neural models** for perceptual aesthetic evaluation
- 🎼 Predicts **five aesthetic dimensions**:
  - Overall Coherence
  - Memorability
  - Naturalness of Vocal Breathing and Phrasing
  - Clarity of Song Structure
  - Overall Musicality
<!-- - 🧪 Supports **batch evaluation** for model benchmarking -->
- 🎧 Accepts **full-length songs** (vocals + accompaniment) as input
- ⚙️ Simple inference interface

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/ASLP-lab/SongEval.git
cd SongEval
pip install -r requirements.txt
```

## 🚀 Quick Start

- Evaluate a single audio file:

```bash
python eval.py -i /path/to/audio.mp3 -o /path/to/output
```

- Evaluate a list of audio files:

```bash
python eval.py -i /path/to/audio_list.txt -o /path/to/output
```

- Evaluate all audio files in a directory:

```bash
python eval.py -i /path/to/audio_directory -o /path/to/output
```

- Force evaluation on CPU  (⚠️ CPU evaluation may be significantly slower) :


```bash
python eval.py -i /path/to/audio.wav -o /path/to/output --use_cpu True
```


## 🙏 Acknowledgement
This project is mainly organized by the audio, speech and language processing lab [(ASLP@NPU)](http://www.npu-aslp.org/).

We sincerely thank the **Shanghai Conservatory of Music** for their expert guidance on music theory, aesthetics, and annotation design.
Meanwhile, we thank AISHELL to help with the orgnization of the song annotations.

<p align="center"> <img src="assets/logo.png" alt="Shanghai Conservatory of Music Logo"/> </p>

## 📑 License
This project is released under the CC BY-NC-SA 4.0 license. 

You are free to use, modify, and build upon it for non-commercial purposes, with attribution.

## 📚 Citation
If you use this toolkit or the SongEval dataset, please cite the following:
```
@article{yao2025songeval,
  title   = {SongEval: A Benchmark Dataset for Song Aesthetics Evaluation},
  author  = {Yao, Jixun and Ma, Guobin and Xue, Huixin and Chen, Huakang and Hao, Chunbo and Jiang, Yuepeng and Liu, Haohe and Yuan, Ruibin and Xu, Jin and Xue, Wei and others},
  journal = {arXiv preprint arXiv:2505.10793},
  year={2025}
}

```
