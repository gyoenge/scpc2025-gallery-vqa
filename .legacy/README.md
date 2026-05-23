# SCPC 2025 AI Project
## Multiple-choice VQA Multimodal for Everyday Photos 

- Event: 2025 Samsung Collegiate Programming Challenge : AI (June-July 2025)
- Theme: Develop a **multimodal AI model** that selects the correct answer to **multiple-choice questions** based on various **everyday photos** stored in a user’s smartphone gallery
- Idea: Applying **parameter-efficient fine-tuning (LoRA)** and **partial quantization** to a **BLIP2-flan-t5** multimodal model using a **synthetically generated dataset**
- Individual Participation
- This repo is submission code of competition 

### Ranking

- My final private ranking is **18/1445**! [Leaderboard](https://dacon.io/competitions/official/236500/leaderboard)
<p align="center">
<img width="70%" alt="image" src="https://github.com/user-attachments/assets/41abe434-0ced-474d-b09c-80f39ad537ea" />
</p>

### Preview

<p align="center">
<img width="70%" alt="scpc_ai_multimodal_mockup" src="https://github.com/user-attachments/assets/25e7ed94-b9ee-49df-af7c-dc516fca9e6d" />
</p>

---

## Description

<p align="justify">
The core objective of this project was to develop a <b>multimodal model</b> that accurately answers <b>multiple-choice questions</b> about a given image, for which we implemented a comprehensive approach spanning <b>custom dataset generation</b>, <b>model optimization</b>, and a <b>strategic inference process</b>.
</p>

<p align="justify">
First, we built a <b>three-step pipeline</b> to generate a training dataset of 1,218 examples . In the first step, we used the <b>`Qwen/Qwen-1_8B`</b> model to generate scene prompts across various categories like <b>nature</b>, <b>travel</b>, and <b>food</b>. In the second step, these prompts were fed into the <b>`dreamlike-art/dreamlike-photoreal-2.0`</b> model to create realistic images. In the final step, we utilized the <b>`llava-1.5-7b-hf`</b> model to create a description, a multiple-choice question, and an answer pair for each generated image.
</p>

<p align="justify">
For modeling, we used the <b>`Salesforce/blip2-flan-t5-xl`</b> model as our base, which combines the <b>BLIP-2 architecture</b> with a <b>FLAN-T5-XL text decoder</b> . To ensure efficient training within limited resources, we applied two optimization techniques. First, through <b>partial quantization</b>, we applied <b>4-bit quantization</b> only to the T5 decoder, improving inference efficiency. Second, we used <b>LoRA (Low-Rank Adaptation)</b>, a <b>Parameter-Efficient Fine-Tuning (PEFT)</b> method, to freeze the existing model's weights and add <b>trainable adapters</b> only to specific layers of the Q-Former for training.
</p>

<p align="justify">
To improve accuracy during inference, we employed a <b>two-stage strategy</b>. The model first generates a <b>descriptive text</b> of the situation based on the image and question. It then makes the <b>final answer selection</b> by synthesizing this descriptive text, the original image, the question, and the multiple-choice options.
</p>

<p align="justify">
Through these strategies, our model achieved a score of <b>0.83262</b> on the public leaderboard, significantly outperforming the <b>baseline model</b>'s score of <b>0.30486</b> and the pure <b>`flan-t5`</b> model's score of <b>0.81298</b>. Ultimately, we achieved a <b>final score of 0.8344</b>, finishing <b>22nd</b> on the public leaderboard and <b>18th</b> on the private leaderboard.
</p>


### Environment 

#### System 

- OS: Linux
- GPU: 1x NVIDIA A100 SXM4 (40GB VRAM) 
- CUDA: 12.8

#### Python Environment 

- Python: 3.12.11
- All dependencies are managed via pip.

#### Key Libraries 

| Package                       | Version      |
| ----------------------------- | ------------ |
| torch                         | 2.7.1+cu128  |
| torchaudio                    | 2.7.1+cu128  |
| torchvision                   | 0.22.1+cu128 |
| transformers                  | 4.54.0       |
| accelerate                    | 1.9.0        |
| peft                          | 0.16.0       |
| bitsandbytes                  | 0.46.1       |
| trl                           | 0.19.1       |
| datasets                      | 4.0.0        |
| transformers-stream-generator | 0.0.5        |
| safetensors                   | 0.5.3        |
| einops                        | 0.8.1        |
| tiktoken                      | 0.9.0        |
| diffusers                     | 0.20.0       |
| numpy                         | 2.1.2        |
| pandas                        | 2.3.1        |
| pillow                        | 11.0.0       |
| tqdm                          | 4.67.1       |


Jupyter (Optional)

| Package         | Version |
| --------------- | ------- |
| ipykernel       | 6.30.0  |
| jupyter\_client | 8.6.3   |
| jupyter\_core   | 5.8.1   |

#### Installation

- Install dependencies with requirements.txt

  ```
  pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
  ```

- Install dependencies individually

  ```

  # Install PyTorch (CUDA 12.1)
  pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

  # Core libraries
  pip install transformers==4.54.0 accelerate==1.9.0 peft==0.16.0 bitsandbytes==0.46.1
  pip install trl==0.19.1 datasets==4.0.0 tqdm==4.67.1
  pip install transformers-stream-generator==0.0.5 einops==0.8.1 tiktoken==0.9.0
  pip install diffusers==0.20.0 safetensors==0.5.3

  # Utilities
  pip install numpy==2.1.2 pandas==2.3.1 pillow==11.0.0

  # Jupyter (optional)
  pip install ipykernel==6.30.0 jupyter_client==8.6.3 jupyter_core==5.8.1

  ```

### Run

#### Generate dataset

1. run `generate_scene_prompt.py`
2. run `generate_scene_image.py`
3. run `generate_question_answer.py`

#### Train & Inference

1. run `train.ipynb`
2. run `inference.ipynb`
