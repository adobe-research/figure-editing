# FigEdit: Benchmark for Scientific Figure Editing

[![License](https://img.shields.io/badge/License-Adobe%20Research-red.svg)](./LICENSE.md)
![Data](https://img.shields.io/badge/Dataset-30k%2B_Samples-blue.svg)
![Task](https://img.shields.io/badge/Task-Structured_Editing-green.svg)
![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
[![Paper](https://img.shields.io/badge/ArXiv-2512.00752-b31b1b.svg)](https://arxiv.org/abs/2512.00752)

## 📖 Introduction

**FigEdit** is a comprehensive benchmark designed to bridge the gap between pixel-level generative models and the structured nature of scientific charts.

Generative models, such as diffusion and autoregressive approaches, have demonstrated impressive capabilities in editing natural images. However, applying these tools to scientific charts rests on a flawed assumption: a chart is not merely an arrangement of pixels but a visual representation of structured data governed by a graphical grammar. Consequently, chart editing is not a pixel-manipulation task but a **structured transformation problem**.

To address this fundamental mismatch, we introduce **FigEdit**, a large-scale benchmark for scientific figure editing comprising over 30,000 samples. Grounded in real-world data, our benchmark is distinguished by its diversity, covering 10 distinct chart types and a rich vocabulary of complex editing instructions.

The benchmark is organized into five distinct and progressively challenging tasks:
1.  **Single Edits**: Atomic, one-step operations.
2.  **Multi Edits**: Complex instructions requiring sequential operations.
3.  **Conversational Edits**: Iterative refinement via dialogue.
4.  **Visual Guidance**: Edits driven by visual cues (Visual Circle / Tag-to-Image).
5.  **Style Transfer**: Applying aesthetic styles from a reference chart.

By releasing FigEdit, we aim to enable systematic progress in structure-aware figure editing, provide a common ground for fair comparison, and encourage future research on models that understand both the visual and semantic layers of scientific charts.

## 📊 Dataset Structure

The dataset supports five core tasks, distributed across specific JSON files in the repository:

| Task | Description | Source File | Mode Filter |
| :--- | :--- | :--- | :--- |
| **Single** | Single-turn instruction editing. | `instructions.json` | `mode="single"` |
| **Multi** | Multi-turn/Multi-step instruction editing. | `instructions.json` | `mode="multi"` |
| **Conv** | Conversational/Dialog-based editing. | `annotation_combo2.json` | N/A (Conversational structure) |
| **Visual** | Tag-to-Image / Visual Circle consistency. | `instructions_visual_circle_from_tags.json` | `mode="single_visual_circle_from_tags"` |
| **Transfer** | Reference-based Style Transfer. | `instructions_style_transfer_single.json` | `mode="style_transfer_single"` |

## 🧪 Evaluation

We provide a unified evaluation script (`evaluation/evalByedit.py`) that calculates both low-level vision metrics (SSIM, PSNR, LPIPS) and high-level semantic scores (CLIP, OCR, LLM Judge).

### 1. Preparing Your Model Outputs (Directory Structure)

To evaluate your model, you **must** organize the generated images into a specific directory structure. The script relies on this structure to match generated images with ground truth files.

Assume your output root is `model_outputs/` and your model name is `MyModel`. You must also copy the dataset JSON files into this root.

```text
model_outputs/
├── instructions.json                          # Copy from dataset root
├── instructions_visual_circle_from_tags.json  # Copy from dataset root
├── instructions_style_transfer_single.json    # Copy from dataset root
├── annotation_combo2.json                     # Copy from dataset root (for Conv)
├── bar/                                       # Chart Type
│   ├── 001/                                   # Figure ID
│   │   └── MyModel/                           # <-- Your Model Name Folder
│   │       ├── image_0.png                    # [Single/Multi] Results placed directly here
│   │       ├── Conv/                          # [Conv] Folder for Conversational task
│   │       │   └── seq_final_turn3.png        #      Must start with 'seq_final_'
│   │       ├── Visual/                        # [Visual] Folder for Visual task
│   │       │   └── image_0.png
│   │       └── Transfer/                      # [Transfer] Folder for Style Transfer task
│   │           └── image_0.png
│   └── 002/
│       └── ...
├── line/
│   └── ...
└── ...
```

#### Naming Rules
* **Single / Multi**: Place files directly in `<Model_Name>/`. The filename **must match** the Ground Truth filename (e.g., `image_0.png`).
* **Conv**: Place files in `<Model_Name>/Conv/`. Files must be named starting with `seq_final_` (e.g., `seq_final_turn3.png`).
* **Visual**: Place files in `<Model_Name>/Visual/`. The filename **must match** the Ground Truth filename.
* **Transfer**: Place files in `<Model_Name>/Transfer/`. The filename **must match** the Ground Truth filename.

### 2. Environment Setup

Install the required Python packages:
```bash
pip install torch torchvision clip-score lpips ssim-score
# Add other specific requirements if necessary
```

**LLM Judge Configuration (Optional):**
To enable the GPT-4o based evaluation (Instruction Following, Content Preservation, Quality), set the following environment variables:

```bash
export AZURE_OPENAI_API_KEY="your_key_here"
export AZURE_OPENAI_ENDPOINT="[https://your-endpoint.openai.azure.com/](https://your-endpoint.openai.azure.com/)"
export AZURE_OPENAI_API_VERSION="2024-12-01-preview"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o-mini-20240718"
```

### 3. Running the Evaluation

Use the `evalByedit.py` script to run the evaluation.

```bash
python evaluation/evalByedit.py \
  --out_root /path/to/model_outputs \
  --model_dir_name "MyModel" \
  --result_name "results_mymodel.json"
```

**Arguments:**
* `--out_root`: **(Required)** The root directory containing output images and instruction JSONs.
* `--model_dir_name`: **(Default: GPT-Image)** The name of the subfolder inside each figure directory where your images are stored.
* `--result_name`: **(Optional)** The name of the output JSON file containing metric scores.

**Flags to Disable Components:**
* `--disable_llm`: Skip GPT-4o evaluation (saves cost).
* `--disable_clip`: Skip CLIP similarity.
* `--disable_ocr`: Skip OCR evaluation.
* `--disable_[task]`: Skip specific tasks (e.g., `--disable_conv`, `--disable_transfer`, `--disable_visual`).

## 🏆 Benchmark Results

Below is a performance comparison of state-of-the-art models on the FigEdit benchmark.

* **Instr.**: Instruction Following Score (1-5)
* **Preserv.**: Content Preservation Score (1-5)
* **Qual.**: Image Quality Score (1-5)

| Task | Model | SSIM ↑ | LPIPS ↓ | CLIP ↑ | PSNR ↑ | OCR ↑ | Instr. ↑ | Preserv. ↑ | Qual. ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single** | Imagen 4 | **0.7726** | **0.4094** | 0.7781 | **13.04** | 0.0723 | 1.58 | 1.51 | 2.05 |
| | GPTImage | 0.7295 | 0.5383 | 0.8099 | 10.32 | 0.2054 | **3.47** | 1.71 | 2.45 |
| | InstructPix2Pix | 0.7211 | 0.4811 | 0.8328 | 11.02 | 0.2568 | 3.27 | 2.50 | 2.77 |
| | OmniGen2 | 0.7350 | 0.4705 | **0.8350** | 11.30 | **0.2620** | 3.35 | **2.55** | **2.85** |
| **Multi** | Imagen 4 | 0.6958 | 0.5549 | 0.7738 | **11.02** | 0.1069 | 1.26 | 1.32 | 2.15 |
| | GPTImage | 0.7017 | 0.5787 | **0.8070** | 9.73 | 0.2185 | 2.51 | 1.63 | 2.34 |
| | InstructPix2Pix | 0.6460 | 0.5204 | 0.8043 | 9.83 | 0.2584 | 2.48 | 2.00 | 2.51 |
| | OmniGen2 | **0.7100** | **0.5100** | 0.8220 | 10.15 | **0.2650** | **2.65** | **2.10** | **2.70** |
| **Conv.** | Imagen 4 | **0.7180** | **0.4923** | 0.7599 | **11.58** | 0.0698 | 1.35 | 1.23 | 2.11 |
| | GPTImage | 0.6732 | 0.5257 | **0.8525** | 10.66 | 0.1721 | **4.59** | **2.51** | **2.91** |
| | InstructPix2Pix | 0.6890 | 0.5075 | 0.8200 | 10.40 | 0.2540 | 2.90 | 2.25 | 2.65 |
| | OmniGen2 | 0.7050 | 0.4950 | 0.8280 | 10.80 | **0.2600** | 3.10 | 2.35 | 2.75 |
| **Visual** | Imagen 4 | **0.8420** | **0.5050** | 0.7600 | **13.10** | 0.1200 | 1.40 | 1.35 | 2.20 |
| | GPTImage | 0.8355 | 0.5207 | **0.8444** | 12.85 | **0.4665** | **2.39** | **3.16** | **3.95** |
| | InstructPix2Pix | 0.7380 | 0.5220 | 0.8190 | 10.90 | 0.2200 | 1.85 | 2.20 | 2.80 |
| | OmniGen2 | 0.7508 | 0.5236 | 0.8187 | 8.98 | 0.1806 | 1.19 | 1.85 | 2.74 |
| **Transfer** | Imagen 4 | **0.8500** | 0.4800 | 0.7700 | **14.00** | 0.1300 | 1.30 | 1.25 | 2.15 |
| | GPTImage | 0.8438 | 0.4934 | 0.8054 | 13.81 | **0.5092** | **3.06** | **3.57** | **4.16** |
| | InstructPix2Pix | 0.7960 | 0.5020 | **0.8160** | 12.90 | 0.2400 | 2.20 | 2.60 | 3.10 |
| | OmniGen2 | 0.8246 | **0.4376** | 0.8127 | 12.08 | 0.3147 | 1.53 | 2.14 | 2.64 |

## 📄 License

This project is licensed under the **Adobe Research License**. The material is available for **non-commercial research purposes only**. Please see the [LICENSE.md](./LICENSE.md) file for full terms and conditions.

## 📚 Citation

If you use **FigEdit** in your research, please cite [our ICLR 2026 paper](https://arxiv.org/pdf/2512.00752):

```bibtex
@misc{li2025chartsimageschallengesscientific,
      title={Charts Are Not Images: On the Challenges of Scientific Chart Editing}, 
      author={Shawn Li and Ryan Rossi and Sungchul Kim and Sunav Choudhary and Franck Dernoncourt and Puneet Mathur and Zhengzhong Tu and Yue Zhao},
      year={2025},
      eprint={2512.00752},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.00752}, 
}
```
