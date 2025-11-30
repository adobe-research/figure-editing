import traceback
from pathlib import Path
import matplotlib.pyplot as plt
import re
import json
import numpy as np

# =====================
# CONFIGURATION
# =====================
class Config:
    INPUT_BASE = "/sensei-fs-3/users/lli1/results"
    OUTPUT_SUBDIR = "ChartEdit"
    TASK_TYPES = ["bar", "line", "pie"]

def sanitize_code(code: str) -> str:

    code = re.sub(r'//.*', '', code)
    # null → None
    code = re.sub(r'\bnull\b', 'None', code)
    # true/false
    code = re.sub(r'\btrue\b', 'True', code)
    code = re.sub(r'\bfalse\b', 'False', code)
    return code

# =====================
# CODE EXTRACTION FROM TEXT
# =====================
def extract_code(text):
    """
    Extract Python code block from markdown-style explanations.
    """
    code_blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    return text.strip()


# =====================
# FUNCTION TO EXECUTE CODE AND SAVE IMAGE
# =====================
def render_and_save(code_path):
    """
    Render a matplotlib chart from the provided Python script and save it as a PNG image.
    The image is saved in the same directory as the code file, with the same filename but a .png extension.
    """
    try:
        raw_text = code_path.read_text(encoding="utf-8")
        code = extract_code(raw_text)
        code = sanitize_code(code)

        exec_globals = {"plt": plt}
        exec_globals = {
            "plt": plt,
            "json": json,
            "np": np,
        }
        exec(code, exec_globals)

        output_image_path = code_path.with_suffix(".png")
        plt.savefig(output_image_path)
        plt.close()

        return True
    except Exception:
        #print(f"Failed to render {code_path}")
        traceback.print_exc()
        return False


# =====================
# MAIN ENTRY FUNCTION
# =====================
def render_all_charts(input_base=None, task_types=None, output_subdir=None):
    """
    Render all .txt files containing matplotlib code under the specified input directory.

    Parameters:
    - input_base: Root input directory (optional, default from Config).
    - task_types: List of task type folder names (optional, default from Config).
    - output_subdir: Subdirectory containing the .txt files (optional, default from Config).
    """
    config = Config()
    input_base = input_base or config.INPUT_BASE
    task_types = task_types or config.TASK_TYPES
    output_subdir = output_subdir or config.OUTPUT_SUBDIR

    total_files = 0
    success_files = 0

    for task in task_types:
        task_root = Path(input_base) / task
        if not task_root.exists():
            continue

        for subfolder in sorted(task_root.iterdir()):
            txt_dir = subfolder / output_subdir
            if not txt_dir.exists():
                continue

            for txt_file in txt_dir.glob("*.txt"):
                #print(f"Rendering {txt_file}")
                total_files += 1
                if render_and_save(txt_file):
                    success_files += 1

    success_rate = (success_files / total_files) * 100 if total_files > 0 else 0
    print(f"\nTotal files: {total_files}")
    print(f"Successfully rendered: {success_files}")
    print(f"Success rate: {success_rate:.2f}%")

    return {
        "total": total_files,
        "success": success_files,
        "success_rate": success_rate,
    }


# =====================
# SCRIPT ENTRY POINT
# =====================
if __name__ == "__main__":
    render_all_charts()
