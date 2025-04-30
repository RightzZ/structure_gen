import os
import subprocess
import time
import traceback
import re

RECOMMENDED_PARAMS = {
    "ara": {"render_num": 1, "data_num": 1, "resolution": 450, "radius": 0.1},
    "komatsuna": {"render_num": 1, "data_num": 1, "resolution": 480, "radius": 0.1},
    "amodal": {"render_num": 1, "data_num": 1, "resolution": 1024, "radius": 0},
    "plant": {"render_num": 1, "data_num": 1, "resolution": 1024, "radius": 25}
}

RECOMMENDED_SIZE = {
    "ara": 450,
    "amodal": 1024,
    "komatsuna": 480,
    "plant": 1024
}

def get_recommended_params(data_type):
    params = RECOMMENDED_PARAMS.get(data_type, {})
    return (
        params.get("render_num", 1),
        params.get("data_num", 1),
        params.get("resolution", 450),
        params.get("radius", 0.1),
    )

def get_recommended_size(species_type):
    params = RECOMMENDED_SIZE.get(species_type)
    return params

def run_model_training(dataset_dir, model_name):
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)
        yield "\n".join(log_lines)

    yield from log("🚀 Running Dataset Generation...")

    cmd = [
        "python", "-u", "train.py",
        "--dataroot", dataset_dir,
        "--name", model_name,
        "--gpu_ids", "0",
        "--model", "pix2pix",
        "--netG", "unet_256",
        "--direction", "AtoB",
        "--lambda_L1", "100",
        "--norm", "batch",
        "--display_id", "0",
        "--preprocess", "none",
        "--no_flip",
        "--no_html"
    ]

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                   cwd="./image_generation/pix2pix")

        last_line = None

        for line in process.stdout:
            if line:
                clean_line = line.strip().replace('\r', '')
                if clean_line == last_line:
                    continue
                last_line = clean_line

                yield from log(clean_line)

        process.wait()
        if process.returncode == 0:
            yield from log("✅ Finished successfully!")
        else:
            yield from log(f"❌ Process exited with code {process.returncode}")

    except Exception as e:
        yield from log(f"❌ Exception occurred: {str(e)}")



def run_dataset_generation(plant_pic_dir, plant_img_res, text_prompt, plant_species, train_mode):
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)
        yield "\n".join(log_lines)

    yield from log("🚀 Running Dataset Generation...")

    cmd = [
        "python", "make_dataset.py",
        "--img_res", str(plant_img_res),
        "--text_prompt", text_prompt,
        "--device_id", "0",
        "--mode", train_mode,
        "--src_dir", plant_pic_dir,
        "--species", plant_species
    ]

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd="./data_utility/pix2pix")

        last_line = None

        for line in process.stdout:
            if line:
                clean_line = line.strip().replace('\r', '')
                if clean_line == last_line:
                    continue
                last_line = clean_line

                yield from log(clean_line)

        process.wait()
        if process.returncode == 0:
            yield from log("✅ Finished successfully!")
        else:
            yield from log(f"❌ Process exited with code {process.returncode}")

    except Exception as e:
        yield from log(f"❌ Exception occurred: {str(e)}")

def run_blender_generator(output_path, data_type, render_num, resolution, data_num, radius):
    log_lines = []
    start_time = time.time()
    last_ping_time = start_time

    def log(msg):
        print(msg)
        log_lines.append(msg)
        yield "\n".join(log_lines)


    yield from log("🚀 Running blender...")

    try:
        os.makedirs(output_path, exist_ok=True)
    except Exception as e:
        yield from log(f"❌ Failed to create output directory: {e}")
        return

    blender_path = "../blender-3.1.0-linux-x64/blender"
    blender_script_path = "./mask_generation/make_template_segdata.py"

    if not os.path.exists(blender_path):
        yield from log(f"❌ Blender not found at: {blender_path}")
        return

    if not os.access(blender_path, os.X_OK):
        try:
            os.chmod(blender_path, 0o755)
        except Exception as e:
            yield from log(f"❌ Blender exists but is not executable, and chmod failed: {e}")
            return
        if not os.access(blender_path, os.X_OK):
            yield from log("❌ Blender is not executable even after chmod. Please check permissions.")
            return

    before_dirs = set(os.listdir(output_path))

    cmd = [
        "xvfb-run", blender_path, "-b", "-P", blender_script_path,
        "--",
        "--output_dir", output_path,
        "--data_type", data_type,
        "--render_num", str(render_num),
        "--resolution", str(resolution),
        "--data_num", str(data_num),
        "--radius", str(radius),
    ]

    process = subprocess.Popen(cmd)
    while True:
        retcode = process.poll()
        current_time = time.time()

        if current_time - last_ping_time >= 20:
            yield from log("🕒 Still running...")
            last_ping_time = current_time

        if retcode is not None:
            break
        time.sleep(1)

    elapsed = time.time() - start_time

    if process.returncode != 0:
        yield from log(f"❌ Blender failed with return code {process.returncode}. Elapsed: {elapsed:.2f}s")
        return

    after_dirs = set(os.listdir(output_path))
    new_dirs = sorted(after_dirs - before_dirs)

    if not new_dirs:
        yield from log("⚠️ Blender ran but no new directory was created.")
        return

    new_data_dir = os.path.join(output_path, new_dirs[0])

    if new_data_dir is None:
        yield from log(f"❌ Failed to get the generated file from Blender")
        return

    yield from log("🎉 Completed!")

def run_species_classification(src_dir, dst_dir):
    log_lines = []

    def log(msg):
        log_lines.append(msg)
        yield "\n".join(log_lines)

    yield from log("🚀 Running Classification...")

    cmd = [
        "python", "./data_utility/src/syn_plant/data_classification.py",
        "--src_dir", src_dir,
        "--dst_dir", dst_dir
    ]

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        for line in process.stdout:
            yield from log(line.rstrip())

        process.wait()
        if process.returncode == 0:
            yield from log("✅ Classification completed.")
        else:
            yield from log(f"❌ Classification failed with return code {process.returncode}.")

    except Exception as e:
        yield from log(f"❌ Exception occurred: {str(e)}")

def run_process_dataset(data_dir, output_dir, species, img_shape):
    log_lines = []

    def log(msg):
        log_lines.append(msg)
        yield "\n".join(log_lines)

    yield from log("🚀 Running GT mask generation...")

    cmd = [
        "python", "-u", "./data_utility/src/syn_plant/proc_data.py",
        "--data_dir", data_dir,
        "--output_dir", output_dir,
        "--species", species,
        "--image_shape", str(img_shape)
    ]

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        for line in process.stdout:
            if line:
                print(line.rstrip())
                yield from log(line.rstrip())

        process.wait()
        if process.returncode == 0:
            yield from log("✅ Finished successfully!")
        else:
            yield from log(f"❌ Process exited with code {process.returncode}")

    except Exception as e:
        traceback.print_exc()
        yield from log(f"❌ Exception occurred: {str(e)}")

def run_texturing(species, data_dir, bg_dir, img_res):
    log_lines = []

    def log(msg):
        log_lines.append(msg)
        yield "\n".join(log_lines)

    yield from log("Running Texturing...")

    cmd = [
        "python", "-u", "./texturing.py",
        "--gpu_id", "0",
        "--species", species,
        "--data_dir", data_dir,
        "--bg_dir", bg_dir,
        "--img_res", str(img_res)
    ]

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd="./mask_generation/proc/texturing")

        for line in process.stdout:
            if line:
                print(line.rstrip())
                yield from log(line.rstrip())

        process.wait()
        if process.returncode == 0:
            yield from log("✅ Finished successfully!")
        else:
            yield from log(f"❌ Process exited with code {process.returncode}")

    except Exception as e:
        traceback.print_exc()
        yield from log(f"❌ Exception occurred: {str(e)}")