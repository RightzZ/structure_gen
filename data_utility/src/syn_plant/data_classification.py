import os
import shutil
import argparse

def run_classification(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)

    for dir_name in os.listdir(src_dir):
        src_data_dir = os.path.join(src_dir, dir_name)
        if not os.path.isdir(src_data_dir) or not dir_name.startswith("data_"):
            continue

        species = dir_name.split("_")[-1]
        dst_species_dir = os.path.join(dst_dir, species)
        os.makedirs(dst_species_dir, exist_ok=True)

        for sub_dir_name in os.listdir(src_data_dir):
            src_sub_dir = os.path.join(src_data_dir, sub_dir_name)
            if not os.path.isdir(src_sub_dir):
                continue

            dst_sub_dir = os.path.join(dst_species_dir, sub_dir_name)
            os.makedirs(dst_sub_dir, exist_ok=True)

            if sub_dir_name in ["binary", "amodal_seg"]:
                for inner_id in os.listdir(src_sub_dir):
                    src_inner = os.path.join(src_sub_dir, inner_id)
                    dst_inner = os.path.join(dst_sub_dir, inner_id)

                    if os.path.exists(dst_inner):
                        continue

                    shutil.copytree(src_inner, dst_inner)
                    print(f"📂 Copy {species}: {sub_dir_name}/{inner_id}")

            else:
                for file_name in os.listdir(src_sub_dir):
                    src_file = os.path.join(src_sub_dir, file_name)
                    dst_file = os.path.join(dst_sub_dir, file_name)

                    if os.path.exists(dst_file):
                        continue

                    shutil.copy2(src_file, dst_file)
                    print(f"📄 Copy {species}: {sub_dir_name}/{file_name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", type=str, required=True)
    parser.add_argument("--dst_dir", type=str, required=True)
    args = parser.parse_args()

    run_classification(args.src_dir, args.dst_dir)

if __name__ == "__main__":
    main()