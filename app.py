import gradio as gr
import os
import subprocess

from app_function import *

# Frontend
with gr.Blocks(title="Plant Generator") as demo:
    gr.Markdown("## Plant Generator 🌿")

    with gr.Tabs():

        with gr.Tab("1: Model Training 🧩"):

            with gr.Row():
                gr.Markdown("Dataset Generation")

            with gr.Row():
                with gr.Column():
                    plant_pic_dir = gr.Textbox(
                        label="Plant Picture Directory",
                        value="/home/structure_gen/plant_pic",
                        interactive=True
                    )
                    plant_img_res = gr.Number(label="Picture's Resolution", value=1024, precision=0, interactive=True)
                    text_prompt = gr.Textbox(
                        label="Text Prompt",
                        value="leaf",
                        interactive=True
                    )
                    plant_species = gr.Textbox(
                        label="Species name",
                        interactive=True
                    )
                    train_mode = gr.Dropdown(
                        choices=["train", "test"],
                        label="Train Mode",
                        value="train",
                        interactive=True
                    )
                    run_dataset_generation_button = gr.Button("🚀 Start Generating Dataset")

                    with gr.Row():
                        gr.Markdown("Model Training")

                    with gr.Row():
                        with gr.Column():
                            train_dataset_dir = gr.Textbox(
                                label="Dataset Directory",
                                interactive=True
                            )
                            model_name = gr.Textbox(
                                label="Model Name",
                                interactive=True
                            )
                            run_model_training_button = gr.Button("🚀 Start Training Model")

                with gr.Column():
                    log_output_0 = gr.Textbox(label="Log Output",
                                              lines=15,
                                              max_lines=15,
                                              interactive=False)

        with gr.Tab("2: Mask Generation 🌱"):

            with gr.Row():
                gr.Markdown("### Parameters")

            with gr.Row():

                with gr.Column():
                    blender_output_dir = gr.Textbox(
                        label="Output Directory(In Docker)",
                        value="/home/structure_gen/plant_mask_data",
                        interactive=True
                    )
                    data_type = gr.Dropdown(
                        choices=["amodal", "plant", "komatsuna", "ara"],
                        label="Species",
                        value="ara",
                        interactive=True
                    )
                    render_num = gr.Number(label="Number of Render Times", value=1, precision=0, minimum=1)
                    data_num = gr.Number(label="Number of Models", value=1, precision=0, minimum=1)
                    resolution = gr.Number(label="Resolution", value=450, precision=0, minimum=256)
                    radius = gr.Number(label="Camera Orbit Radius", value=0.1, precision=1)
                    run_blender_button = gr.Button("🚀 Start Blender Generating")

                with gr.Column():
                    log_output_1 = gr.Textbox(label="Log Output",
                                              lines=15,
                                              max_lines=15,
                                              interactive=False)

        with gr.Tab("3: Data Management 🗃️"):

            with gr.Row():
                gr.Markdown("### Dataset Classification")

            with gr.Row():

                with gr.Column():
                    blender_generated_dir = gr.Textbox(
                        label="Blender Generated Directory(In Docker)",
                        value="/home/structure_gen/plant_mask_data",
                        interactive=True
                    )
                    species_sorted_dir = gr.Textbox(
                        label="Species Sorted Directory(In Docker)",
                        value="/home/structure_gen/data_utility/src/syn_plant",
                        interactive=True
                    )
                    run_classification_button = gr.Button("🔖 Start Classification")

                    gr.Markdown("### Mask&Dataset Processing")
                    dataset_dir = gr.Textbox(
                        label="Dataset Saved Directory(In Docker)",
                        value="/home/structure_gen/data_utility/src/syn_plant",
                        interactive=True
                    )
                    gt_mask_dir = gr.Textbox(
                        label="Ground Truth Mask Generated Directory(In Docker)",
                        value="/home/structure_gen/data_utility/segmentation",
                        interactive=True
                    )
                    species_type = gr.Dropdown(
                        choices=["amodal", "plant", "komatsuna", "ara"],
                        label="Species",
                        value="ara",
                        interactive=True
                    )
                    img_shape = gr.Number(label="Image Shape", value=450, precision=0)
                    run_process_button = gr.Button("🚀 Start Dataset Processing")

                with gr.Column():
                    log_output_2 = gr.Textbox(label="Log Output",
                                              lines=15,
                                              max_lines=15,
                                              interactive=False)

        with gr.Tab("4: Texturing 🖼️"):

            with gr.Row():
                gr.Markdown("### Parameters")

            with gr.Row():

                with gr.Column():
                    texture_mask_dir = gr.Textbox(
                        label="Mask Directory(In Docker)",
                        value="/home/structure_gen/data_utility/src/syn_plant/plant",
                        interactive=True
                    )
                    bg_dir = gr.Textbox(
                        label="Background Directory(In Docker)",
                        value="/home/structure_gen/data_utility/src/background/cropped"
                    )
                    species_model = gr.Textbox(
                        label="Species",
                        value="hawthorn",
                        interactive=True
                    )
                    texture_img_res = gr.Number(label="Image Resolution", value=1024, precision=0, minimum=256)
                    run_texturing_button = gr.Button("🚀 Start Texturing")


                with gr.Column():
                    log_output_3 = gr.Textbox(label="Log Output",
                                              lines=15,
                                              max_lines=15,
                                              interactive=False)


    data_type.change(
        fn=get_recommended_params,
        inputs=[data_type],
        outputs=[render_num, data_num, resolution, radius]
    )

    run_dataset_generation_button.click(
        fn=run_dataset_generation,
        inputs=[plant_pic_dir, plant_img_res, text_prompt, plant_species, train_mode],
        outputs=[log_output_0]
    )

    run_model_training_button.click(
        fn=run_model_training,
        inputs=[train_dataset_dir, model_name],
        outputs=[log_output_0]
    )

    run_blender_button.click(
        fn=run_blender_generator,
        inputs=[blender_output_dir, data_type, render_num, resolution, data_num, radius],
        outputs=[log_output_1]
    )

    run_classification_button.click(
        fn=run_species_classification,
        inputs=[blender_generated_dir, species_sorted_dir],
        outputs=[log_output_2]
    )

    run_process_button.click(
        fn=run_process_dataset,
        inputs=[dataset_dir, gt_mask_dir, species_type, img_shape],
        outputs=[log_output_2]
    )
    run_texturing_button.click(
        fn=run_texturing,
        inputs=[species_model, texture_mask_dir, bg_dir, texture_img_res],
        outputs=[log_output_3]
    )

    species_type.change(
        fn=get_recommended_size,
        inputs=[species_type],
        outputs=[img_shape]
    )

if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=8888)
