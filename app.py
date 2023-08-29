import json
import os
import shutil
import requests

import gradio as gr
from huggingface_hub import Repository
from text_generation import Client

from share_btn import community_icon_html, loading_icon_html, share_js, share_btn_css

HF_TOKEN = os.environ.get("HF_TOKEN", None)

API_URL = "https://api-inference.huggingface.co/models/codellama/CodeLlama-7b-hf"


FIM_PREFIX = "<PRE> "
FIM_MIDDLE = " <MID>"
FIM_SUFFIX = " <SUF>"

FIM_INDICATOR = "<FILL_ME>"

EOS_STRING = "</s>"
EOT_STRING = "<EOT>"

theme = gr.themes.Monochrome(
    primary_hue="indigo",
    secondary_hue="blue",
    neutral_hue="slate",
    radius_size=gr.themes.sizes.radius_sm,
    font=[
        gr.themes.GoogleFont("Open Sans"),
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
    ],
)

client = Client(
    API_URL,
    headers={"Authorization": f"Bearer {HF_TOKEN}"},
)


def generate(
    prompt, temperature=0.9, max_new_tokens=256, top_p=0.95, repetition_penalty=1.0,
):

    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)
    fim_mode = False

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=42,
        max_length=500
    )

    if FIM_INDICATOR in prompt:
        fim_mode = True
        try:
            prefix, suffix = prompt.split(FIM_INDICATOR)
        except:
            raise ValueError(f"Only one {FIM_INDICATOR} allowed in prompt!")
        prompt = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"

    
    stream = client.generate_stream(prompt, **generate_kwargs)
    

    if fim_mode:
        output = prefix
    else:
        output = prompt

    previous_token = ""
    for response in stream:
        if any([end_token in response.token.text for end_token in [EOS_STRING, EOT_STRING]]):
            if fim_mode:
                output += suffix
                yield output
                return output
                print("output", output)
            else:
                return output
        else:
            output += response.token.text
        previous_token = response.token.text
        yield output
    return output


examples = [
    "X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.1)\n\n# Train a logistic regression model, predict the labels on the test set and compute the accuracy score",
    "def primes(n: int) -> list(int):",
    "Poor English: She no went to the market. Corrected English:",
    "class Circle():\n \"\"\" <FILL_ME>",
    "def towers_of_hanoi():",
]


def process_example(args):
    for x in generate(args):
        pass
    return x


css = ".generating {visibility: hidden}"

monospace_css = """
#q-input textarea {
    font-family: monospace, 'Consolas', Courier, monospace;
}
"""


css += share_btn_css + monospace_css + ".gradio-container {color: black}"

description = """
<div style="text-align: center;">
    <h1> 🦙 Code Llama Playground</h1>
</div>
<div style="text-align: left;">
    <p>This is a demo to generate text and code with the following <a href="https://huggingface.co/codellama/CodeLlama-13b-hf">Code Llama model (13B)</a>. Please note that this model is not designed for instruction purposes but for code completion. If you're looking for instruction or want to chat with a fine-tuned model, you can use [this demo instead](https://huggingface.co/spaces/codellama/codellama-13b-chat). You can learn more about the model in the <a href="https://huggingface.co/blog/codellama/">blog post<\a> or <a href="https://huggingface.co/papers/2308.12950">paper<\a></p>
</div>
"""

with gr.Blocks(theme=theme, analytics_enabled=False, css=css) as demo:
    with gr.Column():
        gr.Markdown(description)
        with gr.Row():
            with gr.Column():
                instruction = gr.Textbox(
                    placeholder="Enter your code here",
                    lines=5,
                    label="Input",
                    elem_id="q-input",
                )
                submit = gr.Button("Generate", variant="primary")
                output = gr.Code(elem_id="q-output", lines=30, label="Output")
                with gr.Row():
                    with gr.Column():
                        with gr.Accordion("Advanced settings", open=False):
                            with gr.Row():
                                column_1, column_2 = gr.Column(), gr.Column()
                                with column_1:
                                    temperature = gr.Slider(
                                        label="Temperature",
                                        value=0.1,
                                        minimum=0.0,
                                        maximum=1.0,
                                        step=0.05,
                                        interactive=True,
                                        info="Higher values produce more diverse outputs",
                                    )
                                    max_new_tokens = gr.Slider(
                                        label="Max new tokens",
                                        value=256,
                                        minimum=0,
                                        maximum=8192,
                                        step=64,
                                        interactive=True,
                                        info="The maximum numbers of new tokens",
                                    )
                                with column_2:
                                    top_p = gr.Slider(
                                        label="Top-p (nucleus sampling)",
                                        value=0.90,
                                        minimum=0.0,
                                        maximum=1,
                                        step=0.05,
                                        interactive=True,
                                        info="Higher values sample more low-probability tokens",
                                    )
                                    repetition_penalty = gr.Slider(
                                        label="Repetition penalty",
                                        value=1.05,
                                        minimum=1.0,
                                        maximum=2.0,
                                        step=0.05,
                                        interactive=True,
                                        info="Penalize repeated tokens",
                                    )
                                    
                gr.Examples(
                    examples=examples,
                    inputs=[instruction],
                    cache_examples=False,
                    fn=process_example,
                    outputs=[output],
                )

    submit.click(
        generate,
        inputs=[instruction, temperature, max_new_tokens, top_p, repetition_penalty],
        outputs=[output],
    )
demo.queue(concurrency_count=16).launch(debug=True, server_name='0.0.0.0', server_port=8080)

