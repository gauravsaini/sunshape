from __future__ import annotations

from sunshape import SunShapeConfig, SunShapeForCausalLM


def main() -> None:
    model, tokenizer, bundle = SunShapeForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-0.8B",
        traces_path="local_runs/traces/traces_by_layer_qwen08b.pt",
        sunshape_config=SunShapeConfig(
            mode="sunshape_base",
            bits_per_dim=4.0,
            layers=[3, 7, 11, 15, 19, 23],
        ),
        local_files_only=True,
    )
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=12)
    print("bundle layers:", bundle.layers)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
