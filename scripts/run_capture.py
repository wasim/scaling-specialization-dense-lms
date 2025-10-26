import argparse

from sdlms.activations import collect_ffn_acts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", type=int, default=200_000)
    args = parser.parse_args()

    texts = ["Alice met Bob. " * 2000]
    layer_names = ["model.layers.10.mlp"]
    buffers = collect_ffn_acts(args.model, texts, layer_names)
    for name, value in buffers.items():
        print(name, value.shape)
