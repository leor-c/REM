from pathlib import Path

from src.models.tokenizer.lpips import get_ckpt_path


def get_lpips():
    project_root = Path.cwd()

    ckpt_path = project_root / "cache" / "iris" / "tokenizer_pretrained_vgg"
    if not ckpt_path.exists():
        ckpt_path.mkdir(parents=True)

    get_ckpt_path('vgg_lpips', ckpt_path)


if __name__ == '__main__':
    get_lpips()



