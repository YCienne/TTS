import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "patch_recipe.py"
spec = importlib.util.spec_from_file_location("patch_recipe", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def make_recipe(tmp_path):
    (tmp_path / "utils").mkdir()
    main_py = tmp_path / "run_vits_finetuning.py"
    main_py.write_text(
        "def main():\n"
        "    model_outputs = model(\n"
        '        speaker_id=batch["speaker_id"],\n'
        "    )\n"
        "    other = batch2(\n"
        '        speaker_id=batch["speaker_id"],\n'
        "    )\n",
        encoding="utf-8",
    )
    plot_py = tmp_path / "utils" / "plot.py"
    plot_py.write_text(
        "def plot_a():\n"
        "    fig.canvas.draw()\n"
        "    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)\n"
        "    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))\n"
        "    return data\n"
        "\n"
        "def plot_b():\n"
        "    fig.canvas.draw()\n"
        "    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)\n"
        "    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))\n"
        "    return data\n",
        encoding="utf-8",
    )
    return main_py, plot_py


def test_patches_every_unsafe_speaker_id_access(tmp_path):
    main_py, _ = make_recipe(tmp_path)
    n = mod.patch_speaker_id(main_py)
    text = main_py.read_text(encoding="utf-8")
    assert n == 2
    assert 'batch["speaker_id"]' not in text
    assert text.count('speaker_id=batch.get("speaker_id")') == 2


def test_patches_every_matplotlib_call_and_keeps_rgb_shape(tmp_path):
    _, plot_py = make_recipe(tmp_path)
    n = mod.patch_matplotlib(plot_py)
    text = plot_py.read_text(encoding="utf-8")
    assert n == 2
    assert "tostring_rgb" not in text
    assert text.count("fig.canvas.buffer_rgba()") == 2
    assert text.count("get_width_height()[::-1] + (4,))[..., :3]") == 2


def test_idempotent_second_pass_is_a_no_op(tmp_path):
    main_py, plot_py = make_recipe(tmp_path)
    mod.patch_speaker_id(main_py)
    mod.patch_matplotlib(plot_py)
    n1 = mod.patch_speaker_id(main_py)
    n2 = mod.patch_matplotlib(plot_py)
    assert n1 == 0
    assert n2 == 0


def test_main_patches_both_files_under_recipe_dir(tmp_path, capsys):
    main_py, plot_py = make_recipe(tmp_path)
    import sys

    old_argv = sys.argv
    sys.argv = ["patch_recipe.py", "--recipe", str(tmp_path)]
    try:
        mod.main()
    finally:
        sys.argv = old_argv
    assert "patched 2 speaker_id site(s)" in capsys.readouterr().out
    assert 'batch["speaker_id"]' not in main_py.read_text(encoding="utf-8")
    assert "tostring_rgb" not in plot_py.read_text(encoding="utf-8")
