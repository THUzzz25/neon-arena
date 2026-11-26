"""Create the participant's own virtual environment using Python only."""
from pathlib import Path
import subprocess
import sys
import venv


def main():
    if sys.version_info < (3, 11):
        raise SystemExit("需要 Python 3.11 或更新版本，推荐 Python 3.13。")
    root = Path(__file__).resolve().parent
    environment = root / ".venv"
    venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run([str(python), "-m", "pip", "install", "--only-binary=:all:",
                    "-r", str(root / "requirements.txt")], check=True)
    print(f'\n准备完成，运行：\n"{python}" "{root / "run.py"}"')


if __name__ == "__main__":
    main()
