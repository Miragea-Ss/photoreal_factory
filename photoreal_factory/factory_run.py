# factory_run.py
import argparse
import sys
import signal

from engine.gpu_init import init_gpu
from engine.worker import run_job
from profiles.loader import load_profile


def _handle_terminate(signum, frame):
    # 工場停止シグナル用（Ctrl+C / SIGTERM）
    print(f"[FACTORY] Terminated by signal {signum}")
    sys.exit(0)


def main():
    # ===============================
    # Signal handling (本番運用必須)
    # ===============================
    signal.signal(signal.SIGINT, _handle_terminate)
    signal.signal(signal.SIGTERM, _handle_terminate)

    # ===============================
    # GPU 初期化（ここで一度だけ）
    # ===============================
    init_gpu()

    # ===============================
    # CLI
    # ===============================
    parser = argparse.ArgumentParser(prog="factory_run")
    parser.add_argument("--job", required=True, help="profile name (e.g. cinema_log)")
    parser.add_argument("--input", required=True, help="input directory")
    parser.add_argument("--output", required=True, help="output directory")
    args = parser.parse_args()

    # ===============================
    # プロファイルロード
    # ===============================
    profile = load_profile(args.job)

    # ===============================
    # 工場ジョブ実行
    # ===============================
    run_job(
        input_path=args.input,
        output_path=args.output,
        profile=profile
    )


if __name__ == "__main__":
    main()
