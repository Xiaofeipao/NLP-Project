"""Retired historical GVR ablation driver; no model or CUDA imports."""


def main():
    raise SystemExit(
        "This legacy GVR driver is retired. Use scripts/run_post_training.py for new studies. "
        "Historical protocol: commit 23183e2; exact sources: docs/legacy_gvr_drivers_23183e2.zip. "
        "No automatic protocol migration is performed."
    )


if __name__ == "__main__":
    main()
