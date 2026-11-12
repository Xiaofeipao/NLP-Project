"""Retired historical GVR test driver; retain its lightweight cache API."""


class CachedGeneration:
    """One physical G inference per grid; sessions retain logical G calls."""
    def __init__(self, model):
        self.model = model
        self.cache = {}
        self.last_usage = {}
        self.physical_calls = 0

    def predict(self, task, grid, candidate="", feedback="", check=lambda: None):
        key = (grid.layout_id, grid.start, grid.goal)
        if task == "generation" and key in self.cache:
            self.last_usage = {"input_tokens": 0, "output_tokens": 0}
            return self.cache[key]
        value = self.model.predict(task, grid, candidate, feedback, check)
        self.last_usage = dict(self.model.last_usage)
        if task == "generation":
            self.cache[key] = value
            self.physical_calls += 1
        return value


def main():
    raise SystemExit(
        "This legacy GVR driver is retired. Use scripts/run_post_training.py for new studies. "
        "Historical protocol: commit 23183e2; exact sources: docs/legacy_gvr_drivers_23183e2.zip. "
        "No automatic protocol migration is performed."
    )


if __name__ == "__main__":
    main()
