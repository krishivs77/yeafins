"""Fail when a production environment contains GPU runtime packages."""

from importlib.metadata import distributions

BLOCKED_NAMES = ("triton",)
BLOCKED_PREFIXES = ("cuda-", "nvidia-")


def unexpected_gpu_packages() -> list[str]:
    names = {
        distribution.metadata["Name"]
        for distribution in distributions()
        if distribution.metadata["Name"]
    }
    return sorted(
        name
        for name in names
        if name.lower() in BLOCKED_NAMES or name.lower().startswith(BLOCKED_PREFIXES)
    )


def main() -> None:
    unexpected = unexpected_gpu_packages()
    if unexpected:
        raise SystemExit("Unexpected GPU dependencies installed: " + ", ".join(unexpected))
    print("CPU-only runtime dependency check passed.")


if __name__ == "__main__":
    main()
