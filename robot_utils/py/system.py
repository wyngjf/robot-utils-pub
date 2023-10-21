import subprocess
from rich.console import Console

CONSOLE = Console(width=150)


def run(cmd: str, capture_output: bool = True, working_dir: str = None):
    """
    run system command
    Args:
        cmd:
        capture_output: to capture the output of the command
        working_dir: change to the working directory and run script from there

    Returns:

    """
    out = subprocess.run(cmd, capture_output=capture_output, shell=True, cwd=working_dir)
    if out.returncode != 0:
        CONSOLE.rule("[bold red] ERROR", style="red")
        CONSOLE.print(f"[bold red]Command failed: {cmd}")
        CONSOLE.rule(style="red")
        CONSOLE.print(out.stderr.decode("utf-8"))

    return out if out.stdout is None else out.stdout.decode("utf-8")
