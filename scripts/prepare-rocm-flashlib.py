"""Build a local flashlib wheel using ROCm Triton, without unused CuTe dependencies.

FreeToken imports flashlib's Triton slot cache. The upstream wheel requires the
NVIDIA distribution named ``triton``, which overwrites ``triton-rocm``'s module.
Keep the library code intact and mark the adjusted wheel with a local version.
"""

from email import policy
from email.parser import BytesParser
from pathlib import Path
import subprocess
import sys
import tempfile

from packaging.requirements import Requirement


def prepare(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory)
        subprocess.run(
            [sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
             "--dest", str(work), "flashlib==0.3.0"], check=True,
        )
        wheel = next(work.glob("*.whl"))
        subprocess.run([sys.executable, "-m", "wheel", "unpack", str(wheel), "-d", str(work)], check=True)
        source = work / "flashlib-0.3.0"
        dist_info = source / "flashlib-0.3.0.dist-info"
        metadata_path = dist_info / "METADATA"
        metadata = BytesParser(policy=policy.compat32).parsebytes(metadata_path.read_bytes())
        requires = metadata.get_all("Requires-Dist", [])
        del metadata["Requires-Dist"]
        replaced = False
        for dependency in requires:
            req = Requirement(dependency)
            if req.name == "triton":
                metadata["Requires-Dist"] = "triton-rocm==3.6.0"
                replaced = True
            elif req.name != "nvidia-cutlass-dsl":
                metadata["Requires-Dist"] = dependency
        if not replaced:
            raise RuntimeError("flashlib metadata changed: expected a triton requirement")
        metadata.replace_header("Version", "0.3.0+rocm")
        metadata_path.write_bytes(metadata.as_bytes())
        dist_info.rename(source / "flashlib-0.3.0+rocm.dist-info")
        # wheel pack regenerates RECORD hashes after the metadata change.
        subprocess.run([sys.executable, "-m", "wheel", "pack", str(source), "-d", str(output)], check=True)


if __name__ == "__main__":
    prepare(Path(sys.argv[1]))
