import subprocess
import os

# Trick to avoid it running with pytest
__test__ = False

l = [
    "01-RHF_based_afqmc.py",
    "02-RCCSD_based_afqmc.py",
    "03-UHF_based_afqmc.py",
    "04-UCCSD_based_afqmc.py",
    #"05-afqmc_on_GPU.py",
    "06-afqmc_with_MPI.py",
    "07-afqmc_usual_keywords.py",
]

EXAMPLES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "examples"))

def check_example(fname):
    path = os.path.join(EXAMPLES_DIR, fname)

    # Splitting
    print(f"\nRunning {fname}...\n{'=' * 60}", flush=True)

    # Run
    process = subprocess.Popen(
        ["python", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # To print output in real-time
    output_lines = []
    for line in process.stdout:
        print(line, end='')
        output_lines.append(line)

    process.wait()
    assert process.returncode == 0, f"\n{fname} Test failed:\n{''.join(output_lines)}"

def test_examples():
    for fname in l:
        check_example(fname)

if __name__ == "__main__":
    test_examples()
