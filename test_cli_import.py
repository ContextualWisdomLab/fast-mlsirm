import subprocess
try:
    subprocess.check_output(["pytest"], stderr=subprocess.STDOUT)
    print("pytest successful")
except subprocess.CalledProcessError as e:
    print(f"pytest failed:\n{e.output.decode()}")
