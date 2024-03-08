import subprocess

command = "python3 compiler.py build_ext --inplace"  # Example command, you can replace it with any command you want

# Execute the command
output = subprocess.check_output(command, shell=True, universal_newlines=True)
print(output)
