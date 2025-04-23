import os
import sys

def main():
    input_exec = sys.argv[0]
    exec_dir =input_exec[:input_exec.rfind("/")+1]
    if not exec_dir[:2] == "./":
        exec_dir = "./" + exec_dir
    exec_command = exec_dir + "test_suite_execution" + " --config " + exec_dir + "test_suite_config.json"

    os.system(f"OMP_PROC_BIND=close OMP_PLACES=cores "+ exec_command)
    return 0

if __name__ == "__main__":
    main()