import subprocess    
import os
import sys
from pathlib import Path


def main():

    cluster_4 = "--cpus-per-task=4 --mem=16GB"
    cluster_8 = "--cpus-per-task=8 --mem=32GB" 
    cluster_16 = "--cpus-per-task=16 --mem=42GB"
    cluster_32 = "--cpus-per-task=32 --mem=20GB"    
    cluster_44 = "--cpus-per-task=44 --mem=64GB" 
    cluster_128 = "--cpus-per-task=128 --mem=16GB"

    if len(sys.argv) == 5:
    
        input_config = sys.argv[1]
        procs =  sys.argv[2]
        memory = sys.argv[3]
        node = sys.argv[4]
      
        cluster_flags = f"--cpus-per-task={procs} --mem={memory}GB --nodelist={node}"
        cluster_flags = f"--exclusive --mem=0GB --nodelist={node}"
        wdir = os.path.dirname(input_config)

        
        
        job_name = f"test_suite"
        #out_file = f"{wdir}/bspSchedule_{graph_path.stem}_{machine_path.stem}"
        command = f"./test_suite_execution --config {input_config}"
      
      #--output={out_file}.out --error={out_file}.err
        slurm_command = f"srun --job-name={job_name} --export=ALL,OMP_PROC_BIND=close,OMP_PLACES=cores,OMP_NUM_THREADS={procs} --partition=x86 --nodes=1 {cluster_flags} --time=336:15:00 {command}"
        print(f"Command: {command.split()}")
        #print(f"Slurm Command: {slurm_command.split()}") 
        #with open(f"{job_name}.stdout", 'w') as fp_stdout, open(f"{job_name}.stderr", 'w') as fp_stderr:
        proc = subprocess.run(slurm_command.split())
    
        #proc = subprocess.run(command.split())
    
        print(f"slurm exit with code {proc}")
        
    else:
        print("Invalid number of arguments, Usage: python3 cluster_test_suite.py <input_config> <procs> <memory> <node>")
                    
if __name__ == "__main__":
    main()
