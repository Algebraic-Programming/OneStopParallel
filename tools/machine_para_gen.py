import subprocess    
import os
import sys
from pathlib import Path


def main():

    if len(sys.argv) == 4:
        
        num_proc = sys.argv[1]
        g = sys.argv[2] 
        l = sys.argv[3]
         

        file1 = open(f"p{num_proc}_g{g}_l{l}.txt", "a")
        file1.write("% BSP Data\n")
        file1.write(f"{num_proc} {g} {l}\n")
        file1.write("% NUMA Data")
        
        for i in range(int(num_proc)):
            for j in range(int(num_proc)):
            
                if i == j:
                    file1.write(f"\n{i} {j} 0")
                else:
                    file1.write(f"\n{i} {j} 1")        
        
        file1.close()
            
if __name__ == "__main__":
    main()
