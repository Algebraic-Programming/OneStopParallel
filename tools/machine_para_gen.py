# Copyright 2024 Huawei Technologies Co., Ltd.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# @author Toni Boehnlein, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner


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
