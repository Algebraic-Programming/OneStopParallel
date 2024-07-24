
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