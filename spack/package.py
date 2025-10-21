"""
Copyright 2024 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Christos Matzoros, Toni Boehnlein, Pal Andras Papp, Raphael S. Steiner
"""

from spack.package import *


class Onestopparallel(CMakePackage):
    """OneStopParallel (OSP): This project aims to develop scheduling algorithms 
    for parallel computing systems based on the Bulk Synchronous Parallel (BSP) model. 
    The algorithms optimize the allocation of tasks to processors, taking into 
    account factors such as load balancing, memory constraints and communication overhead."""

    homepage = "https://github.com/Algebraic-Programming/OneStopParallel"
    git      = "https://github.com/Algebraic-Programming/OneStopParallel.git"

    maintainers = ['cmatzoros']

    version('master', branch='master')

    # Dependencies
    depends_on('cmake@3.12:', type='build')
    depends_on('boost@1.71.0:+graph+test', type=('build', 'link'))
    depends_on('eigen@3.4.0:', type=('build', 'link'))


    variant('openmp', default=True, description='Enable OpenMP support')

    def cmake_args(self):
        args = []

        # build the library version only
        args.append('-DCMAKE_BUILD_TYPE=Library')
        args.append('-DBUILD_TESTS=OFF')

        if '+openmp' in self.spec:
            args.append('-DOSP_DEPENDS_ON_OPENMP=ON')
        else:
            args.append('-DOSP_DEPENDS_ON_OPENMP=OFF')

        args.append('-DCMAKE_INSTALL_PREFIX={0}'.format(self.prefix))
        return args
