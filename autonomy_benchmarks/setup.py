# Copyright Thinking Cars GmbH
# SPDX-License-Identifier: Apache-2.0

import os
from glob import glob

from setuptools import find_packages, setup

package_name = "autonomy_benchmarks"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(include=[package_name, package_name + ".*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        (os.path.join("share", package_name), ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*launch.[pxy][yma]*")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="root",
    maintainer_email="root@todo.todo",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": ["autonomy_benchmarks = autonomy_benchmarks.autonomy_benchmarks:main"],
    },
)
