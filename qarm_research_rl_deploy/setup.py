"""Package definition for qarm_research_rl_deploy: entry points, config, launch and resources."""

from glob import glob

from setuptools import find_packages, setup

package_name = "qarm_research_rl_deploy"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages",
            ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        # The reach policy is not shipped here; it is loaded from the run_dir
        # passed at launch.
        ("share/" + package_name + "/resource", [
            "resource/yolov8s-shapes-seg.pt",
        ]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*launch.py")),
    ],
    install_requires=["setuptools", "numpy", "torch", "pyyaml"],
    # colcon runs pytest only when it is a declared test dependency. setuptools
    # 72 and later ignore tests_require, so it is declared as an extra.
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="Zinan Cen",
    maintainer_email="zinan.cen@quanser.com",
    description="Deployment nodes for the QArm Research reach policy.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "reach_action_server = qarm_research_rl_deploy.reach_action_server:main",
            "command_bridge = qarm_research_rl_deploy.command_bridge:main",
            "pick_place_planner = qarm_research_rl_deploy.pick_place_planner:main",
            "vision_node = qarm_research_rl_deploy.vision_node:main",
        ],
    },
)
