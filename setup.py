from setuptools import setup
setup(
    name="toppra-path-following",
    package_dir={"following": "following"},
    packages=['following',
              'following.console',
    ],
    entry_points = {
        "console_scripts": [
            'following.icra18.robustsets=following.console.compare_robust_controllable_sets:main',
            'following.icra18.tracking=following.console.compare_tracking_expm:main'
        ]
    }
)
