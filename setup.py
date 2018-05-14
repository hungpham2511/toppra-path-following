from setuptools import setup
setup(
    name="toppra-path-following",
    package_dir={"following": "following"},
    packages=['following',
              'following.console',
    ],
    entry_points = {
        "console_scripts": [
            'following.icra18=following.console.icra18:main',
        ]
    }
)
