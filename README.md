# Introduction
This repository contains code written to explore path tracking
technique for motion control of robotic manipulators. 

All experiments which were presented in [this
paper](https://arxiv.org/abs/1709.05101) can be found in this
repository. There are also additional experiments that did not make to
the paper, or were made after that.

# Installation
This repository has two hard prequesites, which are
[toppra](github.com/hungpham2511/toppra) and
[OpenRAVE](https://github.com/rdiankov/openrave).

Installation instructions for both can be found
[here](github.com/hungpham2511/toppra).

*Note that* `toppra` version v0.1.1 has been tested and should be
used. The lastest version might not work properly. The instruction for
installing `toppra` should be:
``` shell
git clone https://github.com/hungpham2511/toppra
cd toppra/
git checkout tags/v0.1.1 -b stable
pip install -r requirements.txt --user
pip install . -e --user
```

# Usage
The experiments can be ran using the following command
``` shell
following.icra18 tracking_exps
# For more details, do
following.icra18 -h
```

# Got questions?
Have a question, raise an Issue or send me an email.

