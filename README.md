# Visual Geometry - Match-Mover Project

This is the official repo for the Match Mover  Project for 703610 VO/1 VO Visuelle Geometrie 2020



## Demo

![Alt Text](./resources/result.gif)


## Report

A detailed report about this project can be found [here](./report/report.pdf).


## Getting Started

This project was built and executed with Python 3.7.2 and OpenCV 3.4.2. For the optimal experience please use this version. It might run on other python and OpenCV versions, however we can not garantuee it will.



In order to run the project clone the offical repository on git like:

```bash
git clone https://git.uibk.ac.at/csat8361/visualgeometry.git
```



Navigate into the VisualGeometry/src directory

```bash
cd VisualGeometry/src
```



### Prerequisites

To run the project you need the following python packages:

* numpy

* OpenCV 3.4.2.17

* pyyaml

* tqdm

* matplotlib



All of these packages can be obtained by running:

```bash
pip3 install -r requirements.txt
```



Execute the program:

```bash
python3 main.py
```



We provide intermediate results (camera calibration results and keypoint matching) to provide a fast initial run (it takes less than 5 min on MacBook Pro). In order to run all of the steps delete these files:
```bash
rm -rf ../resources/tmp/ ../resources/vid_to_img ../resources/calibration
```



## Authors

* **Simon Rüba** <simon.rueba@student.uibk.ac.at>

* **Daniel Kunis** <daniil.kunis@student.uibk.ac>
* **Florian Maier** <florian.Maier@student.uibk.ac>



If you have any questions regarding this project please do not hesitate to contact us via E-mail.



## License [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the  "Software"), to deal in the Software without restriction, including  without limitation the rights to use, copy, modify, merge, publish,  distribute, sublicense, and/or sell copies of the Software, and to  permit persons to whom the Software is furnished to do so, subject to  the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE  SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
