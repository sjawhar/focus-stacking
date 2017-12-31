# Focus Stacking Pipeline
This project attempts to reproduce [Wang and Chang’s 2011 paper](http://www.ece.drexel.edu/courses/ECE-C662/notes/LaplacianPyramid/laplacian2011.pdf) on using Laplacian pyramids to fuse images with different focal planes. It also explores simpler and faster alternatives and compares the trade-offs.

For the input images used in this project, please see the images/input folder.

# Running the project
Included in this project is a Dockerfile and docker-compose.yml. Just run `docker-compose up` to build and run the container, then go to `localhost:8888` to see the Jupyter notebook for this project.

However, matplotlib displays the images a bit weird. I'd recommend saving the images to a file instead. The CLI will do that for you easily. To use it, run `docker exec -it focus_stack bash` to get inside the container, then navigate to the `/home/jovyan/work` directory and run `python -m focus_stack -h` to get started.

# Languages and Packages Used
* Python 3
* OpenCV 3
* Numpy
* Scipy
