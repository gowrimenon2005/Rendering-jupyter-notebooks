# AE2230-I FM Lecture Notes

## Deploying the notebooks

These jupyter notebooks can be deployed in a sufficient manner using docker. By doing so the notebooks can be read in as a stand alone entity, mimicking the style of an e-book. This deployment can be done by executing the following procedure. Please read through the entire procedure before you begin. Furthermore, note that steps 1 and 2 only have to be performed the first time.

### Step 1. Install docker desktop 
1.1 Download the docker desktop application from https://www.docker.com/products/docker-desktop/

1.2 If you do not have an existing docker account, you can make one following this link https://hub.docker.com/signup

1.3 After installing the application, please sign into docker desktop using your account 

Note: Docker requires Linux to run. Windows users might get a pop up asking to install “Windows
Subsystem for Linux” (WSL). Follow the instructions that appear on screen.

### Step 2: Pull the docker image and create a container
To retrieve the docker image, first open either of the following applications based on your operating system:

• MacOS and Ubuntu: Open the Terminal application.

• Windows: Open the command prompt ‘cmd’

Run the following commands in the terminal window (in order):
```
docker login
docker pull carmvarriale/fm-lecture-notes
docker run --name fm-lecture-notes -p 8866:8866 -it carmvarriale/fm-lecture-notes
```
Note: Windows users might have to add ‘winpty’ before the prompt, such as: ```winpty docker ...```

For executions after the first one, it is going to be sufficient to simply start the container by running the following command:
```
docker start fm-lecture-notes
```

### Step 3: Accessing the notebooks locally
At this point, the docker container will be running and can be accessed either from the command line or from the the Docker desktop GUI. 

- For accessing it from the command line:
    - Open a new instance of your terminal and run the following command
    ```
    explorer "http://localhost:8866/"
    ```
    - If you used the ```docker run``` command, you can also press Ctrl+C then execute the ```docker start``` command from above, and then the ```explorer``` command

- For accessing it from the desktop GUI:
    1. Navigate to docker desktop, and go to 'containers' in the top left. 
    2. Make sure the docker container is running, if it is running correctly the first 'Actions' icon will be a square. (If it is not running it will be a triangle)
    3. Click on the port of the container. This will take you to localhost8866:8866 

NOTE: If you have other jupyter notebooks or voila applications running locally, you may not be able to connect to localhost8866:8866. Hence, when deploying the notebooks it is recommended to not have other jupyter servers open.

### Step 4: Closing the notebooks and stopping the container
To close the notebooks, simply close the web browser and stop the container. This can be done on docker desktop by clicking on the square icon under 'Actions' or from the command line by running the following command:
```
docker stop fm-lecture-notes
```

## Authors and acknowledgment
Carmine Varriale, Gowri Ramesh Menon

## Contributing
This is a living collection of documents that is supposed to be updated yearly with every new run of the course.
Any suggestion or contribution is appreciated.
