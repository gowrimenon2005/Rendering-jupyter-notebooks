# AE2230-I FM Lecture Notes



## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.tudelft.nl/fpp-fm/fm-lecture-notes.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.tudelft.nl/fpp-fm/fm-lecture-notes/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Automatically merge when pipeline succeeds](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thank you to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README
Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Deploying the notebooks

These jupyter notebooks can be deployed in a sufficient manner using docker. By doing so the notebooks can be read in as a stand alone entity, mimicing the style of an e-book. This deployment can be done by executing the following. Please read through the entire procedure before you begin. Furthermore, note that steps 1 and 2 only have to be performed the first time! 

### Step 1. Install docker desktop 
1.1 Download the docker desktop application from https://www.docker.com/products/docker-desktop/

1.2 If you do not have an exisitng docker account, you can make one following this link https://hub.docker.com/signup

1.3 After installing the application, please sign into docker desktop using your account 

Note: Docker requires Linux to run. Windows users might get a pop up asking to install “Windows
Subsystem for Linux” (WSL). Follow the instructions that appear on screen.

### Step 2: Pull the docker image
To retrieve the docker image, first open either of the following applications based on your operating system:

• MacOS and Ubuntu: Open the Terminal application.

• Windows: Open the command prompt ‘cmd’

Run the following commands in the terminal window (in order):
```
docker login
```

```
docker pull carmvarriale/fm-lecture-notes
```

```
docker run -p 8866:8866 -it carmvarriale/fm-lecture-notes
```
Note: Windows users might have to add ‘winpty’ before the prompt, such as:
winpty docker ...

### Step 3: Running the notebooks locally
3.1 At this point, the docker container will be running and can be accessed more easily from docker desktop. Hence, navigate back to docker desktop, and go to 'containers' in the top left. 

3.2 Make sure the docker container is running, if it is running correctly the first 'Actions' icon will be a square. (If it is not running it will be a triangle)

3.3 Click on the port of the container. This will take you to localhost8866:8866 

NOTE: If you have other jupyter notebooks or voila applications running locally, you may not be able to connect to localhost8866:8866. Hence, when deploying the notebooks it is reccommended to not have other jupyter servers open.

3.4 To close the notebooks, simply close the web browser and stop the container. This can be done on docker desktop by clickling on the square icon under 'Actions'

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
