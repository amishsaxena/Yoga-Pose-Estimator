
# AsanAI (Yoga Pose Estimator)

Using deep learning to predict 80+yoga poses from images and videos at 70+ % accuracy.
## Installation


Clone the repo into your local system

```bash
  git clone https://github.com/amishsaxena/Yoga-Pose-Estimator.git
  cd Yoga-Pose-Estimator
```
Use a virtual environment to install the dependencies and requirements
<br>
(Conda env are preferred)
```bash
    conda create --name <env_name> --file requirements
    conda activate <env_name>
```
Make sure the versions of the packages are same as that in ```requirements``` file.

NOTE : It is recommended that you have CUDA and CuDNN installed and run using the GPU. Running using the CPU may result in poor performance.

## Run Locally

Go to the project directory.

```bash
  cd my-project-directory
```
For running on a **live webcam** in the default browser using Streamlit.

```bash
  python3 pose.py
```

For running on a **recorded video** in the default browser using Streamlit.

```bash
  python3 pose.py <relative-location-of-video>
```

For running on a **recorded video** and saving the output as another video.

(Currently only supporting ```.avi``` format)
```bash
  python3 recorded.py <relative-location-of-video> <"name-of-saved-video".avi>
```


## Documentation

[Presentation Link](https://drive.google.com/file/d/1-C8GudRDtf3ZuoL3mTM1PH2wr45SugHb/view?usp=sharing)

[Documentation link](https://github.com/amishsaxena)
(To be updated)
## Demo

The demo can be found linked in the Presentation in the `Documentation` section.

  
## Authors

- [@amishsaxena](https://www.github.com/amishsaxena)
- [@jerryjohnthomas](https://github.com/JerryJohnThomas)
  
## Contributing

Contributions are always welcome!


  