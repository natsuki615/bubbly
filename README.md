# Bubbly 🫧

A real-time interactive game that leverages an ASL (American Sign Language) recognition model to form words. Watch as your hand signs come to life on screen, with letters automatically connecting when they form real words!

<!-- <img src="./assets/intro.gif" alt="Bubbly Demo" width="100%"> -->
<!-- <video src="./assets/intro.mp4" controls></video> -->
https://github.com/user-attachments/assets/67ce416a-1a92-4e97-8113-01cd1445976a


## What It Does

Bubbly uses your webcam to track your right hand in real-time and recognizes ASL finger-spelled letters. As you sign, letters appear on screen. When the letters you've signed form actual words or word prefixes, they automatically connect together visually, creating a satisfying and educational feedback loop.
This visual connection system helps learners to see immediate feedback on their signing, improve awareness of specific gestures and angles, and understand word formation in a unique, interactive way.

### Letters Can Form Words

When signed letters are spatially close and spell a valid word, they visually snap together.

<!-- <img src="./assets/make_words.gif" alt="Bubbly Demo" width="100%"> -->
<!-- <video src="./assets/make_words.mp4" controls></video> -->

https://github.com/user-attachments/assets/fabae53b-7e63-46bb-afb6-b05715244406


## Experiment — February 25, 2026

This update introduces **spatial hashing** for efficient neighbor detection among letter bubbles. Previously, finding neighboring letters required comparing every bubble against every other bubble — an O(n²) operation. With spatial hashing, the canvas is divided into a grid of cells, and each bubble is registered into the cell it occupies. Neighbor lookups then only check bubbles in adjacent cells, reducing the cost of neighbor queries dramatically as the number of bubbles grows.

This creates richer, more performant force interactions between letters, making the simulation feel more alive without sacrificing frame rate.

### Letters Have Noise Force

Each letter bubble is now influenced by a Perlin noise field, giving them organic, flowing movement rather than static or purely physics-driven behavior.

<!-- <img src="./assets/noise.gif" alt="Bubbly Demo" width="100%"> -->
<!-- <video src="./assets/noise.mp4" controls></video> -->

https://github.com/user-attachments/assets/439fe68c-aef5-4a77-8a4a-034ad1e3e814


### Density-Based Force

Letters now respond to local density: when too many letters crowd together, repulsive forces push them apart.

<!-- <img src="./assets/density.gif" alt="Bubbly Demo" width="100%"> -->
<!-- <video src="./assets/density.mp4" controls></video> -->

https://github.com/user-attachments/assets/9a5e6aeb-6209-4aec-b93b-e273fe27b521


### Merge(Weird) Force

If too many of the same word crowd together, they merge into one. This feature is still being implementated, as you can see the merge direction isn't correct

<!-- <video src="./assets/weird.mp4" controls></video> -->

https://github.com/user-attachments/assets/b492d882-268e-4db3-ab0a-6d7f765d466d

## Learning Curves
For this project, I also spent a lot of time looking into data visualization techniques such as Delauray triangulation and Vonoroi diagram, as well as particle simulation which involves GPU programming and texture manipulation. These were difficult ideas to wrap my head around, and my first attempt in implementing these features wasn't super successful. However I think these are important skills to have as I scale up my project in the future, especially for force calculations on the letters. 

## Tech Stack

- **p5.js** - Creative coding framework for visualization and interaction
- **TensorFlow.js** - Machine learning model for hand pose detection and ASL recognition
- **MediaPipe** - Real-time hand landmark detection

## Features

- **Real-time Recognition** - Instant ASL letter detection using your right hand
- **Smart Word Detection** - Automatically identifies when signed letters form valid words or prefixes
- **Visual Connections** - Letters dynamically connect on screen when they form meaningful combinations
- **Educational & Entertaining** - Learn ASL through interactive, visual feedback

<!-- ## Demo (Stage 1)
Visit https://youtu.be/7ZiGdUQ6UsY for a demo on the first version of this program

## Demo (Stage 2 - Image)
![Bubbly Demo](./assets/demo2.png) -->

## Prerequisites
To play with this program, you will need a  modern web browser, webcam access, and preferably good lighting for optimal hand tracking.

## How to Use

1. Position yourself in front of your webcam with good lighting
2. Hold your right hand in view of the camera
3. Sign ASL letters one at a time
4. Watch as letters appear on screen
5. When your letters form words or word prefixes, they'll automatically connect!
6. Keep practicing and watch the bubbles connect


## Future Plans

- Improve recognition accuracy
- Expand vocabulary recognition
- Implement custom word-based or letter-based forces


---

Hope you enjoy a little game of sign language!
