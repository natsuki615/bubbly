// p5.js interface to Google MediaPipe Landmark Tracking
// Combines face, hands, and bodies into one tracker.
// See https://mediapipe-studio.webapps.google.com/home
// Uses p5.js v.1.11.11 + MediaPipe v.0.10.22-rc.20250304
// By Golan Levin, revised as of 10/21/2025

let handLandmarks;
let myCapture;

let trackingConfig = {
  doAcquireHandLandmarks: true,
  maxNumHands: 2
};

let checkboxHand;

let dataset = [];
let currentLabel = 'a';
let recording = false;

async function preload() {
  preloadTracker(); 
}

function setup() {
  createCanvas(640, 480);
  myCapture = createCapture(VIDEO);
  myCapture.size(160, 120);
  myCapture.hide();
  frameRate(30);

  initiateTracking();

  checkboxHand = createCheckbox('hand', trackingConfig.doAcquireHandLandmarks);
  checkboxHand.position(0, 20);
}

function draw() {
  background(255);
  drawVideoBackground();
  trackingConfig.doAcquireHandLandmarks = checkboxHand.checked();
  if (recording) {
	  print("is recording")
    getHandPoints();
  }
  drawHandPoints(); 
}

function getHandPoints() {
    if (!handLandmarks || !handLandmarks.landmarks) return;
    const nHands = handLandmarks.landmarks.length;
    for (let i = 0; i < nHands; i++) {
        let landmarks = handLandmarks.landmarks[i];
        print(landmarks)
        // sample landmarks (length 21)
        // 0: {x: 0.3277987241744995, y: 0.9421300888061523, z: 3.4070731658175646e-7}
        // 1: {x: 0.38314536213874817, y: 0.9353294968605042, z: -0.018506452441215515}
        // 2: {x: 0.4281298518180847, y: 0.8591523766517639, z: -0.022367872297763824}
        // 3: {x: 0.46030014753341675, y: 0.7912210822105408, z: -0.025854606181383133}

        let features = [];
        let wrist_x = landmarks[0].x;
        let wrist_y = landmarks[0].y;
        let wrist_z = landmarks[0].z;
        let min_x = 1;
        let max_x = 0;
        let min_y = 1;
        let max_y = 0;

        landmarks.forEach(p => {
            if (p.x < min_x) min_x = p.x;
            if (p.y < min_y) min_y = p.y;
            if (p.x > max_x) max_x = p.x;
            if (p.y > max_y) max_y = p.y;
        });

        let width = max_x - min_x;
        let height = max_y - min_y;
        let maxDist = max(width, height);
        landmarks.forEach(p => {
            // let norm_x = (p.x - wrist_x) / maxDist;
            // let norm_y = (p.y - wrist_y) / maxDist;
            // let norm_z = (p.z - wrist_z) / maxDist;
            features.push(p.x);
            features.push(p.y);
            features.push(p.y);
        });
        dataset.push({ label: currentLabel, features: features });
        // dataset = [ {
        //                  label: "A"
        //                  features: [x1, y1, x2, y2, ... ] lenfth 63
        //              } 
        //              {
        //                  label: "A"
        //                  features: [x1, y1, x2, y2, ... ]
        //              } 
    }
}

function keyPressed() {
    if (key === 's') {
        recording = !recording;
        // currentLabel = null;
        // print(`Dataset size: ${dataset.length}`);
        print(dataset);
    }
    if (key === 'd') {
        saveJSON(dataset, `${currentLabel}.json`);
        print("dataset saved");
    }
}

function drawVideoBackground() {
    push();
    translate(width, 0);
    scale(-1, 1);
    tint(255, 255, 255, 72);
    image(myCapture, 0, 0, width, height);
    tint(255);
    pop();
}
