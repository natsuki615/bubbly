// p5.js interface to Google MediaPipe Landmark Tracking
// Combines face, hands, and bodies into one tracker.
// See https://mediapipe-studio.webapps.google.com/home
// Uses p5.js v.1.11.11 + MediaPipe v.0.10.22-rc.20250304
// By Golan Levin, revised as of 10/21/2025
//
// This app demonstrates how to access:
// - face points (e.g. clown nose)
// - hand points (e.g. thumb plum)
// - face metrics (e.g. jaw openness)
// - body pose
//----------------------------------------------------

let handLandmarks;
let myCapture;

let trackingConfig = {
  doAcquireHandLandmarks: true,
  maxNumHands: 2
};

let checkboxHand;

let dataset = [];
let currentLabel = 'r';
let recording = true;

//------------------------------------------
async function preload() {
  preloadTracker(); 
}

//------------------------------------------
function setup() {
  createCanvas(640, 480);
  myCapture = createCapture(VIDEO);
  myCapture.size(160, 120);
  myCapture.hide();

  initiateTracking();

  checkboxHand = createCheckbox('hand', trackingConfig.doAcquireHandLandmarks);
  checkboxHand.position(0, 20);
}

//------------------------------------------
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

//------------------------------------------
function getHandPoints() {
  if (!handLandmarks || !handLandmarks.landmarks) return;

  const nHands = handLandmarks.landmarks.length;
  for (let i = 0; i < nHands; i++) {
    let landmarks = handLandmarks.landmarks[i];
    let features = [];

    landmarks.forEach(p => {
      features.push(p.x);
		features.push(p.y);
    });

	  dataset.push({ class: currentLabel, features });
  }
}

//------------------------------------------
function keyPressed() {

  if (key === 's') {
    recording = false;
    // currentLabel = null;
    // print(`Dataset size: ${dataset.length}`);
	  print(dataset);
  }

  if (key === 'd') {
     saveJSON(dataset, `${currentLabel}_fixed.json`);
	  print("dataset saved");
  }
}

//------------------------------------------
function drawVideoBackground() {
  push();
  translate(width, 0);
  scale(-1, 1);
  tint(255, 255, 255, 72);
  image(myCapture, 0, 0, width, height);
  tint(255);
  pop();
}
