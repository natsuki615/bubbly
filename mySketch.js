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
// Don't change the names of these global variables.
let handLandmarks;
let poseLandmarks;
let faceLandmarks;
let myCapture;
let currentLabel;
let letters = [];

//----------------------------------------------------
// For landmarks you want, set to true; set false the ones you don't.
// You'll get best performance with just one or two sets of landmarks.
// (Note, trackers set to false on startup can't be enabled later.)
let trackingConfig = {
  doAcquireHandLandmarks: true,
  doAcquirePoseLandmarks: false,
  doAcquireFaceLandmarks: false,
  doAcquireFaceMetrics: false,
  poseModelLiteOrFull: "full", /* "lite" (3MB) or "full" (6MB) */
  cpuOrGpuString: "GPU", /* "GPU" or "CPU" */
  maxNumHands: 2,
  maxNumPoses: 1,
  maxNumFaces: 1,
};

let ASLmodel;
let dataset;
let video;
let hands;
let allData = [];

const LABELS = "abcdefghijklmnopqrstuvwxyz".split("");

///matterjs------------------------------
const Engine = Matter.Engine,
      World = Matter.World,
      Bodies = Matter.Bodies;

let engine;
let world;
let seeds = [];
let from = "#3A499B";
let to = "#BDE7A8"; 

class Letter {
	constructor(x, y, l, s){
		this.l = l;
		this.s = s;
		this.body = Bodies.circle(x, y, s, { 
			restitution: 0.5, 
			friction: 0.2 });
      World.add(world, this.body);
	}
	show() {
		let pos = this.body.position;
		let angle = this.body.angle;
		let scale = map(pos.y, 0, height, 0,1);
		let col = lerpColor(from, to, scale);
	   push();
	   translate(pos.x, pos.y);
		rotate(angle);
	   fill("#23f758");
	   noStroke();
		textSize(this.s);
		text(this.l, 0,0);
		pop();
	}
}

function preload() {
	for (let l of LABELS) {
		let data = loadJSON(`alphabets/${l}_fixed.json`);
			allData = allData.concat(data);
	} 
	dataset = allData;
	preloadTracker();
}

async function setup() {
	createCanvas(windowWidth, windowHeight);
	frameRate(24);
	textFont('Courier New');
	await tf.setBackend('webgl');
	await tf.ready();
	print("Backend:", tf.getBackend());

	myCapture = createCapture(VIDEO);
	myCapture.size(160,120); 
	myCapture.hide();

	// ----------------------------------
	engine = Engine.create();
   	world = engine.world;

	let ground = Bodies.rectangle(width/2, 0, width, 20, { isStatic: true });
	World.add(world, ground);
	engine.world.gravity.y = -0.8;
	// ----------------------------------

	await initiateTracking();
	await loadASLModel();
}

async function loadASLModel() {
	print("trying to load model");
    try {
	    ASLmodel = await tf.loadLayersModel('indexeddb://ASLmodel');
		 // await tf.io.removeModel('indexeddb://ASLmodel');
		 // const models = await tf.io.listModels();
		 // console.log(models);
	    print("model loaded from IndexedDB!");
    } catch (err) {
	    print("no saved model found, calling trainASLmodel");
		await trainASLmodelCNN();
		// await trainASLmodelMLP();
		print("trainASLmodel should be complete");
    }
}

async function trainASLmodelCNN() {
	const xs = [];
	const ys = []; // the class (predict)
	dataset.forEach(eachAlphabet => {
		// (an object that contains all instances for each class as objects)
		// then each instances in a obj with a class and features
		Object.values(eachAlphabet).forEach(ins => {
			// each instance is an array of 21 arrays
			// flatten so its one array of length 42
			// print(ins.features);
			xs.push((ins.features).flat());
			//so xs is a nested array
			
			//each instance has a ys like [0,1,0,0,...] (this represents "b")
			const y = new Array(LABELS.length).fill(0);
			y[LABELS.indexOf(ins.class)] = 1;
			ys.push(y);
		});
	});
	const xsTensor = tf.tensor2d(xs);
	const ysTensor = tf.tensor2d(ys);
	print("tensors ready")
		
	ASLmodel = tf.sequential();

	//train
}

async function trainASLmodelMLP() {
  const xs = [];
  const ys = []; // the class (predict)
  dataset.forEach(eachAlphabet => {
	  // (an object that contains all instances for each class as objects)
	  // then each instances in a obj with a class and features
	  Object.values(eachAlphabet).forEach(ins => {
		  // each instance is an array of 21 arrays
		  // flatten so its one array of length 42
		  // print(ins.features);
		  xs.push((ins.features).flat());
		  //so xs is a nested array
		  
		  //each instance has a ys like [0,1,0,0,...] (this represents "b")
		  const y = new Array(LABELS.length).fill(0);
		  y[LABELS.indexOf(ins.class)] = 1;
		  ys.push(y);
	  });
  });

	// print(xs[0])
	// print("xs[0]", xs[0]);
	// print("ys[0]", ys[0]);

	//convert to tensors
	const xsTensor = tf.tensor2d(xs);
	const ysTensor = tf.tensor2d(ys);
	print("tensors ready")
		
	ASLmodel = tf.sequential();
	// dense layer means every input neuron is connected to every output neuron
	// 42 = 21 hand landmarks * 2 coords each landmark(x, y)
	// using 64 neurons to learn patterns
	// use each neurons to compute one weighted sum from 42 features
	// relu = Rectified Linear Unit, f(x) = max(0, x)
	// outputs a list of relu(weighted sum) then gets passed to the next layer
	ASLmodel.add(tf.layers.dense({inputShape: 42, units: 64, activation: 'relu'}));
	ASLmodel.add(tf.layers.dense({units: 32, activation: 'relu'}));
	ASLmodel.add(tf.layers.dense({units: LABELS.length, activation: 'softmax'}));
	print("model added")
	
	// adam decides how much to change each weight 
	// based on the gradient of the loss
	// measure error
	ASLmodel.compile(
		{optimizer: 'adam', 
		loss: 'categoricalCrossentropy', 
		metrics: ['accuracy']}
	);
	print("model compiled")
	
	//actual traininggggg!!!
	await ASLmodel.fit(xsTensor, ysTensor, {
		epochs: 10,
		batchSize: 64,
		callbacks: { 
			onEpochEnd: (epoch, logs) => 
				print(`Epoch ${epoch}: ${logs.loss}`) }
	});
	print("Model trained, saving to IndexedDB...");
	await ASLmodel.save('indexeddb://ASLmodel');
	print("Model saved!");
}


function onResults(results) {
	// print(results); // an array of 21 objects
   if (!results) {
	   print("no results");
	   return;
   }
   const features = [];
	results.forEach(obj => {
		features.push(obj.x);
		features.push(obj.y);
	})
	// print(features);
  predictGesture(features);
}

async function predictGesture(features) {
	// print(features)
   if (!ASLmodel) {
	   print("no ASLmodel");
	   return;
   }
   const input = tf.tensor2d([features]);
   const prediction = ASLmodel.predict(input);
   const predArray = await prediction.array();
	// print(predArray);
   const maxIdx = predArray[0].indexOf(Math.max(...predArray[0]));
	// print(maxIdx)
   currentLabel = LABELS[maxIdx];
	if (letters.length > 200) {
		letters.splice(0,1);
	}
	let palmX = random(width);
	let palmY = random(height);
	if (trackingConfig.doAcquireHandLandmarks) {
		if (handLandmarks && handLandmarks.landmarks) {
	      const nHands = handLandmarks.landmarks.length;
	      if (nHands > 0) {
			  for (let i = 0; i < nHands; i++) {
				  let whichHand = handLandmarks.handednesses[i];
				  if (whichHand == "Right") {
					   let joints = handLandmarks.landmarks[i];
					   palmX = (1 - joints[MIDDLE_FINGER_MCP].x) * width;
						palmY = joints[MIDDLE_FINGER_MCP].y * height;
				  }
				}
			}
		}
	}
	letters.push(new Letter(palmX, palmY, 
							currentLabel, random(10,30)));
	// print(label)
	
   input.dispose();
   prediction.dispose();
}

function draw() {
   background(0);
	// "#EB6534"
   // drawVideoBackground();
   drawHandPoints();

	Engine.update(engine);

	if (trackingConfig.doAcquireHandLandmarks) {
		if (handLandmarks && handLandmarks.landmarks) {
	      const nHands = handLandmarks.landmarks.length;
	      if (nHands > 0) {
			  for (let i = 0; i < nHands; i++) {
				  let whichHand = handLandmarks.handednesses[i];
				  if (whichHand == "Right") {
					  let results = handLandmarks.landmarks[i];
					  onResults(results);
				  }
			  }
		  }
		}
	}
	for (let l of letters) {
		l.show();
	}
}


// let frameRateAvg = 60.0; 
// function drawDiagnosticInfo() {
//   noStroke();
//   fill("black");
//   textSize(12); 
// 	frameRateAvg = 0.98*frameRateAvg + 0.02*frameRate();
//   text("FPS: " + nf(frameRateAvg,1,2), 40, 30);
// }