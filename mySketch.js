// p5.js interface to Google MediaPipe Landmark Tracking
// Combines face, hands, and bodies into one tracker.
// See https://mediapipe-studio.webapps.google.com/home
// Uses p5.js v.1.11.11 + MediaPipe v.0.10.22-rc.20250304
// By Golan Levin, revised as of 10/21/2025

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
let wordTrie;
let words = [];

const LABELS = "abcdefghijklmnopqrstuvwxyz".split("");

const Engine = Matter.Engine,
      World = Matter.World,
      Bodies = Matter.Bodies,
	  Constraint = Matter.Constraint;

let engine;
let world;
let seeds = [];

class Letter {
	constructor(x, y, l, s){
		this.l = l;
		this.s = s;
		this.body = Bodies.circle(x, y, s, { 
			restitution: 0.5, 
			friction: 0.2 
		});
      	World.add(world, this.body);
		this.neighbors = []; 
		this.connectedTo = null; 
		this.partOfWord = null; 
		this.constraints = [];
	}
	
	show() {
		let pos = this.body.position;
		let angle = this.body.angle;
	   	push();
		translate(pos.x, pos.y);
		rotate(angle);
		if (this.partOfWord && this.partOfWord.isComplete()){
			fill("#ffff00");
		} else {fill("#23f758"); }
		
		noStroke();
		textSize(this.s);
		text(this.l, 0, 0);
		pop();
	}
	
	addNeighbors(){
		for (let l of letters) {
			if (this.neighbors.includes(l) || l === this) continue;
			let pos = l.body.position;
			let d = dist(this.body.position.x, this.body.position.y, pos.x, pos.y);
			if (d < this.s * 2){ 
				this.neighbors.push(l);
				l.neighbors.push(this);
			}
		}
	}
	
	cleanNeighbors(){
		for (let i = this.neighbors.length-1; i>=0; i--) {
			let n = this.neighbors[i];
			let pos = n.body.position;
			let d = dist(this.body.position.x, this.body.position.y, pos.x, pos.y);
			if (d > this.s * 2.5){
				this.neighbors.splice(i, 1);
			}
		}
	}
	
	tryConnect(){
		if (this.partOfWord) return;
		for (let n of this.neighbors){
			if (n.partOfWord) continue;
			if (wordTrie && this.l in wordTrie && n.l in wordTrie[this.l]){
				let word = new Word([this, n]);
				words.push(word);
				return;
			}
		}
	}

	destroy() {
		for (let c of this.constraints) {
			World.remove(world, c);
		}
		World.remove(world, this.body);
		this.constraints = [];
	}
}

class Word {
	constructor(letterArray){
		this.letters = letterArray; 
		this.suffix = null; 
		// this.constraints = [];

		for (let l of this.letters){
			l.partOfWord = this;
		}
		for (let i = 0; i < this.letters.length-1; i++) {
			let constraint = Constraint.create({
				bodyA: this.letters[i].body,
				bodyB: this.letters[i+1].body,
				length: this.letters[i].s*1.5,
				stiffness: 0.5,
				damping: 0.1
			});
			World.add(world, constraint);
			// this.constraints.push(constraint);
			this.letters[i].constraints.push(constraint);
			this.letters[i+1].constraints.push(constraint);
		}
		this.updateSuffix();
	}
	
	updateSuffix(){
		if (!wordTrie) return;
		let curr = wordTrie;
		for (let letter of this.letters){
			if (letter.l in curr){
				curr = curr[letter.l];
			} else {
				this.suffix = null;
				return;
			}
		}
		this.suffix = curr;
	}
	
	getText(){
		return this.letters.map(l => l.l).join('');
	}
	
	isComplete(){
		return this.suffix && this.suffix.isWord === true;
	}
	
	show(){
		noFill();
		if (this.isComplete()){
			stroke("#ffff00"); 
			strokeWeight(4);
		} else {
			stroke("#23f758");
			strokeWeight(1);
		}
		for (let i = 0; i < this.letters.length-1; i++){
			let pos1 = this.letters[i].body.position;
			let pos2 = this.letters[i + 1].body.position;
			line(pos1.x, pos1.y, pos2.x, pos2.y);
		}
	}
	
	tryExtend(){
		if (!this.suffix) return;
		let lastLetter = this.letters[this.letters.length-1];
		for (let n of lastLetter.neighbors){
			if (n.partOfWord) continue;
			if (n.l in this.suffix){
				this.letters.push(n);
				n.partOfWord = this;

				let constraint = Constraint.create({
					bodyA: lastLetter.body,
					bodyB: n.body,
					length: lastLetter.s * 1.5,
					stiffness: 0.5,
					damping: 0.1
				});
				World.add(world, constraint);
				
				lastLetter.constraints.push(constraint);
				n.constraints.push(constraint);

				this.updateSuffix();
				return;
			}
		}
	}
}

function preload() {
	for (let l of LABELS) {
		let data = loadJSON(`./alphabets/${l}_fixed.json`);
		allData = allData.concat(data);
	} 
	dataset = allData;
	// print(dataset);
	preloadTracker();
	
	wordTrie = loadJSON('common_trie.json');
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

	engine = Engine.create();
    world = engine.world;

    let wallThickness = 20;

	let ground = Bodies.rectangle(width/2, height, width, wallThickness, { isStatic: true });
	let ceiling = Bodies.rectangle(width/2, 0, width, wallThickness, { isStatic: true });
	let leftWall = Bodies.rectangle(0, height/2, wallThickness, height, { isStatic: true });
	let rightWall = Bodies.rectangle(width, height/2, wallThickness, height, { isStatic: true });
    World.add(world, [ground, ceiling, leftWall, rightWall]);
	engine.world.gravity.y = -0.8;

	await initiateTracking();
	await loadASLModel();
}

async function loadASLModel() {
	// print("trying to load model");
    try {
	    ASLmodel = await tf.loadLayersModel('indexeddb://ASLmodel');
		// await tf.io.removeModel('indexeddb://ASLmodel');
		// const models = await tf.io.listModels();
		// console.log(models);
	    print("model loaded from IndexedDB!");
    } catch (err) {
	    print("no saved model found, calling trainASLmodel");
		await trainASLmodel();
		print("trainASLmodel should be complete");
    }
}

async function trainASLmodel() {
	// print("trainASLmodel");
	const xs = [];
	const ys = []; // the class (predict)
	// dataset.forEach(eachAlphabet => {
	// 	Object.values(eachAlphabet).forEach(ins => {
	// 		// each instance has shape
	// 		// {
	// 		// 		label: "A"
	// 		//      features: [x1, y1, x2, y2, ... ] length 63
	// 		// } 
	// 		// print(ins.features);
	// 		xs.push((ins.features).flat());
	// 		//each instance has a ys like [0,1,0,0,...] (this represents "b")
	// 		const y = new Array(LABELS.length).fill(0);
	// 		y[LABELS.indexOf(ins.label)] = 1;
	// 		ys.push(y);
	// 	});
	// });
	// --------------- OLD ---------------
	dataset.forEach(eachAlphabet => {
		// eachAlphabet is an object that contains all instances for that letter)
		// iterate through to get individual instances 
		Object.values(eachAlphabet).forEach(ins => {
			// ins = {
	 		// 		label: "A"
			//      features: [x1, y1, x2, y2, ... ] length 42
			// }
			xs.push((ins.features).flat());
			//each instance has a ys like [0,1,0,0,...] (this represents "b")
			const y = new Array(LABELS.length).fill(0);
			y[LABELS.indexOf(ins.class)] = 1;
			ys.push(y);
		});
	});


	// print("xs[0]", xs[0]);
	// print("ys[0]", ys[0]);

	//convert to tensors
	const xsTensor = tf.tensor2d(xs);
	const ysTensor = tf.tensor2d(ys);
	print("tensors ready");
	
   	ASLmodel = tf.sequential();
	// dense layer means every input neuron is connected to every output neuron
	// 42 = 21 hand landmarks * 3 coords each landmark(x, y, z)
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
	if (!results) {
		print("no results");
		return;
	}
   	const features = [];
	results.forEach(obj => {
		features.push(obj.x);
		features.push(obj.y);
		// features.push(obj.z);
	})	
  	predictGesture(features);
}

async function predictGesture(features) {
	if (!ASLmodel) {
		print("no ASLmodel");
		return;
	}
	
	const input = tf.tensor2d([features]);
	const prediction = ASLmodel.predict(input);
	const predArray = await prediction.array();
	const maxIdx = predArray[0].indexOf(Math.max(...predArray[0]));
	currentLabel = LABELS[maxIdx];
	if (letters.length > 200) {
		letters.splice(0, 1);
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
	let letter_box_radius = random(10,30);
	let newLetter = new Letter(palmX, palmY, currentLabel, letter_box_radius, [])
	newLetter.addNeighbors();
	letters.push(newLetter);
	
	input.dispose();
	prediction.dispose();
}

function draw() {
   	background(0);
	// "#EB6534"
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

	// clean up invalid words (the lines after letters are destroyed)
	for (let i = words.length - 1; i >= 0; i--) {
		let word = words[i];
		let hasDestroyedLetter = false;
		for (let letter of word.letters) {
			if (!letters.includes(letter)) {
				hasDestroyedLetter = true;
				break;
			}
		}
		if (hasDestroyedLetter) {
			for (let letter of word.letters) {
				if (letters.includes(letter)) {
					letter.partOfWord = null;
				}
			}
			words.splice(i, 1);
		}
	}

	for (let l of letters) {
		l.cleanNeighbors();
		l.addNeighbors();
		l.tryConnect();
		l.show();
	}
	for (let w of words) {
		w.tryExtend();
		w.show();
	}
}