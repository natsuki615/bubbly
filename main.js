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

// delaunay triangulation
// let framesSinceTriangulation = 0;
// const TRIANGULATION_UPDATE_FRAMES = 12; 
// const DISPLACEMENT_THRESHOLD = 5; // pixels
// let lastTriangulationPositions = []; // store positions from last triangulation
// let delaunay = null;

// lloyd relaxation 
const LLOYD_STRENGTH = 0.005; // pull toward centroid (0-1)
const LLOYD_UPDATE_FRAMES = 12;
let framesSinceLloyd = 0;


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
			this.letters[i].show();
		}
		this.letters[this.letters.length-1].show();
	}
	
	tryExtend(){
		// right now words don't extend after they're created

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

// cpu based particle class for free letters
class Letter {
	constructor(x, y, label, size) {
		this.label = label;
		this.size = size;
		this.pos = createVector(x, y);
		this.vel = createVector(0, 0);
		this.acc = createVector(0, 0);

		this.neighbors = []; 
		this.connectedTo = null; 
		this.partOfWord = null; 
	}
	
	applyForce(force) {
		this.acc.add(force);
	}
	
	update() {
		// gravity
		this.applyForce(createVector(0, -0.10));
		
		// force with noise
		let angle = noise(this.pos.x * 0.002, this.pos.y * 0.002, frameCount * 0.01) * TWO_PI;
		let wander = p5.Vector.fromAngle(angle).mult(0.2);
		this.applyForce(wander);
		
		// density based repulsion
		for (let other of letters) {
			if (other === this) continue;
			let d = dist(this.pos.x, this.pos.y, other.pos.x, other.pos.y);
			let minDist = (this.size + other.size) * 1.5;
			if (d < minDist * 3) {
				let force = p5.Vector.sub(this.pos, other.pos);
				force.normalize();
				force.mult(0.05 / (d + 1));
				this.applyForce(force);
			}
		}
		// TODO: revisit forces and computation
		
		// physics integration
		this.vel.add(this.acc);
		this.vel.mult(0.97); // damping
		this.pos.add(this.vel);
		this.acc.mult(0);
		
		// boundary wrapping
		if (this.pos.x < 0) this.pos.x = width;
		if (this.pos.x > width) this.pos.x = 0;
		if (this.pos.y < 0) this.pos.y = height;
		if (this.pos.y > height) this.pos.y = 0;
	}
	
	show() {
		push();
		translate(this.pos.x, this.pos.y);
		fill("#23f758");
		noStroke();
		textSize(this.size);
		text(this.label, 0, 0);
		pop();
	}
	getNeighbors() {
		// TODO: use spatial hashing here fo efficiency
	}
	checkWordConnection(wordTrie) {
		// TODO: go through this.neighbors and connect
		
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
	// print("Backend:", tf.getBackend());

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

// lloyd relaxation for letters in words
function applyLloydRelaxation() {
	for (let word of words) {
		if (word.letters.length < 2) continue;
		
		// voronoi diagram for just letters in a word
		const wordPoints = word.letters.map(l => [l.body.position.x, l.body.position.y]);
		const wordDelaunay = d3.Delaunay.from(wordPoints);
		const wordVoronoi = wordDelaunay.voronoi([0, 0, width, height]);
		
		// for each letter in the word, pull it toward its cell's centroid
		for (let i = 0; i < word.letters.length; i++) {
			const letter = word.letters[i];
			const cellPolygon = wordVoronoi.cellPolygon(i);
			
			if (!cellPolygon) continue;
			
			// compute centroid of the voronoi cell
			let centroidX = 0
			let centroidY = 0;
			for (let point of cellPolygon) {
				centroidX += point[0];
				centroidY += point[1];
			}
			centroidX /= cellPolygon.length;
			centroidY /= cellPolygon.length;
			
			// apply force toward centroid
			const dx = centroidX - letter.body.position.x;
			const dy = centroidY - letter.body.position.y;
			
			letter.body.velocity.x += dx * LLOYD_STRENGTH;
			letter.body.velocity.y += dy * LLOYD_STRENGTH;
		}
	}
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

	for (let p of letters) {
		p.update();
		p.show();
		p.checkWordConnection(wordTrie);
	}
	
	// lloyd relaxation to word letters
	framesSinceLloyd++;
	if (framesSinceLloyd >= LLOYD_UPDATE_FRAMES) {
		applyLloydRelaxation();
		framesSinceLloyd = 0;
	}
	
	for (let w of words) {
		w.tryExtend();
		w.show();
	}

	// ------------ DRAW TRIANGLES ------------
	// if (delaunay && delaunay.triangles) {
	// 	stroke(100, 100, 255);
	// 	strokeWeight(1);
	// 	const triangles = delaunay.triangles;
	// 	for (let i = 0; i < triangles.length; i += 3) {
	// 		const p1 = letters[triangles[i]].body.position;
	// 		const p2 = letters[triangles[i + 1]].body.position;
	// 		const p3 = letters[triangles[i + 2]].body.position;
			
	// 		line(p1.x, p1.y, p2.x, p2.y);
	// 		line(p2.x, p2.y, p3.x, p3.y);
	// 		line(p3.x, p3.y, p1.x, p1.y);
	// 	}
	// }
}