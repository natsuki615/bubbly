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

let grid = new SpatialHash(); // you can define cellSize

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

// spring constant
let K = 0.1;

class Word {
	constructor(letterArray){
		this.letters = letterArray; 
		this.suffix = null; 
		this.letter_neighbors = [];
		this.word_neighbors = [];
		[this.x, this.y] = this.getCenter(this.letters);
		this.pos = createVector(this.x, this.y);
		for (let l of this.letters){
			l.partOfWord = this;
		}
		this.updateSuffix();

		// TODO: words dont go off screen but instead wobble in their own position?
	}
	
	getCenter(arr) {
		let sumX = 0;
		let sumY = 0; 
		for (let l of arr) {
			sumX += l.pos.x;
			sumY += l.pos.y;
		}
		let cx = sumX / this.letters.length;
		let cy = sumY / this.letters.length;
		return [cx, cy];
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
		let bgColor = null;
		if (this.isComplete()){
			stroke("#ffff00"); 
			bgColor = color(255, 255, 0);
			strokeWeight(3);
		} else {
			stroke("#23f758");
			bgColor = color(15, 153, 49, 80);
			strokeWeight(1);
		}
		for (let i = 0; i < this.letters.length-1; i++){
			let curr = this.letters[i];
			let next = this.letters[i+1];
			curr.highlight = true;
			curr.show(bgColor);
			this.updateSpring(curr, next);
		}
		this.letters[this.letters.length-1].show(bgColor);
	}

	getNeighborWords() {
		this.word_neighbors = [];
		let raw = grid.findNeighbors(this.x, this.y, grid.cellSize);
		for (let r of raw) {
			let letter = letters[r.index];
			if (letter.partOfWord && letter.partOfWord !== this) {
				let otherWord = letter.partOfWord;

				if (!this.word_neighbors.includes(otherWord)) {
					this.word_neighbors.push(otherWord);
				}
			}
		}
		// print("word neighbor length: " + this.word_neighbors.length);
		let count = 0;
		let merge = [];
		for (let n of this.word_neighbors) {
			if (n.getText() == this.getText()) {
				// print("has dups");
				count++;
				merge.push(n);
			}
		}
		if (count > 2) {
			let [cx, cy] = this.getCenter(merge);
			let target = createVector(cx, cy);
			this.mergeWords(merge, target);
		}
		merge = [];
	}

	mergeWords(wordsArr, target) {
		// TODO: fix merging force(rn they all go towards the bottom right corner)
		print("MERGE WORDS");
		let isClose = false;
		for (let w of wordsArr) {
			for (let l of w.letters) {
				let curr = l.pos;
				let force = p5.Vector.sub(target, curr);
				let dist = force.mag();

				if (dist < 10) {isClose = true};

				force.normalize(); // now this is just direction of the force
				let k = 0.05;
				force.mult(k*dist);
				l.applyForce(force);
			}
		}
		if (isClose) {
			for (let w of wordsArr) {
				let idx = words.indexOf(w);
				if (idx !== -1) {
					words.splice(idx, 1);
				}
			}
		}
		// TODO: increase word size
	}

	tryExtend() {
		if (!this.suffix) return;
		let lastLetter = this.letters[this.letters.length-1];
		for (let n of lastLetter.neighbors){
			if (n.partOfWord) {
				// TODO: does the two parts combine?
				// make new word object, 
				// rebuild constraints,
				// update suffix,
				// remove the old word from words[]
				return;
			} else if (n.l in this.suffix){
				this.letters.push(n);
				n.partOfWord = this;
				this.updateSuffix();
				return;
			}
		}
	}

	updateSpring(curr, next) {
		let force = p5.Vector.sub(next.pos, curr.pos);
		let dist = force.mag();
		let restlen = 40;
		let stretch = dist - restlen;
		force.normalize();
		if (dist < 50) {
			force.mult(-K*stretch);
		}
		curr.applyForce(force);
		next.applyForce(p5.Vector.mult(force, -1));
		line(curr.pos.x, curr.pos.y, next.pos.x, next.pos.y);
	}
}

// cpu based particle class for free letters
class Letter {
	constructor(x, y, label, size) {
		this.l = label;
		this.size = size;
		this.pos = createVector(x, y);
		this.vel = createVector(0, 0);
		this.acc = createVector(0, 0);

		this.neighbors = []; 
		this.connectedTo = null; 
		this.partOfWord = null;  // this is a word object
		this.highlight = false;
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
		
		// density based repulsion using spatial hash!
		if (this.neighbors.length > 10) { // more than 20 cells are within 100 pixels
			for (let other of this.neighbors) {
				let force = p5.Vector.sub(this.pos, other.pos);
				force.normalize();
				force.mult(0.01*this.size);
				this.applyForce(force);
			}
		}

		// TODO: revisit forces and computation
		
		// physics integration
		this.vel.add(this.acc);
		this.vel.mult(0.95); // damping
		this.pos.add(this.vel);
		this.acc.mult(0);
		
		// boundary wrapping
		// if (this.pos.x < 0) this.pos.x = width;
		// if (this.pos.x > width) this.pos.x = 0;
		// if (this.pos.y < 0) this.pos.y = height;
		// if (this.pos.y > height) this.pos.y = 0;
	}
	
	show(bgColor) {
		if (this.highlight) {
			push();
			// translate(this.pos.x, this.pos.y);
			fill(bgColor);
			noStroke();
			circle(this.pos.x, this.pos.y, this.size+10);
			pop();

			push();
			translate(this.pos.x, this.pos.y);
			fill("#000000");
			noStroke();
			textSize(this.size);
			text(this.l, 0, 0);
			pop();
		} else {
			push();
			translate(this.pos.x, this.pos.y);
			fill("#23f758");
			noStroke();
			textSize(this.size);
			text(this.l, 0, 0);
			pop();
		}
	}
	getNeighbors() {
		// use spatial hashing for efficiency
		this.neighbors = [];
		let raw = grid.findNeighbors(this.pos.x, this.pos.y, grid.cellSize);
		for (let r of raw) {
			let other = letters[r.index];
			if (other !== this) {
				this.neighbors.push(other);
			}
		}
	}
	combine() {
		let thisTrie = wordTrie[this.l];
		for (let i = 0; i < this.neighbors.length; i++) {
			let n = this.neighbors[i];
			if (!this.partOfWord && !n.partOfWord) {
				if (thisTrie && n.l in thisTrie) {
					let newWord = new Word([this, n]);
					words.push(newWord);
					this.partOfWord = newWord;
					n.partOfWord = newWord;
					this.highlight = true;
					n.highlight = true;
					// print("combining " + this.l + " and " + n.l);
					return;
				} 
				// other direction
				let nTrie = wordTrie[n.l];
				if (nTrie && this.l in nTrie) {
					let newWord = new Word([n, this]);
					words.push(newWord);
					this.partOfWord = newWord;
					n.partOfWord = newWord;
					this.highlight = true;
					n.highlight = true;
					// print("combining " + n.l + " and " + this.l);
					return;
				}
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
	// print("Backend:", tf.getBackend());

	myCapture = createCapture(VIDEO);
	myCapture.size(160,120); 
	myCapture.hide();

	await initiateTracking();
	await loadASLModel();
}

// lloyd relaxation for letters in words
function applyLloydRelaxation() {
	for (let word of words) {
		if (word.letters.length < 2) continue;
		
		// voronoi diagram for just letters in a word
		const wordPoints = word.letters.map(l => [l.pos.x, l.pos.y]);
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
			const dx = centroidX - letter.pos.x;
			const dy = centroidY - letter.pos.y;
			
			letter.vel.x += dx * LLOYD_STRENGTH;
			letter.vel.y += dy * LLOYD_STRENGTH;
		}
	}
}

function draw() {
   	background(0);
	// "#EB6534"
   	drawHandPoints();

	// Engine.update(engine);

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
	// for (let i = words.length - 1; i >= 0; i--) {
	// 	let word = words[i];
	// 	let hasDestroyedLetter = false;
	// 	for (let letter of word.letters) {
	// 		if (!letters.includes(letter)) {
	// 			hasDestroyedLetter = true;
	// 			break;
	// 		}
	// 	}
	// 	if (hasDestroyedLetter) {
	// 		for (let letter of word.letters) {
	// 			if (letters.includes(letter)) {
	// 				letter.partOfWord = null;
	// 			}
	// 		}
	// 		words.splice(i, 1);
	// 	}
	// }

	// print(letters.length);
	for (let p of letters) {
		p.update();
		p.show("#23f758");
	}
	grid.clear();
	for (let i = 0; i < letters.length; i++) {
		let p = letters[i];
		grid.insert(i, p.pos.x, p.pos.y);
	}
	for (let p of letters) {
		p.getNeighbors();
		p.combine();
	}
	
	// lloyd relaxation to word letters
	framesSinceLloyd++;
	if (framesSinceLloyd >= LLOYD_UPDATE_FRAMES) {
		applyLloydRelaxation();
		framesSinceLloyd = 0;
	}

	// print(words);
	
	for (let w of words) {
		// w.tryExtend();
		w.show();
		w.getNeighborWords();
	}

	// ------------ DRAW TRIANGLES ------------
	// if (delaunay && delaunay.triangles) {
	// 	stroke(100, 100, 255);
	// 	strokeWeight(1);
	// 	const triangles = delaunay.triangles;
	// 	for (let i = 0; i < triangles.length; i += 3) {
	// 		const p1 = letters[triangles[i]].pos;
	// 		const p2 = letters[triangles[i + 1]].pos;
	// 		const p3 = letters[triangles[i + 2]].pos;
			
	// 		line(p1.x, p1.y, p2.x, p2.y);
	// 		line(p2.x, p2.y, p3.x, p3.y);
	// 		line(p3.x, p3.y, p1.x, p1.y);
	// 	}
	// }
}