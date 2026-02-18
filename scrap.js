
function updateDelaunayTriangulation() {
	if (letters.length < 2) return;
	const points = letters.map(l => [l.body.position.x, l.body.position.y]);
	for (let l of letters) {
		l.neighbors = [];
	}
	
	delaunay = d3.Delaunay.from(points);
	
	// each triangle has 3 edges, we extract all edges and map them to letters
	const neighbors = new Map();
	for (let i = 0; i < letters.length; i++) {
		neighbors.set(i, new Set());
	}
	
	// iterate through all triangles in the triangles array
	const triangles = delaunay.triangles;
	for (let i = 0; i < triangles.length; i += 3) {
		const p1 = triangles[i];
		const p2 = triangles[i + 1];
		const p3 = triangles[i + 2];

		//add edges 
		neighbors.get(p1).add(p2);
		neighbors.get(p2).add(p1);
		neighbors.get(p2).add(p3);
		neighbors.get(p3).add(p2);
		neighbors.get(p1).add(p3);
		neighbors.get(p3).add(p1);
	}
	
	for (let i = 0; i < letters.length; i++) {
		for (let neighborIdx of neighbors.get(i)) {
			letters[i].neighbors.push(letters[neighborIdx]);
		}
	}
	// store current positions for displacement tracking
	lastTriangulationPositions = points.map(p => [...p]);
}


	// reset triangles 
	framesSinceTriangulation++;
	if (framesSinceTriangulation >= TRIANGULATION_UPDATE_FRAMES || hasExceededDisplacementThreshold()) {
		updateDelaunayTriangulation();
		framesSinceTriangulation = 0;
	}

	// reset lloyd 
	framesSinceLloyd++;
	if (framesSinceLloyd >= LLOYD_UPDATE_FRAMES) {
		applyLloydRelaxation();
		framesSinceLloyd = 0;
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


	// ------------ OLD LETTER CLASS ------------

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