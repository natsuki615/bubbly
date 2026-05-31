async function loadASLModel() {
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
	const xs = [];
	const ys = []; // the class (predict)
    // TODO: revise model for normalized inputs 
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
		epochs: 15,
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
	// predict every 2 frames
	if (frameCount % 2 === 0) {
        predictGesture(features);
    }
	// predictGesture(features);
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
	if (letters.length > 1000) {
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
	let letter_box_radius = random(10,20);
	// Create a new free letter particle
	let particle = new Letter(palmX, palmY, currentLabel, letter_box_radius);
	letters.push(particle);
	input.dispose();
	prediction.dispose();
}
